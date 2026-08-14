class ConvertDetectionOutput final
  : public OpConversionPattern<DetectionOutputOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    DetectionOutputOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const final {
    Location loc = op.getLoc();
    auto locationType = cast<RankedTensorType>(adaptor.getLocation().getType());
    auto outputType = cast<RankedTensorType>(op.getOutput().getType());
    const int64_t numPrior = locationType.getNumElements() / 4;
    const int64_t numClass = op.getNumClass();
    const int64_t classCapacity =
      std::min(static_cast<int64_t>(op.getNmsTopK()), numPrior);
    const int64_t outputCapacity = outputType.getShape()[0];
    auto f32 = rewriter.getF32Type();
    auto i64 = rewriter.getI64Type();
    auto index = rewriter.getIndexType();
    auto classScoreType = RankedTensorType::get({classCapacity}, f32);
    auto classIndexType = RankedTensorType::get({classCapacity}, i64);
    auto globalScoreType = RankedTensorType::get({outputCapacity}, f32);
    auto globalIndexType = RankedTensorType::get({outputCapacity}, i64);

    auto flatten = [&](Value value, int64_t count) {
      return reshapeValue(
        rewriter,
        loc,
        value,
        RankedTensorType::get(
          {count}, cast<ShapedType>(value.getType()).getElementType()));
    };
    Value locations = flatten(
      applyLowPrecisionBoundary(rewriter, loc, op, adaptor.getLocation()),
      numPrior * 4);
    Value confidences = flatten(
      applyLowPrecisionBoundary(rewriter, loc, op, adaptor.getConfidence()),
      numPrior * numClass);
    Value priors = flatten(
      applyLowPrecisionBoundary(rewriter, loc, op, adaptor.getPriorbox()),
      numPrior * 8);

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value priorLimit = rewriter.create<arith::ConstantIndexOp>(loc, numPrior);
    Value classLimit =
      rewriter.create<arith::ConstantIndexOp>(loc, classCapacity);
    Value classLast =
      rewriter.create<arith::ConstantIndexOp>(loc, classCapacity - 1);
    Value outputLimit =
      rewriter.create<arith::ConstantIndexOp>(loc, outputCapacity);
    Value zero = constantF32(rewriter, loc, 0.0F);
    Value half = constantF32(rewriter, loc, 0.5F);
    Value minusInf =
      constantF32(rewriter, loc, -std::numeric_limits<float>::infinity());
    Value confidenceThreshold =
      constantF32(rewriter, loc, op.getConfidenceThreshold().convertToFloat());
    Value backgroundThreshold = constantF32(
      rewriter, loc, 1.0F - op.getConfidenceThreshold().convertToFloat());
    Value nmsThreshold =
      constantF32(rewriter, loc, op.getNmsThreshold().convertToFloat());

    Value globalScores = splat(rewriter, loc, globalScoreType, minusInf);
    Value globalIndices =
      splat(rewriter,
            loc,
            globalIndexType,
            rewriter.create<arith::ConstantIntOp>(loc, 0, 64));
    Value globalLabels =
      splat(rewriter,
            loc,
            globalIndexType,
            rewriter.create<arith::ConstantIntOp>(loc, 0, 64));
    Value globalCount = c0;

    for (int64_t classNumber = 1; classNumber < numClass; ++classNumber) {
      Value classScores = splat(rewriter, loc, classScoreType, minusInf);
      Value classIndices =
        splat(rewriter,
              loc,
              classIndexType,
              rewriter.create<arith::ConstantIntOp>(loc, 0, 64));

      auto select = rewriter.create<scf::ForOp>(
        loc,
        c0,
        priorLimit,
        c1,
        ValueRange{classScores, classIndices},
        [&](OpBuilder& b, Location bodyLoc, Value prior, ValueRange state) {
          Value classStride =
            b.create<arith::ConstantIndexOp>(bodyLoc, numClass);
          Value base = b.create<arith::MulIOp>(bodyLoc, prior, classStride);
          Value classOffset =
            b.create<arith::ConstantIndexOp>(bodyLoc, classNumber);
          Value scoreAt = b.create<arith::AddIOp>(bodyLoc, base, classOffset);
          Value score =
            b.create<tensor::ExtractOp>(bodyLoc, confidences, scoreAt);
          Value background =
            b.create<tensor::ExtractOp>(bodyLoc, confidences, base);
          Value foregroundPass = b.create<arith::CmpFOp>(
            bodyLoc, arith::CmpFPredicate::OGT, score, confidenceThreshold);
          Value backgroundPass =
            b.create<arith::CmpFOp>(bodyLoc,
                                    arith::CmpFPredicate::ULT,
                                    background,
                                    backgroundThreshold);
          Value accepted =
            b.create<arith::AndIOp>(bodyLoc, foregroundPass, backgroundPass);
          Value acceptedScore =
            b.create<arith::SelectOp>(bodyLoc, accepted, score, minusInf);
          Value priorI64 = b.create<arith::IndexCastOp>(bodyLoc, i64, prior);
          auto inserted = insertSorted(b,
                                       bodyLoc,
                                       state[0],
                                       state[1],
                                       acceptedScore,
                                       priorI64,
                                       classLimit,
                                       c0,
                                       c1);
          b.create<scf::YieldOp>(bodyLoc,
                                 ValueRange{inserted.scores, inserted.indices});
        });
      classScores = select.getResult(0);
      classIndices = select.getResult(1);

      Value picked = splat(rewriter,
                           loc,
                           classIndexType,
                           rewriter.create<arith::ConstantIntOp>(loc, 0, 64));
      Value pickedCount = c0;
      Value label = rewriter.create<arith::ConstantIntOp>(loc, classNumber, 64);
      auto suppress = rewriter.create<scf::ForOp>(
        loc,
        c0,
        classLimit,
        c1,
        ValueRange{picked,
                   pickedCount,
                   globalScores,
                   globalIndices,
                   globalLabels,
                   globalCount},
        [&](OpBuilder& b, Location bodyLoc, Value slot, ValueRange state) {
          Value score = b.create<tensor::ExtractOp>(bodyLoc, classScores, slot);
          Value valid = b.create<arith::CmpFOp>(
            bodyLoc, arith::CmpFPredicate::OGT, score, minusInf);
          Value candidateI64 =
            b.create<tensor::ExtractOp>(bodyLoc, classIndices, slot);
          Value candidate =
            b.create<arith::IndexCastOp>(bodyLoc, index, candidateI64);
          auto candidateBox =
            decode(b, bodyLoc, locations, priors, candidate, numPrior, half);
          auto overlaps = b.create<scf::ForOp>(
            bodyLoc,
            c0,
            classLimit,
            c1,
            ValueRange{valid},
            [&](OpBuilder& inner,
                Location innerLoc,
                Value pickedSlot,
                ValueRange keepState) {
              Value active = inner.create<arith::CmpIOp>(
                innerLoc, arith::CmpIPredicate::ult, pickedSlot, state[1]);
              Value pickedI64 =
                inner.create<tensor::ExtractOp>(innerLoc, state[0], pickedSlot);
              Value pickedIndex =
                inner.create<arith::IndexCastOp>(innerLoc, index, pickedI64);
              auto pickedBox = decode(inner,
                                      innerLoc,
                                      locations,
                                      priors,
                                      pickedIndex,
                                      numPrior,
                                      half);
              Value iou = intersectionOverUnion(
                inner, innerLoc, candidateBox, pickedBox, zero);
              Value suppressed = inner.create<arith::CmpFOp>(
                innerLoc, arith::CmpFPredicate::OGT, iou, nmsThreshold);
              suppressed =
                inner.create<arith::AndIOp>(innerLoc, active, suppressed);
              Value notSuppressed = inner.create<arith::XOrIOp>(
                innerLoc,
                suppressed,
                inner.create<arith::ConstantIntOp>(innerLoc, 1, 1));
              Value keep = inner.create<arith::AndIOp>(
                innerLoc, keepState[0], notSuppressed);
              inner.create<scf::YieldOp>(innerLoc, keep);
            });
          Value keep = overlaps.getResult(0);

          Value pickedHasRoom = b.create<arith::CmpIOp>(
            bodyLoc, arith::CmpIPredicate::ult, state[1], classLimit);
          Value safePickedSlot = b.create<arith::SelectOp>(
            bodyLoc, pickedHasRoom, state[1], classLast);
          Value oldPicked =
            b.create<tensor::ExtractOp>(bodyLoc, state[0], safePickedSlot);
          Value pickedValue =
            b.create<arith::SelectOp>(bodyLoc, keep, candidateI64, oldPicked);
          Value storedPicked = b.create<tensor::InsertOp>(
            bodyLoc, pickedValue, state[0], safePickedSlot);
          Value incrementedPicked =
            b.create<arith::AddIOp>(bodyLoc, state[1], c1);
          Value nextPickedCount = b.create<arith::SelectOp>(
            bodyLoc, keep, incrementedPicked, state[1]);

          Value keptScore =
            b.create<arith::SelectOp>(bodyLoc, keep, score, minusInf);
          auto inserted = insertSorted(b,
                                       bodyLoc,
                                       state[2],
                                       state[3],
                                       keptScore,
                                       candidateI64,
                                       outputLimit,
                                       c0,
                                       c1);
          Value insertedLabels = insertAtPosition(
            b, bodyLoc, state[4], label, inserted.position, c0, c1);
          Value hasRoom = b.create<arith::CmpIOp>(
            bodyLoc, arith::CmpIPredicate::ult, state[5], outputLimit);
          Value incrementGlobal =
            b.create<arith::AndIOp>(bodyLoc, keep, hasRoom);
          Value incrementedGlobal =
            b.create<arith::AddIOp>(bodyLoc, state[5], c1);
          Value nextGlobalCount = b.create<arith::SelectOp>(
            bodyLoc, incrementGlobal, incrementedGlobal, state[5]);
          b.create<scf::YieldOp>(bodyLoc,
                                 ValueRange{storedPicked,
                                            nextPickedCount,
                                            inserted.scores,
                                            inserted.indices,
                                            insertedLabels,
                                            nextGlobalCount});
        });
      globalScores = suppress.getResult(2);
      globalIndices = suppress.getResult(3);
      globalLabels = suppress.getResult(4);
      globalCount = suppress.getResult(5);
    }

    Value output = materializeOutput(rewriter,
                                     loc,
                                     outputType,
                                     globalScores,
                                     globalIndices,
                                     globalLabels,
                                     locations,
                                     priors,
                                     numPrior,
                                     half);
    output = applyLowPrecisionBoundary(rewriter, loc, op, output);
    Value count = rewriter.create<arith::IndexCastOp>(loc, i64, globalCount);
    Value width = rewriter.create<arith::ConstantIntOp>(loc, 6, 64);
    Value actualShape =
      rewriter.create<tensor::FromElementsOp>(loc, ValueRange{count, width});
    rewriter.replaceOp(op, ValueRange{output, actualShape});
    return success();
  }

 private:
  struct InsertResult {
    Value scores;
    Value indices;
    Value position;
  };

  static Value constantF32(OpBuilder& b, Location loc, float value) {
    return b.create<arith::ConstantFloatOp>(
      loc, b.getF32Type(), APFloat(value));
  }

  static Value splat(OpBuilder& b,
                     Location loc,
                     RankedTensorType type,
                     Value value) {
    return {b.create<tensor::SplatOp>(loc, type, value)};
  }

  static Value findInsertionPosition(OpBuilder& b,
                                     Location loc,
                                     Value scores,
                                     Value score,
                                     Value limit,
                                     Value c0,
                                     Value c1) {
    auto loop = b.create<scf::ForOp>(
      loc,
      c0,
      limit,
      c1,
      ValueRange{limit},
      [&](OpBuilder& inner, Location innerLoc, Value slot, ValueRange state) {
        Value oldScore =
          inner.create<tensor::ExtractOp>(innerLoc, scores, slot);
        Value unset = inner.create<arith::CmpIOp>(
          innerLoc, arith::CmpIPredicate::eq, state[0], limit);
        Value higher = inner.create<arith::CmpFOp>(
          innerLoc, arith::CmpFPredicate::OGT, score, oldScore);
        Value select = inner.create<arith::AndIOp>(innerLoc, unset, higher);
        Value position =
          inner.create<arith::SelectOp>(innerLoc, select, slot, state[0]);
        inner.create<scf::YieldOp>(innerLoc, position);
      });
    return loop.getResult(0);
  }

  static Value insertAtPosition(OpBuilder& b,
                                Location loc,
                                Value tensor,
                                Value inserted,
                                Value position,
                                Value c0,
                                Value c1) {
    auto type = cast<RankedTensorType>(tensor.getType());
    Value limit =
      b.create<arith::ConstantIndexOp>(loc, type.getShape().front());
    Value last =
      b.create<arith::ConstantIndexOp>(loc, type.getShape().front() - 1);
    auto result = b.create<scf::ForOp>(
      loc,
      c0,
      limit,
      c1,
      ValueRange{tensor},
      [&](OpBuilder& inner,
          Location innerLoc,
          Value iteration,
          ValueRange state) {
        Value slot = inner.create<arith::SubIOp>(innerLoc, last, iteration);
        Value after = inner.create<arith::CmpIOp>(
          innerLoc, arith::CmpIPredicate::sgt, slot, position);
        Value previous = inner.create<arith::SubIOp>(innerLoc, slot, c1);
        Value safePrevious =
          inner.create<arith::SelectOp>(innerLoc, after, previous, c0);
        Value shifted =
          inner.create<tensor::ExtractOp>(innerLoc, state[0], safePrevious);
        Value current =
          inner.create<tensor::ExtractOp>(innerLoc, state[0], slot);
        Value value =
          inner.create<arith::SelectOp>(innerLoc, after, shifted, current);
        Value atPosition = inner.create<arith::CmpIOp>(
          innerLoc, arith::CmpIPredicate::eq, slot, position);
        value =
          inner.create<arith::SelectOp>(innerLoc, atPosition, inserted, value);
        Value updated =
          inner.create<tensor::InsertOp>(innerLoc, value, state[0], slot);
        inner.create<scf::YieldOp>(innerLoc, updated);
      });
    return result.getResult(0);
  }

  static InsertResult insertSorted(OpBuilder& b,
                                   Location loc,
                                   Value scores,
                                   Value indices,
                                   Value score,
                                   Value index,
                                   Value limit,
                                   Value c0,
                                   Value c1) {
    Value position =
      findInsertionPosition(b, loc, scores, score, limit, c0, c1);
    return {.scores =
              insertAtPosition(b, loc, scores, score, position, c0, c1),
            .indices =
              insertAtPosition(b, loc, indices, index, position, c0, c1),
            .position = position};
  }

  static std::array<Value, 4> decode(OpBuilder& b,
                                     Location loc,
                                     Value locations,
                                     Value priors,
                                     Value prior,
                                     int64_t numPrior,
                                     Value half) {
    Value four = b.create<arith::ConstantIndexOp>(loc, 4);
    Value base = b.create<arith::MulIOp>(loc, prior, four);
    std::array<Value, 4> locationValues;
    std::array<Value, 4> priorValues;
    std::array<Value, 4> varianceValues;
    for (int64_t coordinate = 0; coordinate < 4; ++coordinate) {
      Value offset = b.create<arith::ConstantIndexOp>(loc, coordinate);
      Value at = b.create<arith::AddIOp>(loc, base, offset);
      locationValues[coordinate] =
        b.create<tensor::ExtractOp>(loc, locations, at);
      priorValues[coordinate] = b.create<tensor::ExtractOp>(loc, priors, at);
      Value varianceAt =
        b.create<arith::ConstantIndexOp>(loc, (numPrior * 4) + coordinate);
      varianceValues[coordinate] =
        b.create<tensor::ExtractOp>(loc, priors, varianceAt);
    }
    Value priorWidth =
      b.create<arith::SubFOp>(loc, priorValues[2], priorValues[0]);
    Value priorHeight =
      b.create<arith::SubFOp>(loc, priorValues[3], priorValues[1]);
    Value priorCenterX = b.create<arith::MulFOp>(
      loc, b.create<arith::AddFOp>(loc, priorValues[0], priorValues[2]), half);
    Value priorCenterY = b.create<arith::MulFOp>(
      loc, b.create<arith::AddFOp>(loc, priorValues[1], priorValues[3]), half);
    Value centerX = b.create<arith::AddFOp>(
      loc,
      b.create<arith::MulFOp>(
        loc,
        b.create<arith::MulFOp>(loc, varianceValues[0], locationValues[0]),
        priorWidth),
      priorCenterX);
    Value centerY = b.create<arith::AddFOp>(
      loc,
      b.create<arith::MulFOp>(
        loc,
        b.create<arith::MulFOp>(loc, varianceValues[1], locationValues[1]),
        priorHeight),
      priorCenterY);
    Value width = b.create<arith::MulFOp>(
      loc,
      b.create<math::ExpOp>(
        loc,
        b.create<arith::MulFOp>(loc, varianceValues[2], locationValues[2])),
      priorWidth);
    Value height = b.create<arith::MulFOp>(
      loc,
      b.create<math::ExpOp>(
        loc,
        b.create<arith::MulFOp>(loc, varianceValues[3], locationValues[3])),
      priorHeight);
    Value halfWidth = b.create<arith::MulFOp>(loc, width, half);
    Value halfHeight = b.create<arith::MulFOp>(loc, height, half);
    return {b.create<arith::SubFOp>(loc, centerX, halfWidth),
            b.create<arith::SubFOp>(loc, centerY, halfHeight),
            b.create<arith::AddFOp>(loc, centerX, halfWidth),
            b.create<arith::AddFOp>(loc, centerY, halfHeight)};
  }

  static Value intersectionOverUnion(OpBuilder& b,
                                     Location loc,
                                     const std::array<Value, 4>& a,
                                     const std::array<Value, 4>& other,
                                     Value zero) {
    Value separatedX = b.create<arith::OrIOp>(
      loc,
      b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, a[0], other[2]),
      b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, a[2], other[0]));
    Value separatedY = b.create<arith::OrIOp>(
      loc,
      b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, a[1], other[3]),
      b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, a[3], other[1]));
    Value separated = b.create<arith::OrIOp>(loc, separatedX, separatedY);
    Value intersectionWidth =
      b.create<arith::SubFOp>(loc,
                              b.create<arith::MinNumFOp>(loc, a[2], other[2]),
                              b.create<arith::MaxNumFOp>(loc, a[0], other[0]));
    Value intersectionHeight =
      b.create<arith::SubFOp>(loc,
                              b.create<arith::MinNumFOp>(loc, a[3], other[3]),
                              b.create<arith::MaxNumFOp>(loc, a[1], other[1]));
    Value intersection =
      b.create<arith::MulFOp>(loc, intersectionWidth, intersectionHeight);
    intersection =
      b.create<arith::SelectOp>(loc, separated, zero, intersection);
    auto area = [&](const std::array<Value, 4>& box) {
      return b.create<arith::MulFOp>(
        loc,
        b.create<arith::SubFOp>(loc, box[2], box[0]),
        b.create<arith::SubFOp>(loc, box[3], box[1]));
    };
    Value unionArea = b.create<arith::SubFOp>(
      loc, b.create<arith::AddFOp>(loc, area(a), area(other)), intersection);
    return b.create<arith::DivFOp>(loc, intersection, unionArea);
  }

  static Value materializeOutput(OpBuilder& b,
                                 Location loc,
                                 RankedTensorType outputType,
                                 Value scores,
                                 Value indices,
                                 Value labels,
                                 Value locations,
                                 Value priors,
                                 int64_t numPrior,
                                 Value half) {
    Value empty = b.create<tensor::EmptyOp>(loc, outputType, ValueRange{});
    auto result = b.create<linalg::GenericOp>(
      loc,
      outputType,
      ValueRange{},
      ValueRange{empty},
      ArrayRef<AffineMap>{b.getMultiDimIdentityMap(2)},
      ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                    utils::IteratorType::parallel},
      [&](OpBuilder& inner, Location innerLoc, ValueRange) {
        Value row = inner.create<linalg::IndexOp>(innerLoc, 0);
        Value column = inner.create<linalg::IndexOp>(innerLoc, 1);
        Value priorI64 =
          inner.create<tensor::ExtractOp>(innerLoc, indices, row);
        Value prior = inner.create<arith::IndexCastOp>(
          innerLoc, inner.getIndexType(), priorI64);
        auto box =
          decode(inner, innerLoc, locations, priors, prior, numPrior, half);
        Value score = inner.create<tensor::ExtractOp>(innerLoc, scores, row);
        Value labelI64 = inner.create<tensor::ExtractOp>(innerLoc, labels, row);
        Value label =
          inner.create<arith::SIToFPOp>(innerLoc, inner.getF32Type(), labelI64);
        std::array<Value, 6> fields{
          label, score, box[0], box[1], box[2], box[3]};
        Value value = fields[0];
        for (int64_t field = 1; field < 6; ++field) {
          Value fieldIndex =
            inner.create<arith::ConstantIndexOp>(innerLoc, field);
          Value selected = inner.create<arith::CmpIOp>(
            innerLoc, arith::CmpIPredicate::eq, column, fieldIndex);
          value = inner.create<arith::SelectOp>(
            innerLoc, selected, fields[field], value);
        }
        inner.create<linalg::YieldOp>(innerLoc, value);
      });
    return result.getResult(0);
  }
};
