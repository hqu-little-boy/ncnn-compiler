module attributes {ncnn.int8_depthwise = false, ncnn.int8_kernel = "portable", ncnn.int8_target = "portable"} {
  llvm.mlir.global private constant @__constant_2304x32xf32(dense<1.000000e+00> : tensor<2304x32xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2304 x array<32 x f32>>
  llvm.mlir.global private constant @__constant_576x32xf32(dense<1.000000e+00> : tensor<576x32xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<576 x array<32 x f32>>
  llvm.func @bench_shallow(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3735928559 : index) : i64
    %1 = llvm.mlir.addressof @__constant_576x32xf32 : !llvm.ptr
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = llvm.mlir.constant(32 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(4 : index) : i64
    %6 = llvm.mlir.constant(16 : index) : i64
    %7 = llvm.mlir.constant(2 : index) : i64
    %8 = llvm.mlir.constant(3 : index) : i64
    %9 = llvm.mlir.constant(576 : index) : i64
    %10 = llvm.mlir.constant(1024 : index) : i64
    %11 = llvm.mlir.constant(64 : index) : i64
    %12 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<576 x array<32 x f32>>
    %14 = llvm.inttoptr %0 : i64 to !llvm.ptr
    %15 = llvm.insertvalue %14, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %13, %15[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %2, %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.insertvalue %9, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.insertvalue %3, %18[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.insertvalue %3, %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.insertvalue %4, %20[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb1(%2 : i64)
  ^bb1(%22: i64):  // 2 preds: ^bb0, ^bb20
    %23 = llvm.icmp "slt" %22, %10 : i64
    llvm.cond_br %23, ^bb2, ^bb21
  ^bb2:  // pred: ^bb1
    %24 = llvm.mul %22, %9 overflow<nsw> : i64
    llvm.br ^bb3(%2 : i64)
  ^bb3(%25: i64):  // 2 preds: ^bb2, ^bb19
    %26 = llvm.icmp "slt" %25, %11 : i64
    llvm.cond_br %26, ^bb4, ^bb20
  ^bb4:  // pred: ^bb3
    %27 = llvm.mul %22, %11 overflow<nsw> : i64
    %28 = llvm.add %27, %25 : i64
    %29 = llvm.insertvalue %arg7, %12[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.insertvalue %arg8, %29[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.insertvalue %28, %30[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %3, %31[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.insertvalue %11, %32[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %34 = llvm.insertvalue %3, %33[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.insertvalue %4, %34[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb5(%2 : i64)
  ^bb5(%36: i64):  // 2 preds: ^bb4, ^bb12
    %37 = llvm.icmp "slt" %36, %3 : i64
    llvm.cond_br %37, ^bb6, ^bb13
  ^bb6:  // pred: ^bb5
    %38 = llvm.add %36, %4 : i64
    %39 = llvm.add %36, %7 : i64
    %40 = llvm.add %36, %8 : i64
    llvm.br ^bb7(%2 : i64)
  ^bb7(%41: i64):  // 2 preds: ^bb6, ^bb11
    %42 = llvm.icmp "slt" %41, %3 : i64
    llvm.cond_br %42, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    %43 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %44 = llvm.extractvalue %35[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.getelementptr %43[%44] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %46 = llvm.mlir.constant(64 : index) : i64
    %47 = llvm.mul %36, %46 : i64
    %48 = llvm.add %47, %41 : i64
    %49 = llvm.getelementptr %45[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %50 = llvm.load %49 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %51 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %52 = llvm.extractvalue %35[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %53 = llvm.getelementptr %51[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %54 = llvm.mlir.constant(64 : index) : i64
    %55 = llvm.mul %38, %54 : i64
    %56 = llvm.add %55, %41 : i64
    %57 = llvm.getelementptr %53[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %58 = llvm.load %57 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %59 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.extractvalue %35[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %61 = llvm.getelementptr %59[%60] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %62 = llvm.mlir.constant(64 : index) : i64
    %63 = llvm.mul %39, %62 : i64
    %64 = llvm.add %63, %41 : i64
    %65 = llvm.getelementptr %61[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %66 = llvm.load %65 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %67 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %68 = llvm.extractvalue %35[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %69 = llvm.getelementptr %67[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %70 = llvm.mlir.constant(64 : index) : i64
    %71 = llvm.mul %40, %70 : i64
    %72 = llvm.add %71, %41 : i64
    %73 = llvm.getelementptr %69[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %74 = llvm.load %73 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    llvm.br ^bb9(%2, %50, %58, %66, %74 : i64, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>)
  ^bb9(%75: i64, %76: vector<16xf32>, %77: vector<16xf32>, %78: vector<16xf32>, %79: vector<16xf32>):  // 2 preds: ^bb8, ^bb10
    %80 = llvm.icmp "slt" %75, %9 : i64
    llvm.cond_br %80, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %81 = llvm.extractvalue %21[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %82 = llvm.mlir.constant(32 : index) : i64
    %83 = llvm.mul %75, %82 : i64
    %84 = llvm.add %83, %41 : i64
    %85 = llvm.getelementptr %81[%84] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %86 = llvm.load %85 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %87 = llvm.getelementptr %arg1[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %88 = llvm.mul %36, %9 overflow<nsw, nuw> : i64
    %89 = llvm.add %88, %75 overflow<nsw, nuw> : i64
    %90 = llvm.getelementptr inbounds|nuw %87[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %91 = llvm.load %90 : !llvm.ptr -> f32
    %92 = llvm.mlir.poison : vector<16xf32>
    %93 = llvm.mlir.constant(0 : i32) : i32
    %94 = llvm.insertelement %91, %92[%93 : i32] : vector<16xf32>
    %95 = llvm.shufflevector %94, %92 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %96 = llvm.intr.fmuladd(%95, %86, %76) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %97 = llvm.getelementptr %arg1[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %98 = llvm.mul %38, %9 overflow<nsw, nuw> : i64
    %99 = llvm.add %98, %75 overflow<nsw, nuw> : i64
    %100 = llvm.getelementptr inbounds|nuw %97[%99] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %101 = llvm.load %100 : !llvm.ptr -> f32
    %102 = llvm.mlir.poison : vector<16xf32>
    %103 = llvm.mlir.constant(0 : i32) : i32
    %104 = llvm.insertelement %101, %102[%103 : i32] : vector<16xf32>
    %105 = llvm.shufflevector %104, %102 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %106 = llvm.intr.fmuladd(%105, %86, %77) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %107 = llvm.getelementptr %arg1[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %108 = llvm.mul %39, %9 overflow<nsw, nuw> : i64
    %109 = llvm.add %108, %75 overflow<nsw, nuw> : i64
    %110 = llvm.getelementptr inbounds|nuw %107[%109] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %111 = llvm.load %110 : !llvm.ptr -> f32
    %112 = llvm.mlir.poison : vector<16xf32>
    %113 = llvm.mlir.constant(0 : i32) : i32
    %114 = llvm.insertelement %111, %112[%113 : i32] : vector<16xf32>
    %115 = llvm.shufflevector %114, %112 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %116 = llvm.intr.fmuladd(%115, %86, %78) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %117 = llvm.getelementptr %arg1[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %118 = llvm.mul %40, %9 overflow<nsw, nuw> : i64
    %119 = llvm.add %118, %75 overflow<nsw, nuw> : i64
    %120 = llvm.getelementptr inbounds|nuw %117[%119] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %121 = llvm.load %120 : !llvm.ptr -> f32
    %122 = llvm.mlir.poison : vector<16xf32>
    %123 = llvm.mlir.constant(0 : i32) : i32
    %124 = llvm.insertelement %121, %122[%123 : i32] : vector<16xf32>
    %125 = llvm.shufflevector %124, %122 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %126 = llvm.intr.fmuladd(%125, %86, %79) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %127 = llvm.add %75, %4 : i64
    llvm.br ^bb9(%127, %96, %106, %116, %126 : i64, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>)
  ^bb11:  // pred: ^bb9
    %128 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %129 = llvm.extractvalue %35[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %130 = llvm.getelementptr %128[%129] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %131 = llvm.mlir.constant(64 : index) : i64
    %132 = llvm.mul %36, %131 : i64
    %133 = llvm.add %132, %41 : i64
    %134 = llvm.getelementptr %130[%133] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %76, %134 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %135 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %136 = llvm.extractvalue %35[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %137 = llvm.getelementptr %135[%136] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %138 = llvm.mlir.constant(64 : index) : i64
    %139 = llvm.mul %38, %138 : i64
    %140 = llvm.add %139, %41 : i64
    %141 = llvm.getelementptr %137[%140] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %77, %141 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %142 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %143 = llvm.extractvalue %35[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %144 = llvm.getelementptr %142[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %145 = llvm.mlir.constant(64 : index) : i64
    %146 = llvm.mul %39, %145 : i64
    %147 = llvm.add %146, %41 : i64
    %148 = llvm.getelementptr %144[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %78, %148 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %149 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %150 = llvm.extractvalue %35[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %151 = llvm.getelementptr %149[%150] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %152 = llvm.mlir.constant(64 : index) : i64
    %153 = llvm.mul %40, %152 : i64
    %154 = llvm.add %153, %41 : i64
    %155 = llvm.getelementptr %151[%154] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %79, %155 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %156 = llvm.add %41, %6 : i64
    llvm.br ^bb7(%156 : i64)
  ^bb12:  // pred: ^bb7
    %157 = llvm.add %36, %5 : i64
    llvm.br ^bb5(%157 : i64)
  ^bb13:  // pred: ^bb5
    llvm.br ^bb14(%2 : i64)
  ^bb14(%158: i64):  // 2 preds: ^bb13, ^bb18
    %159 = llvm.icmp "slt" %158, %3 : i64
    llvm.cond_br %159, ^bb15, ^bb19
  ^bb15:  // pred: ^bb14
    llvm.br ^bb16(%2 : i64)
  ^bb16(%160: i64):  // 2 preds: ^bb15, ^bb17
    %161 = llvm.icmp "slt" %160, %3 : i64
    llvm.cond_br %161, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %162 = llvm.getelementptr %arg8[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %163 = llvm.mul %158, %11 overflow<nsw, nuw> : i64
    %164 = llvm.add %163, %160 overflow<nsw, nuw> : i64
    %165 = llvm.getelementptr inbounds|nuw %162[%164] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %166 = llvm.load %165 : !llvm.ptr -> f32
    %167 = llvm.getelementptr %arg8[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %168 = llvm.mul %158, %11 overflow<nsw, nuw> : i64
    %169 = llvm.add %168, %160 overflow<nsw, nuw> : i64
    %170 = llvm.getelementptr inbounds|nuw %167[%169] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %166, %170 : f32, !llvm.ptr
    %171 = llvm.add %160, %4 : i64
    llvm.br ^bb16(%171 : i64)
  ^bb18:  // pred: ^bb16
    %172 = llvm.add %158, %4 : i64
    llvm.br ^bb14(%172 : i64)
  ^bb19:  // pred: ^bb14
    %173 = llvm.add %25, %3 : i64
    llvm.br ^bb3(%173 : i64)
  ^bb20:  // pred: ^bb3
    %174 = llvm.add %22, %3 : i64
    llvm.br ^bb1(%174 : i64)
  ^bb21:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_bench_shallow(%arg0: !llvm.ptr, %arg1: !llvm.ptr {bufferize.result}) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @bench_shallow(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
  llvm.func @bench_deep(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3735928559 : index) : i64
    %1 = llvm.mlir.addressof @__constant_2304x32xf32 : !llvm.ptr
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = llvm.mlir.constant(32 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(4 : index) : i64
    %6 = llvm.mlir.constant(16 : index) : i64
    %7 = llvm.mlir.constant(2 : index) : i64
    %8 = llvm.mlir.constant(3 : index) : i64
    %9 = llvm.mlir.constant(2304 : index) : i64
    %10 = llvm.mlir.constant(256 : index) : i64
    %11 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2304 x array<32 x f32>>
    %13 = llvm.inttoptr %0 : i64 to !llvm.ptr
    %14 = llvm.insertvalue %13, %11[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %12, %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %2, %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %9, %16[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %18 = llvm.insertvalue %3, %17[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %19 = llvm.insertvalue %3, %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.insertvalue %4, %19[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb1(%2 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb20
    %22 = llvm.icmp "slt" %21, %10 : i64
    llvm.cond_br %22, ^bb2, ^bb21
  ^bb2:  // pred: ^bb1
    %23 = llvm.mul %21, %9 overflow<nsw> : i64
    llvm.br ^bb3(%2 : i64)
  ^bb3(%24: i64):  // 2 preds: ^bb2, ^bb19
    %25 = llvm.icmp "slt" %24, %10 : i64
    llvm.cond_br %25, ^bb4, ^bb20
  ^bb4:  // pred: ^bb3
    %26 = llvm.mul %21, %10 overflow<nsw> : i64
    %27 = llvm.add %26, %24 : i64
    %28 = llvm.insertvalue %arg7, %11[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %29 = llvm.insertvalue %arg8, %28[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %30 = llvm.insertvalue %27, %29[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.insertvalue %3, %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %10, %31[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.insertvalue %3, %32[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %34 = llvm.insertvalue %4, %33[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb5(%2 : i64)
  ^bb5(%35: i64):  // 2 preds: ^bb4, ^bb12
    %36 = llvm.icmp "slt" %35, %3 : i64
    llvm.cond_br %36, ^bb6, ^bb13
  ^bb6:  // pred: ^bb5
    %37 = llvm.add %35, %4 : i64
    %38 = llvm.add %35, %7 : i64
    %39 = llvm.add %35, %8 : i64
    llvm.br ^bb7(%2 : i64)
  ^bb7(%40: i64):  // 2 preds: ^bb6, ^bb11
    %41 = llvm.icmp "slt" %40, %3 : i64
    llvm.cond_br %41, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    %42 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %43 = llvm.extractvalue %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %44 = llvm.getelementptr %42[%43] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %45 = llvm.mlir.constant(256 : index) : i64
    %46 = llvm.mul %35, %45 : i64
    %47 = llvm.add %46, %40 : i64
    %48 = llvm.getelementptr %44[%47] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %49 = llvm.load %48 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %50 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %51 = llvm.extractvalue %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %52 = llvm.getelementptr %50[%51] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %53 = llvm.mlir.constant(256 : index) : i64
    %54 = llvm.mul %37, %53 : i64
    %55 = llvm.add %54, %40 : i64
    %56 = llvm.getelementptr %52[%55] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %57 = llvm.load %56 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %58 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %59 = llvm.extractvalue %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.getelementptr %58[%59] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %61 = llvm.mlir.constant(256 : index) : i64
    %62 = llvm.mul %38, %61 : i64
    %63 = llvm.add %62, %40 : i64
    %64 = llvm.getelementptr %60[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %65 = llvm.load %64 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %66 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %67 = llvm.extractvalue %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %68 = llvm.getelementptr %66[%67] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %69 = llvm.mlir.constant(256 : index) : i64
    %70 = llvm.mul %39, %69 : i64
    %71 = llvm.add %70, %40 : i64
    %72 = llvm.getelementptr %68[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %73 = llvm.load %72 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    llvm.br ^bb9(%2, %49, %57, %65, %73 : i64, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>)
  ^bb9(%74: i64, %75: vector<16xf32>, %76: vector<16xf32>, %77: vector<16xf32>, %78: vector<16xf32>):  // 2 preds: ^bb8, ^bb10
    %79 = llvm.icmp "slt" %74, %9 : i64
    llvm.cond_br %79, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %80 = llvm.extractvalue %20[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %81 = llvm.mlir.constant(32 : index) : i64
    %82 = llvm.mul %74, %81 : i64
    %83 = llvm.add %82, %40 : i64
    %84 = llvm.getelementptr %80[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %85 = llvm.load %84 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %86 = llvm.getelementptr %arg1[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %87 = llvm.mul %35, %9 overflow<nsw, nuw> : i64
    %88 = llvm.add %87, %74 overflow<nsw, nuw> : i64
    %89 = llvm.getelementptr inbounds|nuw %86[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %90 = llvm.load %89 : !llvm.ptr -> f32
    %91 = llvm.mlir.poison : vector<16xf32>
    %92 = llvm.mlir.constant(0 : i32) : i32
    %93 = llvm.insertelement %90, %91[%92 : i32] : vector<16xf32>
    %94 = llvm.shufflevector %93, %91 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %95 = llvm.intr.fmuladd(%94, %85, %75) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %96 = llvm.getelementptr %arg1[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %97 = llvm.mul %37, %9 overflow<nsw, nuw> : i64
    %98 = llvm.add %97, %74 overflow<nsw, nuw> : i64
    %99 = llvm.getelementptr inbounds|nuw %96[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %100 = llvm.load %99 : !llvm.ptr -> f32
    %101 = llvm.mlir.poison : vector<16xf32>
    %102 = llvm.mlir.constant(0 : i32) : i32
    %103 = llvm.insertelement %100, %101[%102 : i32] : vector<16xf32>
    %104 = llvm.shufflevector %103, %101 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %105 = llvm.intr.fmuladd(%104, %85, %76) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %106 = llvm.getelementptr %arg1[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %107 = llvm.mul %38, %9 overflow<nsw, nuw> : i64
    %108 = llvm.add %107, %74 overflow<nsw, nuw> : i64
    %109 = llvm.getelementptr inbounds|nuw %106[%108] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %110 = llvm.load %109 : !llvm.ptr -> f32
    %111 = llvm.mlir.poison : vector<16xf32>
    %112 = llvm.mlir.constant(0 : i32) : i32
    %113 = llvm.insertelement %110, %111[%112 : i32] : vector<16xf32>
    %114 = llvm.shufflevector %113, %111 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %115 = llvm.intr.fmuladd(%114, %85, %77) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %116 = llvm.getelementptr %arg1[%23] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %117 = llvm.mul %39, %9 overflow<nsw, nuw> : i64
    %118 = llvm.add %117, %74 overflow<nsw, nuw> : i64
    %119 = llvm.getelementptr inbounds|nuw %116[%118] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %120 = llvm.load %119 : !llvm.ptr -> f32
    %121 = llvm.mlir.poison : vector<16xf32>
    %122 = llvm.mlir.constant(0 : i32) : i32
    %123 = llvm.insertelement %120, %121[%122 : i32] : vector<16xf32>
    %124 = llvm.shufflevector %123, %121 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %125 = llvm.intr.fmuladd(%124, %85, %78) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %126 = llvm.add %74, %4 : i64
    llvm.br ^bb9(%126, %95, %105, %115, %125 : i64, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>)
  ^bb11:  // pred: ^bb9
    %127 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %128 = llvm.extractvalue %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %129 = llvm.getelementptr %127[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %130 = llvm.mlir.constant(256 : index) : i64
    %131 = llvm.mul %35, %130 : i64
    %132 = llvm.add %131, %40 : i64
    %133 = llvm.getelementptr %129[%132] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %75, %133 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %134 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %135 = llvm.extractvalue %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %136 = llvm.getelementptr %134[%135] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %137 = llvm.mlir.constant(256 : index) : i64
    %138 = llvm.mul %37, %137 : i64
    %139 = llvm.add %138, %40 : i64
    %140 = llvm.getelementptr %136[%139] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %76, %140 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %141 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %142 = llvm.extractvalue %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %143 = llvm.getelementptr %141[%142] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %144 = llvm.mlir.constant(256 : index) : i64
    %145 = llvm.mul %38, %144 : i64
    %146 = llvm.add %145, %40 : i64
    %147 = llvm.getelementptr %143[%146] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %77, %147 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %148 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %149 = llvm.extractvalue %34[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %150 = llvm.getelementptr %148[%149] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %151 = llvm.mlir.constant(256 : index) : i64
    %152 = llvm.mul %39, %151 : i64
    %153 = llvm.add %152, %40 : i64
    %154 = llvm.getelementptr %150[%153] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %78, %154 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %155 = llvm.add %40, %6 : i64
    llvm.br ^bb7(%155 : i64)
  ^bb12:  // pred: ^bb7
    %156 = llvm.add %35, %5 : i64
    llvm.br ^bb5(%156 : i64)
  ^bb13:  // pred: ^bb5
    llvm.br ^bb14(%2 : i64)
  ^bb14(%157: i64):  // 2 preds: ^bb13, ^bb18
    %158 = llvm.icmp "slt" %157, %3 : i64
    llvm.cond_br %158, ^bb15, ^bb19
  ^bb15:  // pred: ^bb14
    llvm.br ^bb16(%2 : i64)
  ^bb16(%159: i64):  // 2 preds: ^bb15, ^bb17
    %160 = llvm.icmp "slt" %159, %3 : i64
    llvm.cond_br %160, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %161 = llvm.getelementptr %arg8[%27] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %162 = llvm.mul %157, %10 overflow<nsw, nuw> : i64
    %163 = llvm.add %162, %159 overflow<nsw, nuw> : i64
    %164 = llvm.getelementptr inbounds|nuw %161[%163] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %165 = llvm.load %164 : !llvm.ptr -> f32
    %166 = llvm.getelementptr %arg8[%27] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %167 = llvm.mul %157, %10 overflow<nsw, nuw> : i64
    %168 = llvm.add %167, %159 overflow<nsw, nuw> : i64
    %169 = llvm.getelementptr inbounds|nuw %166[%168] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %165, %169 : f32, !llvm.ptr
    %170 = llvm.add %159, %4 : i64
    llvm.br ^bb16(%170 : i64)
  ^bb18:  // pred: ^bb16
    %171 = llvm.add %157, %4 : i64
    llvm.br ^bb14(%171 : i64)
  ^bb19:  // pred: ^bb14
    %172 = llvm.add %24, %3 : i64
    llvm.br ^bb3(%172 : i64)
  ^bb20:  // pred: ^bb3
    %173 = llvm.add %21, %3 : i64
    llvm.br ^bb1(%173 : i64)
  ^bb21:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_bench_deep(%arg0: !llvm.ptr, %arg1: !llvm.ptr {bufferize.result}) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.call @bench_deep(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
}
