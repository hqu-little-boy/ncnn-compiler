module attributes {ncnn.int8_depthwise = false, ncnn.int8_kernel = "portable", ncnn.int8_target = "portable"} {
  llvm.mlir.global private constant @__constant_576x32xf32(dense<1.000000e+00> : tensor<576x32xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<576 x array<32 x f32>>
  llvm.mlir.global private constant @__constant_2304x32xf32(dense<1.000000e+00> : tensor<2304x32xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2304 x array<32 x f32>>
  llvm.func @bench_shallow(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.mlir.constant(3735928559 : index) : i64
    %2 = llvm.mlir.addressof @__constant_576x32xf32 : !llvm.ptr
    %3 = llvm.mlir.constant(18432 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(32 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(4 : index) : i64
    %8 = llvm.mlir.constant(16 : index) : i64
    %9 = llvm.mlir.constant(2 : index) : i64
    %10 = llvm.mlir.constant(3 : index) : i64
    %11 = llvm.mlir.constant(576 : index) : i64
    %12 = llvm.mlir.constant(9216 : index) : i64
    %13 = llvm.mlir.constant(1024 : index) : i64
    %14 = llvm.mlir.constant(64 : index) : i64
    %15 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<576 x array<32 x f32>>
    %17 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %18 = llvm.insertvalue %17, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %16, %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.insertvalue %4, %19[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %3, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.insertvalue %6, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%4 : i64)
  ^bb1(%23: i64):  // 2 preds: ^bb0, ^bb20
    %24 = llvm.icmp "slt" %23, %13 : i64
    llvm.cond_br %24, ^bb2, ^bb21
  ^bb2:  // pred: ^bb1
    %25 = llvm.mul %23, %11 overflow<nsw> : i64
    llvm.br ^bb3(%4 : i64)
  ^bb3(%26: i64):  // 2 preds: ^bb2, ^bb19
    %27 = llvm.icmp "slt" %26, %14 : i64
    llvm.cond_br %27, ^bb4, ^bb20
  ^bb4:  // pred: ^bb3
    %28 = llvm.mul %23, %14 overflow<nsw> : i64
    %29 = llvm.add %28, %26 : i64
    %30 = llvm.insertvalue %arg7, %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.insertvalue %arg8, %30[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %29, %31[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.insertvalue %5, %32[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %34 = llvm.insertvalue %14, %33[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.insertvalue %5, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %36 = llvm.insertvalue %6, %35[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb5(%4 : i64)
  ^bb5(%37: i64):  // 2 preds: ^bb4, ^bb12
    %38 = llvm.icmp "slt" %37, %5 : i64
    llvm.cond_br %38, ^bb6, ^bb13
  ^bb6:  // pred: ^bb5
    %39 = llvm.add %37, %6 : i64
    %40 = llvm.add %37, %9 : i64
    %41 = llvm.add %37, %10 : i64
    llvm.br ^bb7(%4 : i64)
  ^bb7(%42: i64):  // 2 preds: ^bb6, ^bb11
    %43 = llvm.icmp "slt" %42, %5 : i64
    llvm.cond_br %43, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    %44 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.getelementptr %44[%45] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %47 = llvm.mlir.constant(64 : index) : i64
    %48 = llvm.mul %37, %47 : i64
    %49 = llvm.add %48, %42 : i64
    %50 = llvm.getelementptr %46[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %51 = llvm.load %50 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %52 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %53 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %54 = llvm.getelementptr %52[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %55 = llvm.mlir.constant(64 : index) : i64
    %56 = llvm.mul %39, %55 : i64
    %57 = llvm.add %56, %42 : i64
    %58 = llvm.getelementptr %54[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %59 = llvm.load %58 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %60 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %61 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %62 = llvm.getelementptr %60[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %63 = llvm.mlir.constant(64 : index) : i64
    %64 = llvm.mul %40, %63 : i64
    %65 = llvm.add %64, %42 : i64
    %66 = llvm.getelementptr %62[%65] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %67 = llvm.load %66 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %68 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %69 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %70 = llvm.getelementptr %68[%69] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %71 = llvm.mlir.constant(64 : index) : i64
    %72 = llvm.mul %41, %71 : i64
    %73 = llvm.add %72, %42 : i64
    %74 = llvm.getelementptr %70[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %75 = llvm.load %74 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %76 = llvm.udiv %42, %8 : i64
    %77 = llvm.urem %42, %8 : i64
    %78 = llvm.mul %76, %12 : i64
    llvm.br ^bb9(%4, %51, %59, %67, %75 : i64, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>)
  ^bb9(%79: i64, %80: vector<16xf32>, %81: vector<16xf32>, %82: vector<16xf32>, %83: vector<16xf32>):  // 2 preds: ^bb8, ^bb10
    %84 = llvm.icmp "slt" %79, %11 : i64
    llvm.cond_br %84, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %85 = llvm.mul %79, %8 : i64
    %86 = llvm.add %78, %85 : i64
    %87 = llvm.add %86, %77 : i64
    %88 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %89 = llvm.getelementptr %88[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %90 = llvm.load %89 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %91 = llvm.getelementptr %arg1[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %92 = llvm.mul %37, %11 overflow<nsw, nuw> : i64
    %93 = llvm.add %92, %79 overflow<nsw, nuw> : i64
    %94 = llvm.getelementptr inbounds|nuw %91[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %95 = llvm.load %94 : !llvm.ptr -> f32
    %96 = llvm.mlir.poison : vector<16xf32>
    %97 = llvm.mlir.constant(0 : i32) : i32
    %98 = llvm.insertelement %95, %96[%97 : i32] : vector<16xf32>
    %99 = llvm.shufflevector %98, %96 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %100 = llvm.intr.fmuladd(%99, %90, %80) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %101 = llvm.getelementptr %arg1[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %102 = llvm.mul %39, %11 overflow<nsw, nuw> : i64
    %103 = llvm.add %102, %79 overflow<nsw, nuw> : i64
    %104 = llvm.getelementptr inbounds|nuw %101[%103] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %105 = llvm.load %104 : !llvm.ptr -> f32
    %106 = llvm.mlir.poison : vector<16xf32>
    %107 = llvm.mlir.constant(0 : i32) : i32
    %108 = llvm.insertelement %105, %106[%107 : i32] : vector<16xf32>
    %109 = llvm.shufflevector %108, %106 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %110 = llvm.intr.fmuladd(%109, %90, %81) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %111 = llvm.getelementptr %arg1[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %112 = llvm.mul %40, %11 overflow<nsw, nuw> : i64
    %113 = llvm.add %112, %79 overflow<nsw, nuw> : i64
    %114 = llvm.getelementptr inbounds|nuw %111[%113] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %115 = llvm.load %114 : !llvm.ptr -> f32
    %116 = llvm.mlir.poison : vector<16xf32>
    %117 = llvm.mlir.constant(0 : i32) : i32
    %118 = llvm.insertelement %115, %116[%117 : i32] : vector<16xf32>
    %119 = llvm.shufflevector %118, %116 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %120 = llvm.intr.fmuladd(%119, %90, %82) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %121 = llvm.getelementptr %arg1[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %122 = llvm.mul %41, %11 overflow<nsw, nuw> : i64
    %123 = llvm.add %122, %79 overflow<nsw, nuw> : i64
    %124 = llvm.getelementptr inbounds|nuw %121[%123] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %125 = llvm.load %124 : !llvm.ptr -> f32
    %126 = llvm.mlir.poison : vector<16xf32>
    %127 = llvm.mlir.constant(0 : i32) : i32
    %128 = llvm.insertelement %125, %126[%127 : i32] : vector<16xf32>
    %129 = llvm.shufflevector %128, %126 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %130 = llvm.intr.fmuladd(%129, %90, %83) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %131 = llvm.add %79, %6 : i64
    llvm.br ^bb9(%131, %100, %110, %120, %130 : i64, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>)
  ^bb11:  // pred: ^bb9
    %132 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %133 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %134 = llvm.getelementptr %132[%133] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %135 = llvm.mlir.constant(64 : index) : i64
    %136 = llvm.mul %37, %135 : i64
    %137 = llvm.add %136, %42 : i64
    %138 = llvm.getelementptr %134[%137] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %80, %138 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %139 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %140 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %141 = llvm.getelementptr %139[%140] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %142 = llvm.mlir.constant(64 : index) : i64
    %143 = llvm.mul %39, %142 : i64
    %144 = llvm.add %143, %42 : i64
    %145 = llvm.getelementptr %141[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %81, %145 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %146 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %147 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %148 = llvm.getelementptr %146[%147] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %149 = llvm.mlir.constant(64 : index) : i64
    %150 = llvm.mul %40, %149 : i64
    %151 = llvm.add %150, %42 : i64
    %152 = llvm.getelementptr %148[%151] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %82, %152 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %153 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %154 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %155 = llvm.getelementptr %153[%154] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %156 = llvm.mlir.constant(64 : index) : i64
    %157 = llvm.mul %41, %156 : i64
    %158 = llvm.add %157, %42 : i64
    %159 = llvm.getelementptr %155[%158] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %83, %159 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %160 = llvm.add %42, %8 : i64
    llvm.br ^bb7(%160 : i64)
  ^bb12:  // pred: ^bb7
    %161 = llvm.add %37, %7 : i64
    llvm.br ^bb5(%161 : i64)
  ^bb13:  // pred: ^bb5
    llvm.br ^bb14(%4 : i64)
  ^bb14(%162: i64):  // 2 preds: ^bb13, ^bb18
    %163 = llvm.icmp "slt" %162, %5 : i64
    llvm.cond_br %163, ^bb15, ^bb19
  ^bb15:  // pred: ^bb14
    llvm.br ^bb16(%4 : i64)
  ^bb16(%164: i64):  // 2 preds: ^bb15, ^bb17
    %165 = llvm.icmp "slt" %164, %5 : i64
    llvm.cond_br %165, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %166 = llvm.getelementptr %arg8[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %167 = llvm.mul %162, %14 overflow<nsw, nuw> : i64
    %168 = llvm.add %167, %164 overflow<nsw, nuw> : i64
    %169 = llvm.getelementptr inbounds|nuw %166[%168] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %170 = llvm.load %169 : !llvm.ptr -> f32
    %171 = llvm.getelementptr %arg8[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %172 = llvm.mul %162, %14 overflow<nsw, nuw> : i64
    %173 = llvm.add %172, %164 overflow<nsw, nuw> : i64
    %174 = llvm.getelementptr inbounds|nuw %171[%173] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %170, %174 : f32, !llvm.ptr
    %175 = llvm.add %164, %6 : i64
    llvm.br ^bb16(%175 : i64)
  ^bb18:  // pred: ^bb16
    %176 = llvm.add %162, %6 : i64
    llvm.br ^bb14(%176 : i64)
  ^bb19:  // pred: ^bb14
    %177 = llvm.add %26, %5 : i64
    llvm.br ^bb3(%177 : i64)
  ^bb20:  // pred: ^bb3
    %178 = llvm.add %23, %5 : i64
    llvm.br ^bb1(%178 : i64)
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
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.mlir.constant(3735928559 : index) : i64
    %2 = llvm.mlir.addressof @__constant_2304x32xf32 : !llvm.ptr
    %3 = llvm.mlir.constant(73728 : index) : i64
    %4 = llvm.mlir.constant(0 : index) : i64
    %5 = llvm.mlir.constant(32 : index) : i64
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(4 : index) : i64
    %8 = llvm.mlir.constant(16 : index) : i64
    %9 = llvm.mlir.constant(2 : index) : i64
    %10 = llvm.mlir.constant(3 : index) : i64
    %11 = llvm.mlir.constant(2304 : index) : i64
    %12 = llvm.mlir.constant(128 : index) : i64
    %13 = llvm.mlir.constant(36864 : index) : i64
    %14 = llvm.mlir.constant(256 : index) : i64
    %15 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.getelementptr %2[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2304 x array<32 x f32>>
    %17 = llvm.inttoptr %1 : i64 to !llvm.ptr
    %18 = llvm.insertvalue %17, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %16, %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.insertvalue %4, %19[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %21 = llvm.insertvalue %3, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %22 = llvm.insertvalue %6, %21[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.br ^bb1(%4 : i64)
  ^bb1(%23: i64):  // 2 preds: ^bb0, ^bb23
    %24 = llvm.icmp "slt" %23, %14 : i64
    llvm.cond_br %24, ^bb2, ^bb24
  ^bb2:  // pred: ^bb1
    %25 = llvm.mul %23, %11 overflow<nsw> : i64
    llvm.br ^bb3(%4 : i64)
  ^bb3(%26: i64):  // 2 preds: ^bb2, ^bb22
    %27 = llvm.icmp "slt" %26, %14 : i64
    llvm.cond_br %27, ^bb4, ^bb23
  ^bb4:  // pred: ^bb3
    %28 = llvm.mul %23, %14 overflow<nsw> : i64
    %29 = llvm.add %28, %26 : i64
    %30 = llvm.insertvalue %arg7, %15[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %31 = llvm.insertvalue %arg8, %30[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %32 = llvm.insertvalue %29, %31[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %33 = llvm.insertvalue %5, %32[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %34 = llvm.insertvalue %14, %33[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.insertvalue %5, %34[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %36 = llvm.insertvalue %6, %35[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb5(%4 : i64)
  ^bb5(%37: i64):  // 2 preds: ^bb4, ^bb15
    %38 = llvm.icmp "slt" %37, %5 : i64
    llvm.cond_br %38, ^bb6, ^bb16
  ^bb6:  // pred: ^bb5
    %39 = llvm.add %37, %6 : i64
    %40 = llvm.add %37, %9 : i64
    %41 = llvm.add %37, %10 : i64
    llvm.br ^bb7(%4 : i64)
  ^bb7(%42: i64):  // 2 preds: ^bb6, ^bb14
    %43 = llvm.icmp "slt" %42, %5 : i64
    llvm.cond_br %43, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    %44 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %46 = llvm.getelementptr %44[%45] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %47 = llvm.mlir.constant(256 : index) : i64
    %48 = llvm.mul %37, %47 : i64
    %49 = llvm.add %48, %42 : i64
    %50 = llvm.getelementptr %46[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %51 = llvm.load %50 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %52 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %53 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %54 = llvm.getelementptr %52[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %55 = llvm.mlir.constant(256 : index) : i64
    %56 = llvm.mul %39, %55 : i64
    %57 = llvm.add %56, %42 : i64
    %58 = llvm.getelementptr %54[%57] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %59 = llvm.load %58 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %60 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %61 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %62 = llvm.getelementptr %60[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %63 = llvm.mlir.constant(256 : index) : i64
    %64 = llvm.mul %40, %63 : i64
    %65 = llvm.add %64, %42 : i64
    %66 = llvm.getelementptr %62[%65] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %67 = llvm.load %66 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %68 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %69 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %70 = llvm.getelementptr %68[%69] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %71 = llvm.mlir.constant(256 : index) : i64
    %72 = llvm.mul %41, %71 : i64
    %73 = llvm.add %72, %42 : i64
    %74 = llvm.getelementptr %70[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %75 = llvm.load %74 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %76 = llvm.udiv %42, %8 : i64
    %77 = llvm.urem %42, %8 : i64
    %78 = llvm.mul %76, %13 : i64
    llvm.br ^bb9(%4, %51, %59, %67, %75 : i64, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>)
  ^bb9(%79: i64, %80: vector<16xf32>, %81: vector<16xf32>, %82: vector<16xf32>, %83: vector<16xf32>):  // 2 preds: ^bb8, ^bb13
    %84 = llvm.icmp "slt" %79, %11 : i64
    llvm.cond_br %84, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    %85 = llvm.add %79, %12 : i64
    llvm.br ^bb11(%79, %80, %81, %82, %83 : i64, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>)
  ^bb11(%86: i64, %87: vector<16xf32>, %88: vector<16xf32>, %89: vector<16xf32>, %90: vector<16xf32>):  // 2 preds: ^bb10, ^bb12
    %91 = llvm.icmp "slt" %86, %85 : i64
    llvm.cond_br %91, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %92 = llvm.mul %86, %8 : i64
    %93 = llvm.add %78, %92 : i64
    %94 = llvm.add %93, %77 : i64
    %95 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %96 = llvm.getelementptr %95[%94] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %97 = llvm.load %96 {alignment = 4 : i64} : !llvm.ptr -> vector<16xf32>
    %98 = llvm.getelementptr %arg1[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %99 = llvm.mul %37, %11 overflow<nsw, nuw> : i64
    %100 = llvm.add %99, %86 overflow<nsw, nuw> : i64
    %101 = llvm.getelementptr inbounds|nuw %98[%100] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %102 = llvm.load %101 : !llvm.ptr -> f32
    %103 = llvm.mlir.poison : vector<16xf32>
    %104 = llvm.mlir.constant(0 : i32) : i32
    %105 = llvm.insertelement %102, %103[%104 : i32] : vector<16xf32>
    %106 = llvm.shufflevector %105, %103 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %107 = llvm.intr.fmuladd(%106, %97, %87) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %108 = llvm.getelementptr %arg1[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %109 = llvm.mul %39, %11 overflow<nsw, nuw> : i64
    %110 = llvm.add %109, %86 overflow<nsw, nuw> : i64
    %111 = llvm.getelementptr inbounds|nuw %108[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %112 = llvm.load %111 : !llvm.ptr -> f32
    %113 = llvm.mlir.poison : vector<16xf32>
    %114 = llvm.mlir.constant(0 : i32) : i32
    %115 = llvm.insertelement %112, %113[%114 : i32] : vector<16xf32>
    %116 = llvm.shufflevector %115, %113 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %117 = llvm.intr.fmuladd(%116, %97, %88) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %118 = llvm.getelementptr %arg1[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %119 = llvm.mul %40, %11 overflow<nsw, nuw> : i64
    %120 = llvm.add %119, %86 overflow<nsw, nuw> : i64
    %121 = llvm.getelementptr inbounds|nuw %118[%120] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %122 = llvm.load %121 : !llvm.ptr -> f32
    %123 = llvm.mlir.poison : vector<16xf32>
    %124 = llvm.mlir.constant(0 : i32) : i32
    %125 = llvm.insertelement %122, %123[%124 : i32] : vector<16xf32>
    %126 = llvm.shufflevector %125, %123 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %127 = llvm.intr.fmuladd(%126, %97, %89) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %128 = llvm.getelementptr %arg1[%25] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %129 = llvm.mul %41, %11 overflow<nsw, nuw> : i64
    %130 = llvm.add %129, %86 overflow<nsw, nuw> : i64
    %131 = llvm.getelementptr inbounds|nuw %128[%130] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %132 = llvm.load %131 : !llvm.ptr -> f32
    %133 = llvm.mlir.poison : vector<16xf32>
    %134 = llvm.mlir.constant(0 : i32) : i32
    %135 = llvm.insertelement %132, %133[%134 : i32] : vector<16xf32>
    %136 = llvm.shufflevector %135, %133 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>
    %137 = llvm.intr.fmuladd(%136, %97, %90) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %138 = llvm.add %86, %6 : i64
    llvm.br ^bb11(%138, %107, %117, %127, %137 : i64, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>)
  ^bb13:  // pred: ^bb11
    %139 = llvm.add %79, %12 : i64
    llvm.br ^bb9(%139, %87, %88, %89, %90 : i64, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>)
  ^bb14:  // pred: ^bb9
    %140 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %141 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %142 = llvm.getelementptr %140[%141] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %143 = llvm.mlir.constant(256 : index) : i64
    %144 = llvm.mul %37, %143 : i64
    %145 = llvm.add %144, %42 : i64
    %146 = llvm.getelementptr %142[%145] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %80, %146 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %147 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %148 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %149 = llvm.getelementptr %147[%148] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %150 = llvm.mlir.constant(256 : index) : i64
    %151 = llvm.mul %39, %150 : i64
    %152 = llvm.add %151, %42 : i64
    %153 = llvm.getelementptr %149[%152] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %81, %153 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %154 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %155 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %156 = llvm.getelementptr %154[%155] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %157 = llvm.mlir.constant(256 : index) : i64
    %158 = llvm.mul %40, %157 : i64
    %159 = llvm.add %158, %42 : i64
    %160 = llvm.getelementptr %156[%159] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %82, %160 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %161 = llvm.extractvalue %36[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %162 = llvm.extractvalue %36[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %163 = llvm.getelementptr %161[%162] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %164 = llvm.mlir.constant(256 : index) : i64
    %165 = llvm.mul %41, %164 : i64
    %166 = llvm.add %165, %42 : i64
    %167 = llvm.getelementptr %163[%166] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %83, %167 {alignment = 4 : i64} : vector<16xf32>, !llvm.ptr
    %168 = llvm.add %42, %8 : i64
    llvm.br ^bb7(%168 : i64)
  ^bb15:  // pred: ^bb7
    %169 = llvm.add %37, %7 : i64
    llvm.br ^bb5(%169 : i64)
  ^bb16:  // pred: ^bb5
    llvm.br ^bb17(%4 : i64)
  ^bb17(%170: i64):  // 2 preds: ^bb16, ^bb21
    %171 = llvm.icmp "slt" %170, %5 : i64
    llvm.cond_br %171, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    llvm.br ^bb19(%4 : i64)
  ^bb19(%172: i64):  // 2 preds: ^bb18, ^bb20
    %173 = llvm.icmp "slt" %172, %5 : i64
    llvm.cond_br %173, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %174 = llvm.getelementptr %arg8[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %175 = llvm.mul %170, %14 overflow<nsw, nuw> : i64
    %176 = llvm.add %175, %172 overflow<nsw, nuw> : i64
    %177 = llvm.getelementptr inbounds|nuw %174[%176] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %178 = llvm.load %177 : !llvm.ptr -> f32
    %179 = llvm.getelementptr %arg8[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %180 = llvm.mul %170, %14 overflow<nsw, nuw> : i64
    %181 = llvm.add %180, %172 overflow<nsw, nuw> : i64
    %182 = llvm.getelementptr inbounds|nuw %179[%181] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %178, %182 : f32, !llvm.ptr
    %183 = llvm.add %172, %6 : i64
    llvm.br ^bb19(%183 : i64)
  ^bb21:  // pred: ^bb19
    %184 = llvm.add %170, %6 : i64
    llvm.br ^bb17(%184 : i64)
  ^bb22:  // pred: ^bb17
    %185 = llvm.add %26, %5 : i64
    llvm.br ^bb3(%185 : i64)
  ^bb23:  // pred: ^bb3
    %186 = llvm.add %23, %5 : i64
    llvm.br ^bb1(%186 : i64)
  ^bb24:  // pred: ^bb1
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
