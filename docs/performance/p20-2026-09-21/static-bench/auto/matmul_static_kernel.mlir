module attributes {ncnn.int8_depthwise = false, ncnn.int8_kernel = "portable", ncnn.int8_target = "portable"} {
  memref.global "private" constant @__constant_576x32xf32 : memref<576x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2304x32xf32 : memref<2304x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @bench_shallow(%arg0: memref<1024x576xf32>, %arg1: memref<1024x64xf32> {bufferize.result}) attributes {llvm.emit_c_interface} {
    %c9216 = arith.constant 9216 : index
    %c576 = arith.constant 576 : index
    %0 = ub.poison : f32
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %1 = memref.get_global @__constant_576x32xf32 : memref<576x32xf32> {ncnn.alignment = "64", ncnn.pack_bytes = 147456 : i64, ncnn.pack_factor = 16 : i64, ncnn.pack_runtime = "compile_time_B", ncnn.pack_schema = "p20-panel-nk-v1", ncnn.pack_tile_k = 128 : i64, ncnn.packed_weight = true, ncnn.weight_layout = "panel_nk"}
    scf.forall (%arg2, %arg3) = (0, 0) to (1024, 64) step (32, 32) {
      %subview = memref.subview %arg0[%arg2, 0] [32, 576] [1, 1] : memref<1024x576xf32> to memref<32x576xf32, strided<[576, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg2, %arg3] [32, 32] [1, 1] : memref<1024x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %1 [[0, 1]] : memref<576x32xf32> into memref<18432xf32>
      scf.for %arg4 = %c0 to %c32 step %c4 {
        scf.for %arg5 = %c0 to %c32 step %c16 {
          %2 = arith.addi %arg4, %c1 : index
          %3 = arith.addi %arg4, %c2 : index
          %4 = arith.addi %arg4, %c3 : index
          %5 = vector.transfer_read %subview_0[%arg4, %arg5], %0 {in_bounds = [true]} : memref<32x32xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
          %6 = vector.transfer_read %subview_0[%2, %arg5], %0 {in_bounds = [true]} : memref<32x32xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
          %7 = vector.transfer_read %subview_0[%3, %arg5], %0 {in_bounds = [true]} : memref<32x32xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
          %8 = vector.transfer_read %subview_0[%4, %arg5], %0 {in_bounds = [true]} : memref<32x32xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
          %9:4 = scf.for %arg6 = %c0 to %c576 step %c1 iter_args(%arg7 = %5, %arg8 = %6, %arg9 = %7, %arg10 = %8) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
            %10 = arith.divui %arg5, %c16 : index
            %11 = arith.remui %arg5, %c16 : index
            %12 = arith.muli %10, %c9216 : index
            %13 = arith.muli %arg6, %c16 : index
            %14 = arith.addi %12, %13 : index
            %15 = arith.addi %14, %11 : index
            %16 = vector.transfer_read %collapse_shape[%15], %0 {in_bounds = [true]} : memref<18432xf32>, vector<16xf32>
            %17 = memref.load %subview[%arg4, %arg6] : memref<32x576xf32, strided<[576, 1], offset: ?>>
            %18 = vector.broadcast %17 : f32 to vector<16xf32>
            %19 = vector.fma %18, %16, %arg7 : vector<16xf32>
            %20 = memref.load %subview[%2, %arg6] : memref<32x576xf32, strided<[576, 1], offset: ?>>
            %21 = vector.broadcast %20 : f32 to vector<16xf32>
            %22 = vector.fma %21, %16, %arg8 : vector<16xf32>
            %23 = memref.load %subview[%3, %arg6] : memref<32x576xf32, strided<[576, 1], offset: ?>>
            %24 = vector.broadcast %23 : f32 to vector<16xf32>
            %25 = vector.fma %24, %16, %arg9 : vector<16xf32>
            %26 = memref.load %subview[%4, %arg6] : memref<32x576xf32, strided<[576, 1], offset: ?>>
            %27 = vector.broadcast %26 : f32 to vector<16xf32>
            %28 = vector.fma %27, %16, %arg10 : vector<16xf32>
            scf.yield %19, %22, %25, %28 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
          }
          vector.transfer_write %9#0, %subview_0[%arg4, %arg5] {in_bounds = [true]} : vector<16xf32>, memref<32x32xf32, strided<[64, 1], offset: ?>>
          vector.transfer_write %9#1, %subview_0[%2, %arg5] {in_bounds = [true]} : vector<16xf32>, memref<32x32xf32, strided<[64, 1], offset: ?>>
          vector.transfer_write %9#2, %subview_0[%3, %arg5] {in_bounds = [true]} : vector<16xf32>, memref<32x32xf32, strided<[64, 1], offset: ?>>
          vector.transfer_write %9#3, %subview_0[%4, %arg5] {in_bounds = [true]} : vector<16xf32>, memref<32x32xf32, strided<[64, 1], offset: ?>>
        }
      }
      scf.for %arg4 = %c0 to %c32 step %c1 {
        scf.for %arg5 = %c0 to %c32 step %c1 {
          %2 = memref.load %subview_0[%arg4, %arg5] : memref<32x32xf32, strided<[64, 1], offset: ?>>
          memref.store %2, %subview_0[%arg4, %arg5] : memref<32x32xf32, strided<[64, 1], offset: ?>>
        }
      }
    } {ncnn.alias = "disjoint_output_proven", ncnn.alignment = "unknown", ncnn.contract = "selected", ncnn.fma = "vector.fma", ncnn.input_layout = "identity", ncnn.kernel = "f32_packed_mxn_fma", ncnn.layout = "identity", ncnn.output_layout = "identity", ncnn.pack_bytes = 147456 : i64, ncnn.pack_factor = 16 : i64, ncnn.pack_raw_bytes = 147456 : i64, ncnn.pack_runtime = "compile_time_B", ncnn.pack_schema = "p20-panel-nk-v1", ncnn.pack_tile_k = 128 : i64, ncnn.packing = "prepacked_B", ncnn.parallel = "outer_tile+inner_simd", ncnn.simd_chunk = 16 : i64, ncnn.tail = "none", ncnn.tile_k = 128 : i64, ncnn.tile_m = 4 : i64, ncnn.tile_n = 16 : i64, ncnn.unpack_bytes = 0 : i64, ncnn.weight_layout = "panel_nk"}
    return
  }
  func.func @bench_deep(%arg0: memref<256x2304xf32>, %arg1: memref<256x256xf32> {bufferize.result}) attributes {llvm.emit_c_interface} {
    %c36864 = arith.constant 36864 : index
    %c128 = arith.constant 128 : index
    %c2304 = arith.constant 2304 : index
    %0 = ub.poison : f32
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %1 = memref.get_global @__constant_2304x32xf32 : memref<2304x32xf32> {ncnn.alignment = "64", ncnn.pack_bytes = 2359296 : i64, ncnn.pack_factor = 16 : i64, ncnn.pack_runtime = "compile_time_B", ncnn.pack_schema = "p20-panel-nk-v1", ncnn.pack_tile_k = 128 : i64, ncnn.packed_weight = true, ncnn.weight_layout = "panel_nk"}
    scf.forall (%arg2, %arg3) = (0, 0) to (256, 256) step (32, 32) {
      %subview = memref.subview %arg0[%arg2, 0] [32, 2304] [1, 1] : memref<256x2304xf32> to memref<32x2304xf32, strided<[2304, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg2, %arg3] [32, 32] [1, 1] : memref<256x256xf32> to memref<32x32xf32, strided<[256, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %1 [[0, 1]] : memref<2304x32xf32> into memref<73728xf32>
      scf.for %arg4 = %c0 to %c32 step %c4 {
        scf.for %arg5 = %c0 to %c32 step %c16 {
          %2 = arith.addi %arg4, %c1 : index
          %3 = arith.addi %arg4, %c2 : index
          %4 = arith.addi %arg4, %c3 : index
          %5 = vector.transfer_read %subview_0[%arg4, %arg5], %0 {in_bounds = [true]} : memref<32x32xf32, strided<[256, 1], offset: ?>>, vector<16xf32>
          %6 = vector.transfer_read %subview_0[%2, %arg5], %0 {in_bounds = [true]} : memref<32x32xf32, strided<[256, 1], offset: ?>>, vector<16xf32>
          %7 = vector.transfer_read %subview_0[%3, %arg5], %0 {in_bounds = [true]} : memref<32x32xf32, strided<[256, 1], offset: ?>>, vector<16xf32>
          %8 = vector.transfer_read %subview_0[%4, %arg5], %0 {in_bounds = [true]} : memref<32x32xf32, strided<[256, 1], offset: ?>>, vector<16xf32>
          %9:4 = scf.for %arg6 = %c0 to %c2304 step %c128 iter_args(%arg7 = %5, %arg8 = %6, %arg9 = %7, %arg10 = %8) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
            %10 = arith.addi %arg6, %c128 : index
            %11:4 = scf.for %arg11 = %arg6 to %10 step %c1 iter_args(%arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9, %arg15 = %arg10) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
              %12 = arith.divui %arg5, %c16 : index
              %13 = arith.remui %arg5, %c16 : index
              %14 = arith.muli %12, %c36864 : index
              %15 = arith.muli %arg11, %c16 : index
              %16 = arith.addi %14, %15 : index
              %17 = arith.addi %16, %13 : index
              %18 = vector.transfer_read %collapse_shape[%17], %0 {in_bounds = [true]} : memref<73728xf32>, vector<16xf32>
              %19 = memref.load %subview[%arg4, %arg11] : memref<32x2304xf32, strided<[2304, 1], offset: ?>>
              %20 = vector.broadcast %19 : f32 to vector<16xf32>
              %21 = vector.fma %20, %18, %arg12 : vector<16xf32>
              %22 = memref.load %subview[%2, %arg11] : memref<32x2304xf32, strided<[2304, 1], offset: ?>>
              %23 = vector.broadcast %22 : f32 to vector<16xf32>
              %24 = vector.fma %23, %18, %arg13 : vector<16xf32>
              %25 = memref.load %subview[%3, %arg11] : memref<32x2304xf32, strided<[2304, 1], offset: ?>>
              %26 = vector.broadcast %25 : f32 to vector<16xf32>
              %27 = vector.fma %26, %18, %arg14 : vector<16xf32>
              %28 = memref.load %subview[%4, %arg11] : memref<32x2304xf32, strided<[2304, 1], offset: ?>>
              %29 = vector.broadcast %28 : f32 to vector<16xf32>
              %30 = vector.fma %29, %18, %arg15 : vector<16xf32>
              scf.yield %21, %24, %27, %30 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
            }
            scf.yield %11#0, %11#1, %11#2, %11#3 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
          }
          vector.transfer_write %9#0, %subview_0[%arg4, %arg5] {in_bounds = [true]} : vector<16xf32>, memref<32x32xf32, strided<[256, 1], offset: ?>>
          vector.transfer_write %9#1, %subview_0[%2, %arg5] {in_bounds = [true]} : vector<16xf32>, memref<32x32xf32, strided<[256, 1], offset: ?>>
          vector.transfer_write %9#2, %subview_0[%3, %arg5] {in_bounds = [true]} : vector<16xf32>, memref<32x32xf32, strided<[256, 1], offset: ?>>
          vector.transfer_write %9#3, %subview_0[%4, %arg5] {in_bounds = [true]} : vector<16xf32>, memref<32x32xf32, strided<[256, 1], offset: ?>>
        }
      }
      scf.for %arg4 = %c0 to %c32 step %c1 {
        scf.for %arg5 = %c0 to %c32 step %c1 {
          %2 = memref.load %subview_0[%arg4, %arg5] : memref<32x32xf32, strided<[256, 1], offset: ?>>
          memref.store %2, %subview_0[%arg4, %arg5] : memref<32x32xf32, strided<[256, 1], offset: ?>>
        }
      }
    } {ncnn.alias = "disjoint_output_proven", ncnn.alignment = "unknown", ncnn.contract = "selected", ncnn.fma = "vector.fma", ncnn.input_layout = "identity", ncnn.kernel = "f32_packed_mxn_fma", ncnn.layout = "identity", ncnn.output_layout = "identity", ncnn.pack_bytes = 2359296 : i64, ncnn.pack_factor = 16 : i64, ncnn.pack_raw_bytes = 2359296 : i64, ncnn.pack_runtime = "compile_time_B", ncnn.pack_schema = "p20-panel-nk-v1", ncnn.pack_tile_k = 128 : i64, ncnn.packing = "prepacked_B", ncnn.parallel = "outer_tile+inner_simd", ncnn.simd_chunk = 16 : i64, ncnn.tail = "none", ncnn.tile_k = 128 : i64, ncnn.tile_m = 4 : i64, ncnn.tile_n = 16 : i64, ncnn.unpack_bytes = 0 : i64, ncnn.weight_layout = "panel_nk"}
    return
  }
}
