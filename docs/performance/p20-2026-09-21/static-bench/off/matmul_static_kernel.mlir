module attributes {ncnn.int8_depthwise = false, ncnn.int8_kernel = "portable", ncnn.int8_target = "portable"} {
  memref.global "private" constant @__constant_2304x32xf32 : memref<2304x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_576x32xf32 : memref<576x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @bench_shallow(%arg0: memref<1024x576xf32>, %arg1: memref<1024x64xf32> {bufferize.result}) attributes {llvm.emit_c_interface} {
    %c576 = arith.constant 576 : index
    %0 = ub.poison : f32
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %1 = memref.get_global @__constant_576x32xf32 : memref<576x32xf32>
    scf.forall (%arg2, %arg3) = (0, 0) to (1024, 64) step (32, 32) {
      %subview = memref.subview %arg0[%arg2, 0] [32, 576] [1, 1] : memref<1024x576xf32> to memref<32x576xf32, strided<[576, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg2, %arg3] [32, 32] [1, 1] : memref<1024x64xf32> to memref<32x32xf32, strided<[64, 1], offset: ?>>
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
            %10 = vector.transfer_read %1[%arg6, %arg5], %0 {in_bounds = [true]} : memref<576x32xf32>, vector<16xf32>
            %11 = memref.load %subview[%arg4, %arg6] : memref<32x576xf32, strided<[576, 1], offset: ?>>
            %12 = vector.broadcast %11 : f32 to vector<16xf32>
            %13 = vector.fma %12, %10, %arg7 : vector<16xf32>
            %14 = memref.load %subview[%2, %arg6] : memref<32x576xf32, strided<[576, 1], offset: ?>>
            %15 = vector.broadcast %14 : f32 to vector<16xf32>
            %16 = vector.fma %15, %10, %arg8 : vector<16xf32>
            %17 = memref.load %subview[%3, %arg6] : memref<32x576xf32, strided<[576, 1], offset: ?>>
            %18 = vector.broadcast %17 : f32 to vector<16xf32>
            %19 = vector.fma %18, %10, %arg9 : vector<16xf32>
            %20 = memref.load %subview[%4, %arg6] : memref<32x576xf32, strided<[576, 1], offset: ?>>
            %21 = vector.broadcast %20 : f32 to vector<16xf32>
            %22 = vector.fma %21, %10, %arg10 : vector<16xf32>
            scf.yield %13, %16, %19, %22 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
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
    } {ncnn.alias = "disjoint_output_proven", ncnn.alignment = "unknown", ncnn.contract = "selected", ncnn.fallback_reason = "unpacked_direct", ncnn.fma = "vector.fma", ncnn.input_layout = "identity", ncnn.kernel = "f32_mxn_fma", ncnn.layout = "identity", ncnn.output_layout = "identity", ncnn.pack_bytes = 0 : i64, ncnn.pack_factor = 1 : i64, ncnn.packing = "unpacked", ncnn.parallel = "outer_tile+inner_simd", ncnn.simd_chunk = 16 : i64, ncnn.tail = "none", ncnn.tile_k = 576 : i64, ncnn.tile_m = 4 : i64, ncnn.tile_n = 16 : i64, ncnn.unpack_bytes = 0 : i64, ncnn.weight_layout = "row_major_kxn"}
    return
  }
  func.func @bench_deep(%arg0: memref<256x2304xf32>, %arg1: memref<256x256xf32> {bufferize.result}) attributes {llvm.emit_c_interface} {
    %c2304 = arith.constant 2304 : index
    %0 = ub.poison : f32
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %1 = memref.get_global @__constant_2304x32xf32 : memref<2304x32xf32>
    scf.forall (%arg2, %arg3) = (0, 0) to (256, 256) step (32, 32) {
      %subview = memref.subview %arg0[%arg2, 0] [32, 2304] [1, 1] : memref<256x2304xf32> to memref<32x2304xf32, strided<[2304, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg2, %arg3] [32, 32] [1, 1] : memref<256x256xf32> to memref<32x32xf32, strided<[256, 1], offset: ?>>
      scf.for %arg4 = %c0 to %c32 step %c4 {
        scf.for %arg5 = %c0 to %c32 step %c16 {
          %2 = arith.addi %arg4, %c1 : index
          %3 = arith.addi %arg4, %c2 : index
          %4 = arith.addi %arg4, %c3 : index
          %5 = vector.transfer_read %subview_0[%arg4, %arg5], %0 {in_bounds = [true]} : memref<32x32xf32, strided<[256, 1], offset: ?>>, vector<16xf32>
          %6 = vector.transfer_read %subview_0[%2, %arg5], %0 {in_bounds = [true]} : memref<32x32xf32, strided<[256, 1], offset: ?>>, vector<16xf32>
          %7 = vector.transfer_read %subview_0[%3, %arg5], %0 {in_bounds = [true]} : memref<32x32xf32, strided<[256, 1], offset: ?>>, vector<16xf32>
          %8 = vector.transfer_read %subview_0[%4, %arg5], %0 {in_bounds = [true]} : memref<32x32xf32, strided<[256, 1], offset: ?>>, vector<16xf32>
          %9:4 = scf.for %arg6 = %c0 to %c2304 step %c1 iter_args(%arg7 = %5, %arg8 = %6, %arg9 = %7, %arg10 = %8) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
            %10 = vector.transfer_read %1[%arg6, %arg5], %0 {in_bounds = [true]} : memref<2304x32xf32>, vector<16xf32>
            %11 = memref.load %subview[%arg4, %arg6] : memref<32x2304xf32, strided<[2304, 1], offset: ?>>
            %12 = vector.broadcast %11 : f32 to vector<16xf32>
            %13 = vector.fma %12, %10, %arg7 : vector<16xf32>
            %14 = memref.load %subview[%2, %arg6] : memref<32x2304xf32, strided<[2304, 1], offset: ?>>
            %15 = vector.broadcast %14 : f32 to vector<16xf32>
            %16 = vector.fma %15, %10, %arg8 : vector<16xf32>
            %17 = memref.load %subview[%3, %arg6] : memref<32x2304xf32, strided<[2304, 1], offset: ?>>
            %18 = vector.broadcast %17 : f32 to vector<16xf32>
            %19 = vector.fma %18, %10, %arg9 : vector<16xf32>
            %20 = memref.load %subview[%4, %arg6] : memref<32x2304xf32, strided<[2304, 1], offset: ?>>
            %21 = vector.broadcast %20 : f32 to vector<16xf32>
            %22 = vector.fma %21, %10, %arg10 : vector<16xf32>
            scf.yield %13, %16, %19, %22 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
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
    } {ncnn.alias = "disjoint_output_proven", ncnn.alignment = "unknown", ncnn.contract = "selected", ncnn.fallback_reason = "unpacked_direct", ncnn.fma = "vector.fma", ncnn.input_layout = "identity", ncnn.kernel = "f32_mxn_fma", ncnn.layout = "identity", ncnn.output_layout = "identity", ncnn.pack_bytes = 0 : i64, ncnn.pack_factor = 1 : i64, ncnn.packing = "unpacked", ncnn.parallel = "outer_tile+inner_simd", ncnn.simd_chunk = 16 : i64, ncnn.tail = "none", ncnn.tile_k = 2304 : i64, ncnn.tile_m = 4 : i64, ncnn.tile_n = 16 : i64, ncnn.unpack_bytes = 0 : i64, ncnn.weight_layout = "row_major_kxn"}
    return
  }
}
