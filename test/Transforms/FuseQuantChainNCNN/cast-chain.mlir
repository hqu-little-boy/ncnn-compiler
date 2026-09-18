// RUN: ncnn-mlir-opt --fuse-quant-chain-ncnn --split-input-file %s | FileCheck %s

// P16 map-consumer 规范化（module 属性 ncnn.int8_cast_chain 选择进入）：
// linalg.map 作为 consumer 的 cast 尾部（如 dequant 后再接 fptrunc）先等价
// 重写为 generic（body 逐语句克隆，保留每个 cast、无抵消/重排），使既有的
// generic-consumer 融合能吸收单用户逐元素 producer，整链收拢为单一 generic。

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#mapb = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>

// 门开启：sitofp map → mulf generic → fptrunc map 收拢为单一 generic。
// CHECK-LABEL: func.func @cast_chain_normalized
module attributes {ncnn.int8_cast_chain = true} {
  func.func @cast_chain_normalized(%arg0: tensor<1x3x4x3xi32>) -> tensor<1x3x4x3xf16> {
    %cst_scale = arith.constant dense<[[[[5.0, 6.5, 8.0]]]]> : tensor<1x1x1x3xf32>
    %e0 = tensor.empty() : tensor<1x3x4x3xf32>
    %0 = linalg.map { arith.sitofp } ins(%arg0 : tensor<1x3x4x3xi32>) outs(%e0 : tensor<1x3x4x3xf32>)
    %e1 = tensor.empty() : tensor<1x3x4x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #mapb, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %cst_scale : tensor<1x3x4x3xf32>, tensor<1x1x1x3xf32>) outs(%e1 : tensor<1x3x4x3xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %3 = arith.mulf %in, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<1x3x4x3xf32>
    %e2 = tensor.empty() : tensor<1x3x4x3xf16>
    %2 = linalg.map { arith.truncf } ins(%1 : tensor<1x3x4x3xf32>) outs(%e2 : tensor<1x3x4x3xf16>)
    // CHECK-NOT: linalg.map
    // CHECK: linalg.generic
    // CHECK-SAME: ins({{.*}}tensor<1x1x1x3xf32>, tensor<1x3x4x3xi32>
    // CHECK-SAME: outs({{.*}}tensor<1x3x4x3xf16>
    // body 原序：sitofp → mulf → fptrunc，cast 全部保留。
    // CHECK: arith.sitofp
    // CHECK: arith.mulf
    // CHECK: arith.truncf
    // CHECK-NOT: linalg.generic
    // CHECK: return
    return %2 : tensor<1x3x4x3xf16>
  }
}

// -----

// map body 读取多个输入块参（numInputs 不再硬编码 1）：双输入 addf map 与
// 单用户 sitofp map producer 收拢。
// CHECK-LABEL: func.func @multi_input_map
module attributes {ncnn.int8_cast_chain = true} {
  func.func @multi_input_map(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %e0 = tensor.empty() : tensor<4xf32>
    %0 = linalg.map { arith.sitofp } ins(%arg0 : tensor<4xi32>) outs(%e0 : tensor<4xf32>)
    %e1 = tensor.empty() : tensor<4xf32>
    %1 = linalg.map ins(%0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%e1 : tensor<4xf32>)
      (%in: f32, %in_1: f32) {
      %2 = arith.addf %in, %in_1 : f32
      linalg.yield %2 : f32
    }
    // CHECK-NOT: linalg.map
    // CHECK: linalg.generic
    // CHECK-SAME: ins({{.*}}tensor<4xf32>{{.*}}tensor<4xi32>
    // CHECK: arith.sitofp
    // CHECK: arith.addf
    // CHECK: return
    return %1 : tensor<4xf32>
  }
}

// -----

// 门未开（无 module 属性）：map consumer 不规范化，保持 linalg.map。
// CHECK-LABEL: func.func @cast_chain_gate_off
func.func @cast_chain_gate_off(%arg0: tensor<4xi32>) -> tensor<4xf16> {
  %e0 = tensor.empty() : tensor<4xf32>
  %0 = linalg.map { arith.sitofp } ins(%arg0 : tensor<4xi32>) outs(%e0 : tensor<4xf32>)
  %e1 = tensor.empty() : tensor<4xf16>
  %1 = linalg.map { arith.truncf } ins(%0 : tensor<4xf32>) outs(%e1 : tensor<4xf16>)
  // CHECK: linalg.map
  // CHECK: arith.sitofp
  // CHECK: linalg.map
  // CHECK: arith.truncf
  // CHECK: return
  return %1 : tensor<4xf16>
}

// -----

// 多用户 producer 不规范化（融合只吸收单用户链条节点）。
// CHECK-LABEL: func.func @multi_use_fallback
module attributes {ncnn.int8_cast_chain = true} {
  func.func @multi_use_fallback(%arg0: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xf16>) {
    %e0 = tensor.empty() : tensor<4xf32>
    %0 = linalg.map { arith.sitofp } ins(%arg0 : tensor<4xi32>) outs(%e0 : tensor<4xf32>)
    %e1 = tensor.empty() : tensor<4xf16>
    %1 = linalg.map { arith.truncf } ins(%0 : tensor<4xf32>) outs(%e1 : tensor<4xf16>)
    // CHECK: linalg.map
    // CHECK: arith.sitofp
    // CHECK: linalg.map
    // CHECK: arith.truncf
    // CHECK: return
    return %0, %1 : tensor<4xf32>, tensor<4xf16>
  }
}

// -----

// 动态 shape 不规范化（静态 shape 门禁）。
// CHECK-LABEL: func.func @dynamic_fallback
module attributes {ncnn.int8_cast_chain = true} {
  func.func @dynamic_fallback(%arg0: tensor<?xi32>) -> tensor<?xf16> {
    %c0 = arith.constant 0 : index
    %d0 = tensor.dim %arg0, %c0 : tensor<?xi32>
    %e0 = tensor.empty(%d0) : tensor<?xf32>
    %0 = linalg.map { arith.sitofp } ins(%arg0 : tensor<?xi32>) outs(%e0 : tensor<?xf32>)
    %e1 = tensor.empty(%d0) : tensor<?xf16>
    %1 = linalg.map { arith.truncf } ins(%0 : tensor<?xf32>) outs(%e1 : tensor<?xf16>)
    // CHECK: linalg.map
    // CHECK: arith.sitofp
    // CHECK: linalg.map
    // CHECK: arith.truncf
    // CHECK: return
    return %1 : tensor<?xf16>
  }
}

// -----

// 非 linalg producer（arith.extsi）不规范化，cast map 原样保留。
// CHECK-LABEL: func.func @non_linalg_producer_fallback
module attributes {ncnn.int8_cast_chain = true} {
  func.func @non_linalg_producer_fallback(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    %0 = arith.extsi %arg0 : tensor<4xi32> to tensor<4xi64>
    %e1 = tensor.empty() : tensor<4xi32>
    %1 = linalg.map { arith.trunci } ins(%0 : tensor<4xi64>) outs(%e1 : tensor<4xi32>)
    // CHECK: arith.extsi
    // CHECK: linalg.map
    // CHECK: arith.trunci
    // CHECK: return
    return %1 : tensor<4xi32>
  }
}

// -----

// 非 linalg producer（matmul i32 累加器）不规范化，cast map 原样保留。
// CHECK-LABEL: func.func @matmul_producer_fallback
module attributes {ncnn.int8_cast_chain = true} {
  func.func @matmul_producer_fallback(%arg0: tensor<4x8xi32>, %arg1: tensor<8x4xi32>) -> tensor<4x4xi8> {
    %zero = arith.constant 0 : i32
    %e = tensor.empty() : tensor<4x4xi32>
    %fill = linalg.fill ins(%zero : i32) outs(%e : tensor<4x4xi32>) -> tensor<4x4xi32>
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<4x8xi32>, tensor<8x4xi32>) outs(%fill : tensor<4x4xi32>) -> tensor<4x4xi32>
    %e1 = tensor.empty() : tensor<4x4xi8>
    %1 = linalg.map { arith.trunci } ins(%0 : tensor<4x4xi32>) outs(%e1 : tensor<4x4xi8>)
    // CHECK: linalg.matmul
    // CHECK: linalg.map
    // CHECK: arith.trunci
    // CHECK: return
    return %1 : tensor<4x4xi8>
  }
}
