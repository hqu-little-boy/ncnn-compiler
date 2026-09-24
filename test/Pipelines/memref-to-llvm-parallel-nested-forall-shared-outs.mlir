// RUN: not ncnn-mlir-opt '--ncnn-memref-to-llvm-pipeline=threads=4' %s 2>&1 | FileCheck %s

// Exercise serial lowering of a nested forall with shared tensor outputs.
module {
  func.func @nested_shared_outs(%input: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %init = tensor.empty() : tensor<4x4xf32>
    %result = scf.forall (%i) in (%c4) shared_outs(%outer = %init) -> tensor<4x4xf32> {
      %row = tensor.extract_slice %input[%i, 0] [1, 4] [1, 1]
        : tensor<4x4xf32> to tensor<1x4xf32>
      %inner = scf.forall (%j) in (%c4) shared_outs(%inner_out = %row) -> tensor<1x4xf32> {
        %value = tensor.extract %row[%c0, %j] : tensor<1x4xf32>
        %piece = tensor.from_elements %value : tensor<1xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %piece into %inner_out[0, %j] [1, 1] [1, 1]
            : tensor<1xf32> into tensor<1x4xf32>
        }
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %inner into %outer[%i, 0] [1, 4] [1, 1]
          : tensor<1x4xf32> into tensor<4x4xf32>
      }
    }
    return %result : tensor<4x4xf32>
  }
}

// CHECK: error: cannot serialize nested scf.forall with shared outputs
