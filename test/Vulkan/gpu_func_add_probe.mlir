module attributes {gpu.container_module} {
  gpu.module @v0_add {
    gpu.func @add(
        %lhs: memref<257xf32>,
        %rhs: memref<257xf32>,
        %out: memref<257xf32>) kernel attributes {
          spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1]>
        } {
      %index = gpu.global_id x
      %size = arith.constant 257 : index
      %in_bounds = arith.cmpi ult, %index, %size : index
      scf.if %in_bounds {
        %a = memref.load %lhs[%index] : memref<257xf32>
        %b = memref.load %rhs[%index] : memref<257xf32>
        %sum = arith.addf %a, %b : f32
        memref.store %sum, %out[%index] : memref<257xf32>
      }
      gpu.return
    }
  }
}
