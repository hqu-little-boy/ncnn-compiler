module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, api=Vulkan, #spirv.resource_limits<>>} {
  gpu.module @v0_add {
    gpu.func @add(
        %lhs: memref<257xf32, #spirv.storage_class<StorageBuffer>>,
        %rhs: memref<257xf32, #spirv.storage_class<StorageBuffer>>,
        %out: memref<257xf32, #spirv.storage_class<StorageBuffer>>) kernel attributes {
          spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1]>
        } {
      %index = gpu.global_id x
      %size = arith.constant 257 : index
      %in_bounds = arith.cmpi ult, %index, %size : index
      scf.if %in_bounds {
        %a = memref.load %lhs[%index] : memref<257xf32, #spirv.storage_class<StorageBuffer>>
        %b = memref.load %rhs[%index] : memref<257xf32, #spirv.storage_class<StorageBuffer>>
        %sum = arith.addf %a, %b : f32
        memref.store %sum, %out[%index] : memref<257xf32, #spirv.storage_class<StorageBuffer>>
      }
      gpu.return
    }
  }
}
