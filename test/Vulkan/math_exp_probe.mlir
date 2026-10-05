module attributes {gpu.container_module, spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, api=Vulkan, #spirv.resource_limits<>>} {
  gpu.module @v52_exp {
    gpu.func @kernel_0(
        %input: memref<257xf32, #spirv.storage_class<StorageBuffer>>,
        %output: memref<257xf32, #spirv.storage_class<StorageBuffer>>) kernel attributes {
          spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1]>
        } {
      %index = gpu.global_id x
      %size = arith.constant 257 : index
      %in_bounds = arith.cmpi ult, %index, %size : index
      scf.if %in_bounds {
        %value = memref.load %input[%index] : memref<257xf32, #spirv.storage_class<StorageBuffer>>
        %exponent = math.exp %value : f32
        memref.store %exponent, %output[%index] : memref<257xf32, #spirv.storage_class<StorageBuffer>>
      }
      gpu.return
    }
  }
}
