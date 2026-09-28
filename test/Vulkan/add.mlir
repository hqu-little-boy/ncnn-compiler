spirv.module @add Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, api=Vulkan, #spirv.resource_limits<>>
} {
  spirv.GlobalVariable @gid built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @lhs bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.array<257xf32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @rhs bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.array<257xf32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @out bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.array<257xf32, stride=4> [0])>, StorageBuffer>

  spirv.func @add() "None" attributes {
    spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [64, 1, 1]>
  } {
    %gid_ptr = spirv.mlir.addressof @gid : !spirv.ptr<vector<3xi32>, Input>
    %gid_vec = spirv.Load "Input" %gid_ptr : vector<3xi32>
    %index = spirv.CompositeExtract %gid_vec[0 : i32] : vector<3xi32>
    %limit = spirv.Constant 257 : i32
    %in_bounds = spirv.ULessThan %index, %limit : i32
    spirv.mlir.selection {
      spirv.BranchConditional %in_bounds, ^bb1, ^bb2
    ^bb1:
      %zero = spirv.Constant 0 : i32
      %lhs_ptr = spirv.mlir.addressof @lhs : !spirv.ptr<!spirv.struct<(!spirv.array<257xf32, stride=4> [0])>, StorageBuffer>
      %rhs_ptr = spirv.mlir.addressof @rhs : !spirv.ptr<!spirv.struct<(!spirv.array<257xf32, stride=4> [0])>, StorageBuffer>
      %out_ptr = spirv.mlir.addressof @out : !spirv.ptr<!spirv.struct<(!spirv.array<257xf32, stride=4> [0])>, StorageBuffer>
      %lhs_elem = spirv.AccessChain %lhs_ptr[%zero, %index] : !spirv.ptr<!spirv.struct<(!spirv.array<257xf32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %rhs_elem = spirv.AccessChain %rhs_ptr[%zero, %index] : !spirv.ptr<!spirv.struct<(!spirv.array<257xf32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %out_elem = spirv.AccessChain %out_ptr[%zero, %index] : !spirv.ptr<!spirv.struct<(!spirv.array<257xf32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %a = spirv.Load "StorageBuffer" %lhs_elem : f32
      %b = spirv.Load "StorageBuffer" %rhs_elem : f32
      %sum = spirv.FAdd %a, %b : f32
      spirv.Store "StorageBuffer" %out_elem, %sum : f32
      spirv.Branch ^bb2
    ^bb2:
      spirv.mlir.merge
    }
    spirv.Return
  }
}
