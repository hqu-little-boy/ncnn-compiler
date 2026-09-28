spirv.module @add Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Shader], []>, api=Vulkan, #spirv.resource_limits<>>} {
  spirv.GlobalVariable @gid built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @lhs bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.array<257 x f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @rhs bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.array<257 x f32, stride=4> [0])>, StorageBuffer>
  spirv.GlobalVariable @out bind(0, 2) : !spirv.ptr<!spirv.struct<(!spirv.array<257 x f32, stride=4> [0])>, StorageBuffer>
  spirv.func @add() "None" {
    %gid_addr = spirv.mlir.addressof @gid : !spirv.ptr<vector<3xi32>, Input>
    %0 = spirv.Load "Input" %gid_addr : vector<3xi32>
    %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi32>
    %cst257_i32 = spirv.Constant 257 : i32
    %2 = spirv.ULessThan %1, %cst257_i32 : i32
    spirv.mlir.selection {
      spirv.BranchConditional %2, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %cst0_i32 = spirv.Constant 0 : i32
      %lhs_addr = spirv.mlir.addressof @lhs : !spirv.ptr<!spirv.struct<(!spirv.array<257 x f32, stride=4> [0])>, StorageBuffer>
      %rhs_addr = spirv.mlir.addressof @rhs : !spirv.ptr<!spirv.struct<(!spirv.array<257 x f32, stride=4> [0])>, StorageBuffer>
      %out_addr = spirv.mlir.addressof @out : !spirv.ptr<!spirv.struct<(!spirv.array<257 x f32, stride=4> [0])>, StorageBuffer>
      %3 = spirv.AccessChain %lhs_addr[%cst0_i32, %1] : !spirv.ptr<!spirv.struct<(!spirv.array<257 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %4 = spirv.AccessChain %rhs_addr[%cst0_i32, %1] : !spirv.ptr<!spirv.struct<(!spirv.array<257 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %5 = spirv.AccessChain %out_addr[%cst0_i32, %1] : !spirv.ptr<!spirv.struct<(!spirv.array<257 x f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
      %6 = spirv.Load "StorageBuffer" %3 : f32
      %7 = spirv.Load "StorageBuffer" %4 : f32
      %8 = spirv.FAdd %6, %7 : f32
      spirv.Store "StorageBuffer" %5, %8 : f32
      spirv.Branch ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      spirv.mlir.merge
    }
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @add, @gid
  spirv.ExecutionMode @add "LocalSize", 64, 1, 1
}

