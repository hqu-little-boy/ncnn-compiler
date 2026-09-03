# fixture 编译计时包装（追平计划 P0：编译耗时预算告警）。
#
# 用法（add_custom_command 内）：
#   cmake -Dlabel=<fixture 名> [-Dbudget_seconds=<N>] -P timed_compile.cmake
#        -- <命令> <参数...>
#
# 行为：透传执行命令（stdout/stderr 不捕获，保持构建实时输出），结束后
# 打印 STATUS 耗时行（构建日志内 `-- fixture <名> 编译耗时 <N>s`，可
# grep 汇总）；超过 budget_seconds（默认 300，对应追平计划"单 fixture
# 编译预算 5 分钟"）升级为 WARNING——编译爆炸（超宽向量 LLVM 合法化）
# 的早期预警，见 docs/ncnn-performance-parity-plan.md §1/§3-P0。

set(timed_compile_command "")
set(timed_compile_seen_separator FALSE)
math(EXPR timed_compile_last "${CMAKE_ARGC} - 1")
foreach(timed_compile_index RANGE 0 ${timed_compile_last})
  set(timed_compile_argument "${CMAKE_ARGV${timed_compile_index}}")
  if(timed_compile_seen_separator)
    list(APPEND timed_compile_command "${timed_compile_argument}")
  elseif(timed_compile_argument STREQUAL "--")
    set(timed_compile_seen_separator TRUE)
  endif()
endforeach()
if(NOT timed_compile_command)
  message(FATAL_ERROR "timed_compile.cmake: 需要在 -- 之后给出待执行命令")
endif()
if(NOT DEFINED budget_seconds)
  set(budget_seconds 300)
endif()

string(TIMESTAMP timed_compile_start "%s")
execute_process(COMMAND ${timed_compile_command} RESULT_VARIABLE timed_compile_result)
string(TIMESTAMP timed_compile_end "%s")
math(EXPR timed_compile_elapsed "${timed_compile_end} - ${timed_compile_start}")

if(NOT timed_compile_result EQUAL 0)
  message(FATAL_ERROR
    "fixture ${label} 编译失败（exit=${timed_compile_result}，耗时 "
    "${timed_compile_elapsed}s）")
endif()
message(STATUS "fixture ${label} 编译耗时 ${timed_compile_elapsed}s")
if(timed_compile_elapsed GREATER budget_seconds)
  message(WARNING
    "fixture ${label} 编译耗时 ${timed_compile_elapsed}s 超过预算 "
    "${budget_seconds}s——疑似编译爆炸回归（ncnn-performance-parity-plan.md P0）")
endif()
