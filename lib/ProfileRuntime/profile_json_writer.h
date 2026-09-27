/* profile_json_writer.h —— profile runtime 的 JSON 编码原语。
 *
 * 职责
 *   只负责「值怎么编成 JSON 字节」：字符串转义（含 UTF-8 校验）、整数/浮点/
 *   布尔/null 的字面量、对象/数组的定界符与键名。**不负责 profile schema**
 *   （哪些键、什么顺序、什么缩进）——那部分留在 profile_runtime.c。
 *
 * 硬约束（为什么不用现成库）
 *   1. 本文件以 `.c` 源形式 install，由 ncnn-compile 在**每个模型编译期**
 *      随产物编进生成的 `.so`（见 tools/CMakeLists.txt 的 install(FILES)）。
 *      因此必须是纯 C11、零第三方依赖。
 *   2. 生成的 `.so` 受未定义符号白名单 profile_allowed 约束
 *      （tools/ncnn-compile.cpp 的 undefined-symbol 审计）。任何新符号都
 *      必须落在 libc 已有符号内——引 LLVM Support 会打破 P26 归档的正面
 *      性质「.so 只依赖 libomp + libc + libm」。
 *   3. 也不能引 strlen：白名单没有它，而手写的「数到 NUL」循环会被 clang
 *      idiomize 回 strlen。写字符串一律按 `*cursor != '\0'` 走，UTF-8 校验
 *      直接把 NUL 当非法续字节截断（续字节必在 0x80–0xBF）。
 *   4. 因此 **不用 llvm::json**（需要 LLVM Support）、**不用 qsort**
 *      （要带 comparator 与 stdlib 依赖面，且本处排序已在调用侧手写插入/
 *      归并排序以保稳定）、**不用 strtoul**（解析侧已有 parse_unsigned，
 *      且 strtoul 的错误语义无法区分「0」与「非法」）。
 *
 * 浮点输出
 *   数值字段绝大多数走 pj_u64（整数，避开手写 JSON writer 最经典的
 *   NaN/Infinity 陷阱）。仅 share / projection 三个**既有**字段需要小数：
 *     wall_attributed_share_of_sampled_window
 *     wall_projection_factor
 *     wall_coverage_share
 *   它们是「已存在」而非本次新增，值域已由调用侧约束为有限小数。
 *   **不要**用 pj_f64 新增任何字段。
 *
 * UTF-8
 *   pj_string 对 ≥0x80 的字节做 RFC 3629 校验：合法多字节序列原样透传
 *   （层名里的中文/日文保持可读），非法字节按「最大非法子部分」输出一个
 *   JSON 转义 ��。不把非法字节当码位转成 \uXXXX——那是伪造数据；
 *   U+FFFD 明确表示「此处原有不可解码内容」。合法 ASCII 路径逐字节不变。
 */
#ifndef NCNN_PROFILE_JSON_WRITER_H
#define NCNN_PROFILE_JSON_WRITER_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 定界符。缩进与换行由调用侧自己写：那是 schema 的布局，不是编码。 */
void pj_object_start(FILE* file); /* "{"  */
void pj_object_end(FILE* file);   /* "}"  */
void pj_array_start(FILE* file);  /* "["  */
void pj_array_end(FILE* file);    /* "]"  */

/* 键名：写入 "key":␠（含前后的双引号与冒号空格）。键名只接受 ASCII，
 * 不做转义之外的变换；含引号/反斜杠/控制字符时按 pj_string 同规则转义。 */
void pj_key(FILE* file, const char* key);

/* 值。 */
void pj_string(FILE* file, const char* value); /* 含引号与转义 */
void pj_u64(FILE* file, uint64_t value);
void pj_i64(FILE* file, int64_t value); /* 带符号；本 runtime 暂无调用点 */
void pj_bool(FILE* file, int value);    /* "true" / "false" */
void pj_null(FILE* file);               /* "null" */
void pj_f64(FILE* file, double value);  /* %.9f；仅既有 share/projection 字段 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NCNN_PROFILE_JSON_WRITER_H */
