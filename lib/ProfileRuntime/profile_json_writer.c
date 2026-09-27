/* profile_json_writer.c —— JSON 编码原语实现。
 *
 * 设计说明见 profile_json_writer.h。本文件只做「值 → JSON 字节」，不含
 * 任何 profile 语义，因而可以直接对转义行为做单元测试
 * （test/Native/check_profile_runtime.py 的转义边界用例）。
 *
 * 未定义符号面：只用 fputc / fputs / fprintf（libc）。这满足
 * tools/ncnn-compile.cpp 里 profile_allowed 白名单约束，新增符号会
 * 被 undefined-symbol 审计拒掉。
 */
#include "profile_json_writer.h"

#include <stddef.h>

void pj_object_start(FILE* file) {
  fputc('{', file);
}

void pj_object_end(FILE* file) {
  fputc('}', file);
}

void pj_array_start(FILE* file) {
  fputc('[', file);
}

void pj_array_end(FILE* file) {
  fputc(']', file);
}

/* RFC 3629：lead 字节决定序列总长度（1..4），lead 非法返回 0。
 * lo2/hi2 收紧第二字节区间，排除 overlong（C0/C1、F0·80）、UTF-16
 * 代理区（ED·A0..BF）与超出 U+10FFFF（F4·90..BF）的编码。 */
static size_t utf8_lead_length(const unsigned char lead,
                               unsigned char* lo2,
                               unsigned char* hi2) {
  *lo2 = 0x80;
  *hi2 = 0xBF;
  if (lead < 0x80) {
    return 1;
  }
  if (lead >= 0xC2 && lead <= 0xDF) {
    return 2;
  }
  if (lead == 0xE0) {
    *lo2 = 0xA0;
    return 3;
  }
  if (lead >= 0xE1 && lead <= 0xEC) {
    return 3;
  }
  if (lead == 0xED) {
    *hi2 = 0x9F;
    return 3;
  }
  if (lead >= 0xEE && lead <= 0xEF) {
    return 3;
  }
  if (lead == 0xF0) {
    *lo2 = 0x90;
    return 4;
  }
  if (lead >= 0xF1 && lead <= 0xF3) {
    return 4;
  }
  if (lead == 0xF4) {
    *hi2 = 0x8F;
    return 4;
  }
  return 0; /* 0x80–0xBF 孤立续字节、0xC0/0xC1 overlong、0xF5–0xFF */
}

/* 解析一个 UTF-8 序列，序列以 NUL 结尾。
 * 成功：*consumed = 序列长度，返回 1。
 * 失败：*consumed = 「最大非法子部分」长度（≥1），返回 0 —— 与 WHATWG
 * replacement 策略一致：截断的多字节只出一个 U+FFFD，不是每字节一个。
 *
 * 不传「剩余长度」是因为 NUL 天然就是边界：合法续字节都在 0x80–0xBF，
 * 0x00 一定进不了区间，因此「撞到 NUL」与「续字节非法」走同一条 break。
 * 这样也就不需要 strlen —— profile_allowed 白名单不含它，而手写的长度
 * 循环又会被 clang idiomize 回 strlen。 */
static int utf8_decode(const unsigned char* cursor, size_t* consumed) {
  unsigned char lo2;
  unsigned char hi2;
  const size_t need = utf8_lead_length(cursor[0], &lo2, &hi2);
  if (need == 0) {
    *consumed = 1;
    return 0;
  }
  size_t matched = 1;
  while (matched < need) {
    const unsigned char byte = cursor[matched];
    const unsigned char lo = matched == 1 ? lo2 : 0x80;
    const unsigned char hi = matched == 1 ? hi2 : 0xBF;
    if (byte < lo || byte > hi) {
      break;
    }
    ++matched;
  }
  if (matched == need) {
    *consumed = need;
    return 1;
  }
  *consumed = matched;
  return 0;
}

/* U+FFFD 的 JSON 转义。用 ASCII 转义而非 EF BF BD 原始字节：输出保持
 * 纯 ASCII 可见，且与「输入本来就含 U+FFFD」可区分。 */
static void write_replacement(FILE* file) {
  fputs("\\ufffd", file);
}

static void write_escaped_string(FILE* file, const char* value) {
  const unsigned char* cursor =
    (const unsigned char*)(value != NULL ? value : "");
  fputc('"', file);
  while (*cursor != '\0') {
    const unsigned char byte = *cursor;
    switch (byte) {
      case '"':
        fputs("\\\"", file);
        ++cursor;
        continue;
      case '\\':
        fputs("\\\\", file);
        ++cursor;
        continue;
      case '\b':
        fputs("\\b", file);
        ++cursor;
        continue;
      case '\f':
        fputs("\\f", file);
        ++cursor;
        continue;
      case '\n':
        fputs("\\n", file);
        ++cursor;
        continue;
      case '\r':
        fputs("\\r", file);
        ++cursor;
        continue;
      case '\t':
        fputs("\\t", file);
        ++cursor;
        continue;
      default:
        break;
    }
    if (byte < 0x20) {
      fprintf(file, "\\u%04x", (unsigned)byte);
      ++cursor;
      continue;
    }
    if (byte < 0x80) {
      fputc((int)byte, file);
      ++cursor;
      continue;
    }
    size_t consumed = 0;
    if (utf8_decode(cursor, &consumed)) {
      for (size_t index = 0; index < consumed; ++index) {
        fputc((int)cursor[index], file);
      }
    } else {
      write_replacement(file);
    }
    cursor += consumed;
  }
  fputc('"', file);
}

void pj_key(FILE* file, const char* key) {
  write_escaped_string(file, key);
  fputs(": ", file);
}

void pj_string(FILE* file, const char* value) {
  write_escaped_string(file, value);
}

void pj_u64(FILE* file, uint64_t value) {
  fprintf(file, "%llu", (unsigned long long)value);
}

void pj_i64(FILE* file, int64_t value) {
  fprintf(file, "%lld", (long long)value);
}

void pj_bool(FILE* file, int value) {
  fputs(value ? "true" : "false", file);
}

void pj_null(FILE* file) {
  fputs("null", file);
}

void pj_f64(FILE* file, double value) {
  fprintf(file, "%.9f", value);
}
