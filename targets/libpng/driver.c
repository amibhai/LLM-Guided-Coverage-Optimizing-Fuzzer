/* Minimal AFL++ driver around a libFuzzer-style entry point.
 *
 * OSS-Fuzz harnesses expose LLVMFuzzerTestOneInput(data, size). AFL++ ships
 * afl-clang-fast's own libFuzzer driver, but linking it pulls in extra
 * machinery we do not need and cannot easily hold constant across our three
 * build variants. This driver is ~40 lines and does exactly one thing, so the
 * three variants differ only in instrumentation flags -- which is what makes
 * their coverage and crash results comparable.
 *
 * Uses AFL++'s persistent mode when available (__AFL_LOOP), which is worth
 * roughly an order of magnitude in executions/second on a target this small.
 * Since exec/s is one of the metrics under study, every arm must get it --
 * and every arm does, because they all run this same binary.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

#ifdef __AFL_FUZZ_TESTCASE_LEN
__AFL_FUZZ_INIT();
#endif

#define MAX_INPUT (1 << 20) /* 1 MiB; larger PNGs are not interesting here */

int main(int argc, char **argv) {
#ifdef __AFL_HAVE_MANUAL_CONTROL
  __AFL_INIT();
#endif

#ifdef __AFL_FUZZ_TESTCASE_LEN
  /* Persistent mode: AFL++ feeds inputs through shared memory. */
  unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;
  while (__AFL_LOOP(10000)) {
    int len = __AFL_FUZZ_TESTCASE_LEN;
    LLVMFuzzerTestOneInput(buf, (size_t)len);
  }
  return 0;
#else
  /* Fallback: one input from a file argument, or stdin. Used by afl-showmap
   * and by the llvm-cov replay pass, neither of which uses persistent mode. */
  static uint8_t data[MAX_INPUT];
  size_t n = 0;

  if (argc > 1) {
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    n = fread(data, 1, MAX_INPUT, f);
    fclose(f);
  } else {
    n = fread(data, 1, MAX_INPUT, stdin);
  }

  LLVMFuzzerTestOneInput(data, n);
  return 0;
#endif
}
