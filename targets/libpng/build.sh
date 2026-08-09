#!/usr/bin/env bash
# Build libpng in three instrumented variants. Runs inside the docker image.
#
#   fuzz/   afl-clang-lto  -> the binary every arm actually fuzzes
#   cov/    llvm source coverage -> ground-truth coverage, replayed post-hoc
#   asan/   AddressSanitizer -> crash triage and stack-hash dedup
#
# Why three builds instead of one
# -------------------------------
# They measure different things and cannot be collapsed without corrupting the
# result. The fuzzing binary must be as fast as possible, because throughput is
# one of the variables under study -- adding ASAN to it would cost ~2x exec
# speed and change the very numbers we are comparing. The coverage binary is for
# replay only, so its speed is irrelevant and its accuracy is everything: AFL++
# edge counts are a hash-bucketed approximation, while llvm-cov gives real
# region/line/function coverage. FuzzBench separates these for the same reason.
#
# Why afl-clang-lto for the fuzzing build
# ---------------------------------------
# LTO mode assigns edge IDs at link time instead of by random hash, which is what
# makes AFL_LLVM_DOCUMENT_IDS able to emit a stable edge-id -> function map. That
# map is the entire grounding for strategy D: without it the planner has no
# honest way to name a code region, and strategy D degrades to strategy C by
# design. It also eliminates edge-ID collisions, which otherwise inflate
# measured coverage.
set -euo pipefail

TARGET_NAME=libpng
LIBPNG_REPO=https://github.com/glennrp/libpng.git
# Pinned commit -- a moving target makes trials incomparable across days.
LIBPNG_COMMIT=f5e92d76973a7255d31d0ed8c1c2b53868cf5e10   # v1.6.37

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/work/targets/${TARGET_NAME}/build}"
SRC="${SRC:-/tmp/src-${TARGET_NAME}}"
JOBS="$(nproc)"

log() { printf '\n=== %s ===\n' "$*"; }

fetch() {
  if [ ! -d "${SRC}/.git" ]; then
    log "fetching libpng @ ${LIBPNG_COMMIT}"
    git clone "${LIBPNG_REPO}" "${SRC}"
  fi
  git -C "${SRC}" fetch --all --tags --quiet || true
  git -C "${SRC}" checkout --quiet "${LIBPNG_COMMIT}"
}

# $1 = variant dir, $2 = CC, $3 = CXX, $4 = extra CFLAGS
build_variant() {
  local variant="$1" cc="$2" cxx="$3" extra="$4"
  local dst="${OUT}/${variant}"
  local work="${SRC}/../work-${TARGET_NAME}-${variant}"

  log "building ${TARGET_NAME} [${variant}]"
  rm -rf "${work}" "${dst}"
  mkdir -p "${dst}"
  cp -r "${SRC}" "${work}"
  cd "${work}"

  export CC="${cc}" CXX="${cxx}"
  export CFLAGS="-g -O2 -fno-omit-frame-pointer ${extra}"
  export CXXFLAGS="${CFLAGS}"

  # AFL_LLVM_DOCUMENT_IDS is honoured at compile time by afl-clang-lto only.
  if [ "${variant}" = "fuzz" ]; then
    export AFL_LLVM_DOCUMENT_IDS="${dst}/edge_ids.txt"
    export AFL_LLVM_INSTRUMENT=LTO
  fi

  ./autogen.sh --maintainer >/dev/null 2>&1 || autoreconf -fi
  ./configure --disable-shared --enable-static >/dev/null
  make -j"${JOBS}" >/dev/null

  # libpng ships an OSS-Fuzz harness (libpng_read_fuzzer.cc) reading a PNG from
  # a file argument. Using upstream's harness rather than writing our own keeps
  # results comparable with published libpng fuzzing numbers.
  local harness="${work}/contrib/oss-fuzz/libpng_read_fuzzer.cc"
  if [ ! -f "${harness}" ]; then
    echo "FATAL: expected harness at ${harness}" >&2
    exit 1
  fi

  "${cxx}" ${CXXFLAGS} -std=c++11 \
      -I"${work}" \
      "${harness}" \
      "${HERE}/driver.c" \
      "${work}/.libs/libpng16.a" \
      -lz -o "${dst}/${TARGET_NAME}_fuzzer"

  if [ "${variant}" = "fuzz" ] && [ -f "${dst}/edge_ids.txt" ]; then
    echo "edge->function map: $(wc -l < "${dst}/edge_ids.txt") entries"
  elif [ "${variant}" = "fuzz" ]; then
    # Not fatal: D detects the empty map and runs as C. But it silently removes
    # an experimental arm's whole mechanism, so make it loud here.
    echo "WARNING: no edge_ids.txt produced -- strategy D will run ungrounded (as C)" >&2
  fi

  cd - >/dev/null
  rm -rf "${work}"
}

fetch

build_variant fuzz  afl-clang-lto afl-clang-lto++ ""
build_variant cov   clang         clang++         "-fprofile-instr-generate -fcoverage-mapping"
build_variant asan  afl-clang-fast afl-clang-fast++ "-fsanitize=address"

# Seeds: a handful of small valid PNGs from libpng's own test corpus. Small and
# valid matters -- an empty or junk seed set makes every arm spend its first
# minutes rediscovering the file format, which compresses the differences the
# experiment is trying to measure.
log "collecting seeds"
mkdir -p "${OUT}/seeds"
find "${SRC}" -name '*.png' -size -32k | head -50 | while read -r png; do
  cp "${png}" "${OUT}/seeds/$(basename "${png}")"
done
[ -z "$(ls -A "${OUT}/seeds")" ] && printf '\x89PNG\r\n\x1a\n' > "${OUT}/seeds/minimal.png"

cp "${HERE}/png.dict" "${OUT}/png.dict" 2>/dev/null || true

log "done"
echo "  fuzz binary : ${OUT}/fuzz/${TARGET_NAME}_fuzzer"
echo "  cov binary  : ${OUT}/cov/${TARGET_NAME}_fuzzer"
echo "  asan binary : ${OUT}/asan/${TARGET_NAME}_fuzzer"
echo "  seeds       : $(ls "${OUT}/seeds" | wc -l) files"
echo "  edge map    : ${OUT}/fuzz/edge_ids.txt"
