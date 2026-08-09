# AFL++ with Python support, plus an LLVM toolchain for the coverage build.
#
# Everything is pinned: AFL++ by commit, the base image by digest-able tag, and
# each target by upstream commit in its own build.sh. A fuzzing comparison whose
# toolchain drifts between arms is not a comparison, and one whose toolchain
# drifts between the pilot and the final run is not reproducible.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LLVM_VERSION=14 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git wget curl ca-certificates \
      clang-14 llvm-14 llvm-14-dev lld-14 \
      libc++-14-dev libc++abi-14-dev \
      python3 python3-dev python3-pip \
      automake autoconf libtool cmake pkg-config bison flex \
      zlib1g-dev libssl-dev \
      gnuplot-nox \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100 \
 && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-14 100 \
 && update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-14 100 \
 && update-alternatives --install /usr/bin/llvm-profdata llvm-profdata /usr/bin/llvm-profdata-14 100 \
 && update-alternatives --install /usr/bin/llvm-cov llvm-cov /usr/bin/llvm-cov-14 100 \
 && update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-14 100

# Pin AFL++. Bump deliberately and re-run the whole matrix; never mid-experiment.
ARG AFLPP_COMMIT=v4.21c
RUN git clone https://github.com/AFLplusplus/AFLplusplus /afl \
 && cd /afl \
 && git checkout ${AFLPP_COMMIT} \
 # PYTHON_INCLUDE is what enables AFL_PYTHON_MODULE. Without it the custom
 # mutator hooks silently do not load and every arm degrades to stock AFL++ --
 # a failure that produces plausible-looking but meaningless results, so the
 # build asserts on it below.
 && PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
    LLVM_CONFIG=llvm-config-14 \
    make -j"$(nproc)" distrib \
 && make install \
 && cd / && rm -rf /afl/.git

# Fail the build, not the experiment, if Python support did not compile in.
RUN afl-fuzz -h 2>&1 | grep -qi "python" \
    || (echo "FATAL: AFL++ built without Python support; AFL_PYTHON_MODULE would be ignored" && exit 1)

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /work
ENV PYTHONPATH=/work \
    AFL_SKIP_CPUFREQ=1 \
    AFL_NO_AFFINITY=1 \
    AFL_MAP_SIZE=262144

CMD ["/bin/bash"]
