#!/usr/bin/env bash
# Builds LiteRT v2.2.0's MediaTek NPU dispatch library and compiler plugin
# for arm64 Android (Jetlink Carrot's MediaTek NPU support).
set -euo pipefail
B=/home/fma1/rust/litert-build
SDK=/home/fma1/Android/Sdk
export JAVA_HOME=/home/fma1/.local/opt/jdk-25 PATH=/home/fma1/.local/opt/jdk-25/bin:$PATH
echo "== sdk packages"
test -d $SDK/ndk/28.1.13356709 || { echo "NDK r28b missing"; exit 1; }
echo "== bazelisk"
if [ ! -x $B/tools/bazel ]; then
  curl -sSL -o $B/tools/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64
  chmod +x $B/tools/bazel
fi
export PATH=$B/tools:$PATH
cd $B/LiteRT
# Jetlink Carrot's change to the dispatch library: NeuroPilot's deadline and
# cache flush hint from the environment (dispatch-tuning.patch, beside this).
git apply --check "$(dirname "$0")/dispatch-tuning.patch" 2>/dev/null && git apply "$(dirname "$0")/dispatch-tuning.patch"
export ANDROID_NDK_HOME=$SDK/ndk/28.1.13356709 ANDROID_NDK_API_LEVEL=28 ANDROID_NDK_VERSION=28
export ANDROID_SDK_HOME=$SDK ANDROID_API_LEVEL=34 ANDROID_SDK_API_LEVEL=34 ANDROID_BUILD_TOOLS_VERSION=34.0.0
export PYTHON_BIN_PATH=/usr/bin/python3 PYTHON_LIB_PATH=/usr/lib/python3/dist-packages
export TF_NEED_CUDA=0 TF_NEED_ROCM=0 TF_DOWNLOAD_CLANG=0 TF_SET_ANDROID_WORKSPACE=1 TF_CONFIGURE_IOS=0
export CLANG_COMPILER_PATH=/home/fma1/.local/bin/clang CC_OPT_FLAGS=-Wno-sign-compare
echo "== configure"
python3 configure.py < /dev/null > $B/configure.log 2>&1 || { tail -30 $B/configure.log; exit 1; }
echo "== build"
bazel build -c opt --config=android_arm64 --jobs=12 \
  //litert/vendors/mediatek/dispatch:dispatch_api_so \
  //litert/vendors/mediatek/compiler:compiler_plugin_so 2>&1 | tail -60
ls -la bazel-bin/litert/vendors/mediatek/dispatch/*.so bazel-bin/litert/vendors/mediatek/compiler/*.so
