# LiteRT's MediaTek NPU libraries

LiteRT ships its NPU runtime libraries for Google Tensor and Qualcomm only
(`litert_npu_runtime_libraries_jit.zip`). For a MediaTek Dimensity's APU
(NeuroPilot v8 and later: the Dimensity 9000 to 9400, MT6895 to MT6993, and
the MT8188/8189/8196 tablet parts, as LiteRT's
`litert/vendors/mediatek/supported_soc.csv` lists them) these two are built
from the source of the same LiteRT release the app uses (v2.2.0):

| File | Bazel target |
| --- | --- |
| `libLiteRtDispatch_MediaTek.so` | `//litert/vendors/mediatek/dispatch:dispatch_api_so` |
| `libLiteRtCompilerPlugin_MediaTek.so` | `//litert/vendors/mediatek/compiler:compiler_plugin_so` |

`build.sh` is how they were built: LiteRT v2.2.0 (`145c7523`), Bazel through
Bazelisk, Android NDK r28b (API 28) and SDK 34 as LiteRT's own
`docker_build/hermetic_build.Dockerfile`, `--config=android_arm64 -c opt`.
NeuroPilot's headers come from MediaTek's public download that LiteRT's
`third_party/neuro_pilot/workspace.bzl` names.

At run time both reach the APU through the phone's NeuroPilot adapter
(`libneuronusdk_adapter.mtk.so` and the other names in the app's manifest).
Whether a phone's vendor partition offers it to apps is the phone maker's
choice; where it does not, the NPU profile says so and the GPU runs the model.

The dispatch library carries one change of Jetlink Carrot's,
`dispatch-tuning.patch`: when `JETLINK_NEURON_DEADLINE` (ms) or
`JETLINK_NEURON_CACHE_FLUSH` (NeuroPilot's flush hint bits) is set, each
execution gets `NeuronExecution_setDeadline` or `setCacheFlushHint`, for the
NPU sweep to time what they do. Unset, it behaves as LiteRT's.

`SHA256SUMS` pins the copies here; `app/build.gradle.kts` checks them.
