#!/usr/bin/env bash
# Compiles a converted model with MediaTek's NeuroPilot ahead of time, to see
# on a PC whether a MediaTek APU takes it, and if not every reason at once:
# the phone's compiler says the same things, but only one stage at a time
# and only on the phone.
#
#   mediatek-compile-check.sh MODEL.tflite [SOC...]
#
# MODEL.tflite is the model.tflite of a LiteRT artifact; JetlinkKit's
# JETLINK_LITERT_PREPARE=<model.onnx> JETLINK_LITERT_OUT=<dir> test writes
# one from an ONNX (LiteRTFixtureTests). SOC is LiteRT's name for a
# Dimensity, MT6989 (9300) by default; MT6985 (9200) and MT6991 (9400) are
# others. Prints, per SoC, whether it compiled, every layer NeuroPilot
# refused and why, and the ops left off the APU.
#
# The compiler is x86-64 Linux only. Its Python environment is kept in
# $JETLINK_MEDIATEK_ENV (~/.cache/jetlink-mediatek by default) between runs.
set -euo pipefail

MODEL=${1:?usage: mediatek-compile-check.sh MODEL.tflite [SOC...]}
shift
SOCS=("$@")
[[ ${#SOCS[@]} -gt 0 ]] || SOCS=(MT6989)
[[ $(uname -sm) == "Linux x86_64" ]] || { echo "MediaTek's compiler runs on x86-64 Linux only" >&2; exit 1; }

# The release the app runs, and the newest SDK package (NeuroPilot 8 and 9).
LITERT=2.2.0
SDK_PACKAGE=2.3.0
ENV_DIR=${JETLINK_MEDIATEK_ENV:-$HOME/.cache/jetlink-mediatek}
if [[ ! -f "$ENV_DIR/installed" ]]; then
  echo "installing the compiler into $ENV_DIR (once)"
  packages=(ai-edge-litert==$LITERT ai-edge-litert-sdk-mediatek==$SDK_PACKAGE tflite)
  if command -v uv >/dev/null; then
    uv venv -q --clear "$ENV_DIR"
    VIRTUAL_ENV="$ENV_DIR" uv pip install -q "${packages[@]}"
  else
    python3 -m venv --clear "$ENV_DIR"
    "$ENV_DIR/bin/python" -m pip install -q "${packages[@]}"
  fi
  touch "$ENV_DIR/installed"
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
for soc in "${SOCS[@]}"; do
  echo "== $soc"
  mkdir -p "$WORK/$soc"
  (cd "$WORK/$soc" && TMPDIR="$WORK/$soc" "$ENV_DIR/bin/python" -I - "$MODEL" "$soc" "$WORK/$soc") <<'PYTHON'
import collections, glob, re, sys
import tflite
from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.vendors.mediatek import target

model, soc, out = sys.argv[1:]
result = aot_compile.aot_compile(model, output_dir=out, target=target.Target(target.SocModel(soc)), keep_going=True)
for _, compiled in result.models_with_backend:
    buf = open(compiled.path, "rb").read()
    m = tflite.Model.GetRootAsModel(buf, 0)
    names = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith("_")}
    codes = []
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        codes.append(oc.CustomCode().decode() if oc.CustomCode() else names.get(code, str(code)))
    g = m.Subgraphs(0)
    ops = collections.Counter(codes[g.Operators(i).OpcodeIndex()] for i in range(g.OperatorsLength()))
    apu = ops.pop("DISPATCH_OP", 0)
    print(f"compiled: {apu} APU partition(s); off the APU: {dict(ops) or 'nothing'}")
for _, error in result.failed_backends:
    print(f"FAILED: {error}")
# NeuroPilot's word on every layer it refused, from the plugin's log.
reasons = collections.Counter()
for log in glob.glob(f"{out}/*.error"):
    layer = None
    for line in open(log, errors="replace"):
        if m := re.search(r"NIR\[\d+\]: (\w+)<\d+>\{(\w+)\}", line):
            layer = m.group(2)
        elif layer and (m := re.search(r"├ (\w+): (.*)", line)):
            reasons[(layer, m.group(1), m.group(2).strip())] += 1
        elif "NoExecPlan" in line or "failed with error" in line:
            reasons[("compile", "", line.strip())] += 1
for (layer, unit, why), n in reasons.most_common():
    print(f"  {n:4d}  {layer:24s} {unit:5s} {why}")
PYTHON
done
