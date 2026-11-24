package io.zoompilot.jetlink.settings

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** The server's backends, as its start config names them. */
enum class Backend(val id: String) {
    Ort("ort"),
    LiteRt("litert"),
}

/**
 * Where the model runs: a backend of the server's and its device, an
 * OrtProfile or a LiteRtProfile, or Automatic, which picks one of those for
 * the phone. The device is also what Settings stores, and no two choices
 * share one.
 */
enum class Processor(val backend: Backend, val device: String, val title: String) {
    /** The phone's NPU where Jetlink has one for it, else the GPU ([automatic]). */
    Auto(Backend.LiteRt, "auto", "Automatic"),

    /**
     * The whole model on the NPU, prepared as the iPhone's. QNN, a Snapdragon's: the fastest
     * choice measured on an 8+ Gen 1, several times the GPU's speed.
     */
    Npu(Backend.Ort, "htp-whole", "NPU"),

    /**
     * A Google Tensor's NPU or a MediaTek Dimensity's APU, which LiteRT
     * compiles the model for on the phone through the vendor's compiler; the
     * GPU runs a model that compiler cannot take.
     */
    TensorNpu(Backend.LiteRt, "npu", "NPU"),

    /** The whole model on the GPU through LiteRT, which drives any phone's: Adreno, Mali, PowerVR. */
    Gpu(Backend.LiteRt, "gpu", "GPU"),

    /**
     * The vision trunk on the NPU, the rest on the GPU: the Mac's split. QNN, a Snapdragon's.
     * Listed after the GPU: the slowest of the three measured on an 8+ Gen 1.
     */
    NpuGpu(Backend.Ort, "htp", "NPU + GPU"),

    /** The CPU: the emulator and tests. Seconds a frame with a real model. */
    Cpu(Backend.Ort, "cpu", "CPU");

    /** Runs on QNN, which needs a Snapdragon: onnxruntime anywhere but its CPU. */
    val usesQnn: Boolean get() = backend == Backend.Ort && device != Cpu.device

    /** The choice the server runs: Automatic's pick for this phone, or this one. */
    fun resolved(automatic: Processor = Chip.automatic): Processor = if (this == Auto) automatic else this

    /**
     * Whether the loaded model runs on the GPU though this choice is a Tensor
     * NPU: the NPU took no model. [accelerator] is the ready engine's.
     */
    fun fellBack(accelerator: String?, automatic: Processor = Chip.automatic): Boolean =
        resolved(automatic) == TensorNpu && hardwareOf(accelerator).let { it != null && it != "NPU" }

    /** What the Processor row says: the choice, and with Automatic what it runs on. */
    fun summary(accelerator: String?, automatic: Processor = Chip.automatic): String {
        val runsOn = if (fellBack(accelerator, automatic)) hardwareOf(accelerator)!! else resolved(automatic).title
        return when {
            this == Auto -> "Automatic ($runsOn)"
            fellBack(accelerator, automatic) -> "$title ($runsOn)"
            else -> title
        }
    }

    companion object {
        /**
         * A stored choice. Older apps stored the GPU as `litert-gpu`, which
         * reads as [Gpu].
         */
        fun of(device: String?): Processor? = if (device == "litert-gpu") Gpu else entries.firstOrNull { it.device == device }

        /** "NPU", "GPU" or "CPU": the hardware an engine's accelerator names, "NPU(Tensor G5)". */
        fun hardwareOf(accelerator: String?): String? = accelerator?.substringBefore('(')?.trim()?.takeIf(String::isNotEmpty)

        /**
         * What a phone can choose, Automatic first. The Snapdragon NPU
         * choices run QNN, which needs a Snapdragon: anywhere else it leaves
         * every op to one CPU thread, minutes a frame. A Google Tensor G3 or
         * later has its own NPU choice. The GPU is LiteRT's, on any phone. The
         * CPU is offered on a Snapdragon only once chosen; the emulator offers
         * everything, for testing.
         */
        fun choices(qualcomm: Boolean, tensorNpu: Boolean, emulator: Boolean, current: Processor): List<Processor> = when {
            emulator -> entries
            qualcomm -> entries.filter { it != TensorNpu && (it != Cpu || current == Cpu) }
            tensorNpu -> listOf(Auto, TensorNpu, Gpu, Cpu)
            else -> listOf(Auto, Gpu, Cpu)
        }

        /**
         * Automatic's pick: a Google Tensor's NPU, which runs on the GPU any
         * model it cannot compile; the CPU on the emulator; a Snapdragon with
         * a Hexagon NPU QNN runs fp16 on (v69 and later: the 8 Gen 1 on) its
         * whole-graph NPU; else the GPU. QNN's NPU runs carrot's N16 and
         * Light Speed 2 right on an 8 Gen 2 and an 8 Elite now that the
         * vision norms are prescaled; LiteRT's GPU on an 8 Gen 2 gave Light
         * Speed 2 non-finite outputs on 27 of 34 frames, at 68 ms. A
         * Snapdragon whose generation this app does not know is taken for a
         * new one when it is an 8-series (SM8...).
         */
        fun automatic(tensorNpu: Boolean, emulator: Boolean, hexagon: Int? = null, qualcommModel: String? = null): Processor = when {
            emulator -> Cpu
            tensorNpu -> TensorNpu
            hexagon != null -> if (hexagon >= 69) Npu else Gpu
            qualcommModel != null && qualcommModel.uppercase().startsWith("SM8") -> Npu
            else -> Gpu
        }

    }
}

/** The few things worth changing on a phone, kept in SharedPreferences. */
data class SettingsValues(
    /** Where bench tools such as `bench_link.py --host` reach the phone, with [developer] on. */
    val port: Int = 5599,
    val processor: Processor = Processor.Auto,
    /** The NPU held in burst mode between frames rather than let it settle. */
    val keepNpuAwake: Boolean = true,
    /** A CPU core kept busy between frames. */
    val keepCpuAwake: Boolean = false,
    /** The screen stays on while Jetlink is on screen. */
    val keepScreenOn: Boolean = true,
    /**
     * Seven taps on Version turn it on: the Developer section, and the server
     * listening on [port] for bench tools. The comma needs neither. The
     * emulator, with no USB host, starts with it on.
     */
    val developer: Boolean = false,
    /**
     * The test app's experiments for the Snapdragon NPU (JetlinkONNX's
     * `CoreMLPreparation.Experiment`), what the driving model is built with;
     * each benchmark also compares every other combination.
     */
    val experiments: Set<String> = emptySet(),
    /** A TURZX panel on the phone's USB, through a hub beside the comma. Never restarts the server. */
    val hud: HudSettings = HudSettings(),
)

/** The external HUD (TURZX) on the phone's USB: the comma's ClusterHud settings, here. */
data class HudSettings(
    /**
     * Drive a TURZX panel when one is plugged in. A trial feature, off until
     * the owner turns it on; it stays a switch once it is settled.
     */
    val enabled: Boolean = false,
    /** 0 to 100, ClusterHudBrightness's fixed values. */
    val brightness: Int = 80,
    /** ClusterHudOrientation 2: the panel turned 180 degrees. */
    val rotate180: Boolean = false,
    /** ClusterHudMirror 1: left and right swapped, for a reflection in the windshield. */
    val mirror: Boolean = false,
)

/** The test app's NPU experiments: id, title, what it does. */
val NPU_EXPERIMENTS = listOf<Triple<String, String, String>>(
    // carrot.52-test: the precision candidates for N16 on an 8 Elite. On a Fold 7
    // over road frames norm-guard-224 took N16 from 0.75 m to 0.45 m; the
    // one-Mul guards (carrot.50) only to 0.69 m and leave the list.
    Triple(
        "norm-guard-224", "정책부 norm 가드 224 (전체)",
        "정책부 norm 76개 모두에 가드. 도로 프레임 0.75→0.45m, 무작위 영상은 0.39→0.55m.",
    ),
    Triple(
        "norm-guard-res-224", "가드 224 (residual norm만)",
        "Add에서 오는 넓은 norm에만 가드를 겁니다. 넘칠 일 없는 q/k norm은 그대로 둡니다.",
    ),
    Triple(
        "norm-guard-hot-224", "가드 224 (넘치는 norm 18개만)",
        "실제 도로에서 값이 커지는 N16 norm 18개에만 가드를 겁니다.",
    ),
    Triple(
        "norm-guard-hot-192", "N16 norm 가드 192 (넘치는 norm 18개만)",
        "실제 도로에서 fp16을 넘는 N16 norm 18개의 큰 행만 192로 줄여 커브 편향을 줄입니다. " +
            "8 Elite 정식 앱 기본값. N16에만 적용됩니다.",
    ),
    Triple(
        "norm-guard-hot-240", "가드 240 (넘치는 norm 18개만)",
        "위와 같고, fp16 한도 바로 아래인 240을 넘는 행을 줄입니다.",
    ),
    // int8 weights (carrot.41-test): N16 583 ms a run on a Galaxy Tab S9, QNN
    // leaving the widened MatMuls to the CPU, and further from the CPU's. Dropped.
    Triple(
        "no-vision-prescale", "No Vision Prescale",
        "Vision norms without their 1/32 (an 8 Elite's fix; an 8 Gen 2 ran right without it).",
    ),
    Triple(
        "fold-vision-prescale", "Fold Vision Prescale",
        "The 1/32 in the convolution before a vision norm instead of a separate multiply.",
    ),
)

/**
 * N16's norm guards, an 8 Elite's (HTP v79) alone: on a Fold 7 the guard on the
 * 18 hot norms took N16 from 0.75 m to 0.35 m from the CPU over road frames,
 * but on a Tab S9 Ultra (8 Gen 2, v73) every guard made it far worse (0.39 m
 * to 2.5 m, the residual one 5.7 m), so no other generation may drive with one.
 */
const val NORM_GUARD_HEXAGON = 79

/**
 * The main app's one experiment (carrot.54): N16's guard on its 18 hot norms
 * at 192, on a Fold 7 over road frames 0.75 m to 0.35 m from the CPU at the
 * plan's end for about 1 ms, and driven there with less lean in curves. On by
 * default on an 8 Elite ([migratedExperiments]); the switch turns it off.
 */
const val MAIN_APP_EXPERIMENT = "norm-guard-hot-192"

/** The experiments this app offers: the main app [MAIN_APP_EXPERIMENT] alone; the norm guards on an 8 Elite only. */
fun offeredExperiments(testApp: Boolean, hexagon: Int? = Chip.hexagon): List<Triple<String, String, String>> =
    NPU_EXPERIMENTS.filter { if (testApp) true else it.first == MAIN_APP_EXPERIMENT }
        .filter { !it.first.startsWith("norm-guard") || hexagon == NORM_GUARD_HEXAGON }

class Settings(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences("settings", Context.MODE_PRIVATE).also(::migrate)
    private val state = MutableStateFlow(read())
    val values: StateFlow<SettingsValues> = state.asStateFlow()

    fun update(change: (SettingsValues) -> SettingsValues) {
        val next = change(state.value)
        prefs.edit()
            .putInt(PORT, next.port)
            .putString(PROCESSOR, next.processor.device)
            .putBoolean(KEEP_NPU_AWAKE, next.keepNpuAwake)
            .putBoolean(KEEP_CPU_AWAKE, next.keepCpuAwake)
            .putBoolean(KEEP_SCREEN_ON, next.keepScreenOn)
            .putBoolean(DEVELOPER, next.developer)
            .putStringSet(EXPERIMENTS, next.experiments)
            .putBoolean(HUD_ENABLED, next.hud.enabled)
            .putInt(HUD_BRIGHTNESS, next.hud.brightness)
            .putBoolean(HUD_ROTATE_180, next.hud.rotate180)
            .putBoolean(HUD_MIRROR, next.hud.mirror)
            .apply()
        state.value = next
    }

    private fun read(): SettingsValues {
        val defaults = SettingsValues()
        val port = prefs.getInt(PORT, defaults.port)
        return SettingsValues(
            port = if (port in 1..65535) port else defaults.port,
            // a stored QNN choice on a phone without a Snapdragon falls back to the default
            processor = Processor.of(prefs.getString(PROCESSOR, null))
                ?.takeIf { it in Chip.processors(it) }
                ?: defaults.processor,
            keepNpuAwake = prefs.getBoolean(KEEP_NPU_AWAKE, defaults.keepNpuAwake),
            keepCpuAwake = prefs.getBoolean(KEEP_CPU_AWAKE, defaults.keepCpuAwake),
            keepScreenOn = prefs.getBoolean(KEEP_SCREEN_ON, defaults.keepScreenOn),
            developer = prefs.getBoolean(DEVELOPER, Chip.isEmulator),
            // an experiment dropped since (fold, carrot.41) is forgotten
            experiments = prefs.getStringSet(EXPERIMENTS, null)?.toSet()?.filter { id -> NPU_EXPERIMENTS.any { it.first == id } }?.toSet()
                ?: defaults.experiments,
            hud = HudSettings(
                enabled = prefs.getBoolean(HUD_ENABLED, defaults.hud.enabled),
                brightness = prefs.getInt(HUD_BRIGHTNESS, defaults.hud.brightness).coerceIn(0, 100),
                rotate180 = prefs.getBoolean(HUD_ROTATE_180, defaults.hud.rotate180),
                mirror = prefs.getBoolean(HUD_MIRROR, defaults.hud.mirror),
            ),
        )
    }

    companion object {
        private const val VERSION = "version"
        private const val PORT = "port"
        private const val PROCESSOR = "processor"
        private const val KEEP_NPU_AWAKE = "keepNpuAwake"
        private const val KEEP_CPU_AWAKE = "keepCpuAwake"
        private const val KEEP_SCREEN_ON = "keepScreenOn"
        private const val DEVELOPER = "developer"
        private const val EXPERIMENTS = "npuExperiments"
        private const val HUD_ENABLED = "hudEnabled"
        private const val HUD_BRIGHTNESS = "hudBrightness"
        private const val HUD_ROTATE_180 = "hudRotate180"
        private const val HUD_MIRROR = "hudMirror"

        /** Settings from before Automatic are version 1; from before the main app's N16 guard (carrot.54), 2. */
        private const val CURRENT = 3

        /** Brings settings stored by an older app up to [CURRENT], once. */
        private fun migrate(prefs: SharedPreferences) {
            val from = prefs.getInt(VERSION, 1)
            if (from >= CURRENT) return
            val edit = prefs.edit().putInt(VERSION, CURRENT)
            if (from < 2) migratedProcessor(prefs.getString(PROCESSOR, null))?.let { edit.putString(PROCESSOR, it) }
            migratedExperiments(
                prefs.getStringSet(EXPERIMENTS, null), io.zoompilot.jetlink.BuildConfig.CHANNEL == "stable", Chip.hexagon,
            )?.let { edit.putStringSet(EXPERIMENTS, it) }
            edit.apply()
        }

        /**
         * carrot.54: the main app on an 8 Elite turns [MAIN_APP_EXPERIMENT] on,
         * once, beside what is stored; turned off later it stays off. Null
         * leaves what is stored (the test app, any other chip).
         */
        fun migratedExperiments(stored: Set<String>?, stable: Boolean, hexagon: Int?): Set<String>? =
            if (stable && hexagon == NORM_GUARD_HEXAGON) (stored ?: emptySet()) + MAIN_APP_EXPERIMENT else null

        /**
         * A processor stored before Automatic existed: `gpu` was the default
         * then, so it becomes Automatic. Null leaves what is stored.
         */
        fun migratedProcessor(stored: String?): String? = if (Processor.of(stored) == Processor.Gpu) Processor.Auto.device else null
    }
}
