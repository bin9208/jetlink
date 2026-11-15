package io.zoompilot.jetlink.logs

import android.content.Context
import android.os.Build
import android.os.Process
import io.zoompilot.jetlink.BuildConfig
import io.zoompilot.jetlink.settings.Chip
import io.zoompilot.jetlink.settings.Processor
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * One text file to analyse a phone's problem with: the phone and the
 * choices that shape a build, Jetlink's own log, and everything this
 * process wrote to Android's log, where LiteRT, its vendor plugins
 * (MediaTek's NeuroPilot, Google Tensor) and QNN say why an NPU took no
 * model. An app may read its own lines of the log without any permission.
 */
object LogBundle {
    /** The server keeps uploads under 16 MB; the bundle stays well below it. */
    const val MAX_BYTES = 12 * 1024 * 1024

    /** At most this many of the process's newest log lines. */
    private const val LOGCAT_LINES = 40_000

    fun stamp(now: Date = Date()): String = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(now)

    fun header(processor: Processor): String = buildString {
        appendLine("== Jetlink ${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE}) ${stamp()}")
        appendLine("device: ${Build.MANUFACTURER} ${Build.MODEL} (${Build.DEVICE}), Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
        appendLine("soc: ${Chip.manufacturer} ${Chip.model} = ${Chip.name}")
        appendLine("npu: qualcomm=${Chip.isQualcomm} hexagon=${Chip.hexagon} tensor=${Chip.hasTensorNpu} mediatek=${Chip.hasMediaTekNpu}")
        appendLine("processor: ${processor.title} (${processor.device}), runs ${processor.resolved().title} (${processor.resolved().device})")
        appendLine("fingerprint: ${Build.FINGERPRINT}")
    }

    /** This process's lines of Android's log, newest last; empty when it cannot be read. */
    fun logcat(): String = runCatching {
        val command = arrayOf("logcat", "-d", "-v", "threadtime", "-t", LOGCAT_LINES.toString(), "--pid=${Process.myPid()}")
        val process = ProcessBuilder(*command).redirectErrorStream(true).start()
        val text = process.inputStream.bufferedReader().use { it.readText() }
        process.waitFor()
        text
    }.getOrElse { "(logcat unavailable: ${it.message})\n" }

    /**
     * The bundle as a file in the app's cache: header, Jetlink's log, then
     * the process log, its oldest lines cut first to stay under [MAX_BYTES].
     */
    fun write(context: Context, lines: List<String>, processor: Processor): File {
        val head = header(processor)
        val jetlink = lines.joinToString("\n", postfix = "\n")
        var android = logcat()
        val room = MAX_BYTES - head.length - jetlink.length - 256
        if (android.length > room) {
            android = "(older lines cut)\n" + android.takeLast(maxOf(room, 0))
        }
        val text = buildString {
            append(head)
            appendLine()
            appendLine("== Jetlink log (${lines.size} lines)")
            append(jetlink)
            appendLine()
            appendLine("== Android log of this process")
            append(android)
        }
        return File(context.cacheDir, "logs").apply { mkdirs() }.resolve("jetlink-${stamp()}.log").apply {
            parentFile?.listFiles()?.filter { it != this }?.forEach { it.delete() }
            writeText(text.takeLast(MAX_BYTES))
        }
    }
}
