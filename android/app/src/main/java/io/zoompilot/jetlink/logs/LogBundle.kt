package io.zoompilot.jetlink.logs

import android.app.ActivityManager
import android.app.ApplicationExitInfo
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

    /** How many of the app's last exits the bundle lists. */
    private const val EXITS = 5

    /** At most this many of Jetlink's lines from processes before this one. */
    private const val EARLIER_LINES = 400

    /**
     * Why the app's last processes ended, newest first: Android keeps this
     * after the process is gone, so a build the phone killed for memory
     * (LOW_MEMORY) or a crash in QNN's native code (CRASH_NATIVE) shows in
     * the next bundle.
     */
    fun exits(context: Context): String = runCatching {
        val am = context.getSystemService(ActivityManager::class.java)
        val stamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
        am.getHistoricalProcessExitReasons(context.packageName, 0, EXITS).joinToString("") { e ->
            "${stamp.format(Date(e.timestamp))} pid ${e.pid} ${reason(e.reason)} status ${e.status}" +
                " importance ${e.importance} pss ${e.pss / 1024} MB rss ${e.rss / 1024} MB" +
                (e.description?.let { ": $it" } ?: "") + "\n"
        }.ifEmpty { "(none recorded)\n" }
    }.getOrElse { "(unavailable: ${it.message})\n" }

    private fun reason(r: Int): String = when (r) {
        ApplicationExitInfo.REASON_ANR -> "ANR"
        ApplicationExitInfo.REASON_CRASH -> "CRASH"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "CRASH_NATIVE"
        ApplicationExitInfo.REASON_DEPENDENCY_DIED -> "DEPENDENCY_DIED"
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "EXCESSIVE_RESOURCE_USAGE"
        ApplicationExitInfo.REASON_EXIT_SELF -> "EXIT_SELF"
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "INITIALIZATION_FAILURE"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "LOW_MEMORY"
        ApplicationExitInfo.REASON_OTHER -> "OTHER"
        ApplicationExitInfo.REASON_PACKAGE_STATE_CHANGE -> "PACKAGE_STATE_CHANGE"
        ApplicationExitInfo.REASON_PACKAGE_UPDATED -> "PACKAGE_UPDATED"
        ApplicationExitInfo.REASON_PERMISSION_CHANGE -> "PERMISSION_CHANGE"
        ApplicationExitInfo.REASON_SIGNALED -> "SIGNALED"
        ApplicationExitInfo.REASON_USER_REQUESTED -> "USER_REQUESTED"
        ApplicationExitInfo.REASON_USER_STOPPED -> "USER_STOPPED"
        else -> "reason $r"
    }

    /**
     * Jetlink's lines and any native crash report from the app's earlier
     * processes, which Android's log still holds: an app reads its own
     * lines of any of its processes. A build that died mid-compile left its
     * last memory lines there.
     */
    fun earlierLogcat(): String = runCatching {
        val command = arrayOf("logcat", "-d", "-v", "threadtime", "-s", "jetlink:*", "DEBUG:*", "libc:F")
        val process = ProcessBuilder(*command).redirectErrorStream(true).start()
        val text = process.inputStream.bufferedReader().use { it.readLines() }
        process.waitFor()
        val mine = Process.myPid().toString()
        // threadtime: date time pid tid level tag: message
        text.filter { line -> line.split(Regex("\\s+")).getOrNull(2).let { it != null && it != mine && it.all(Char::isDigit) } }
            .takeLast(EARLIER_LINES).joinToString("\n", postfix = "\n").ifBlank { "(none in the log)\n" }
    }.getOrElse { "(logcat unavailable: ${it.message})\n" }

    /**
     * The bundle as a file in the app's cache: header, Jetlink's log, the
     * app's last exits and earlier processes' lines, then the process log,
     * its oldest lines cut first to stay under [MAX_BYTES].
     */
    fun write(context: Context, lines: List<String>, processor: Processor): File {
        val head = header(processor)
        val jetlink = lines.joinToString("\n", postfix = "\n")
        val exits = exits(context)
        val earlier = earlierLogcat()
        var android = logcat()
        val room = MAX_BYTES - head.length - jetlink.length - exits.length - earlier.length - 512
        if (android.length > room) {
            android = "(older lines cut)\n" + android.takeLast(maxOf(room, 0))
        }
        val text = buildString {
            append(head)
            appendLine()
            appendLine("== Jetlink log (${lines.size} lines)")
            append(jetlink)
            appendLine()
            appendLine("== Last exits of the app, newest first")
            append(exits)
            appendLine()
            appendLine("== Jetlink lines of earlier processes")
            append(earlier)
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
