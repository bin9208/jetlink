package io.zoompilot.jetlink.logs

import android.content.Context
import android.os.Build
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID
import org.json.JSONObject

/**
 * Sends a log bundle to the openpilot-rust port's own upload receiver
 * (tools/carrot_upload_server), as the comma sends its diagnostics: a
 * session for purpose "tmux", then the file as `files[0]`. The receiver
 * keeps it under `openpilot/jetlink/<phone> <id>/` on the NAS.
 */
object LogUpload {
    /** The port's receiver, as openpilot-rust's `DEFAULT_WEB_UPLOAD_URL`. */
    const val DEFAULT_URL = "https://lunabox.myqnapcloud.com:18443"

    private const val PREFS = "log_upload"
    private const val INSTALL_ID = "install_id"
    private const val TIMEOUT_MS = 15_000
    private const val READ_TIMEOUT_MS = 60_000

    /** Where this phone's uploads land: the same folder every time. */
    fun installId(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        prefs.getString(INSTALL_ID, null)?.let { return it }
        val id = "jetlink-" + UUID.randomUUID().toString().replace("-", "").take(16)
        prefs.edit().putString(INSTALL_ID, id).apply()
        return id
    }

    /** The receiver's folder name part: letters, digits, `_ . -` and spaces. */
    private fun component(text: String): String =
        text.replace(Regex("[^A-Za-z0-9_. -]+"), "_").trim(' ', '.').take(96).ifEmpty { "phone" }

    private fun open(url: String, method: String): HttpURLConnection =
        (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            doOutput = true
            useCaches = false
        }

    private fun HttpURLConnection.answer(): String {
        val stream = if (responseCode in 200..299) inputStream else errorStream
        val text = stream?.bufferedReader()?.use { it.readText() }.orEmpty()
        if (responseCode !in 200..299) {
            throw IllegalStateException("upload server answered HTTP $responseCode: ${text.take(200)}")
        }
        return text
    }

    /**
     * Uploads `file`; returns where it went, for the person to quote. Runs
     * the network on the caller's thread: call it off the main one.
     */
    fun upload(context: Context, file: File, base: String = DEFAULT_URL): String {
        val root = base.trimEnd('/')
        require(root.startsWith("https://")) { "the upload server must be https" }
        val device = installId(context)
        val name = component("${Build.MANUFACTURER} ${Build.MODEL}")
        val session = open("$root/api/v1/session", "POST").run {
            setRequestProperty("Content-Type", "application/json")
            val body = JSONObject()
                .put("deviceId", device)
                .put("purpose", "tmux")
                .put("carName", name)
                .put("branch", "jetlink")
                .put("tmux_why", "jetlink")
            outputStream.use { it.write(body.toString().toByteArray()) }
            JSONObject(answer()).also { disconnect() }
        }
        val token = session.optString("token").ifEmpty { throw IllegalStateException("the upload server gave no session") }
        val boundary = "jetlink" + UUID.randomUUID().toString().replace("-", "")
        val head = ("--$boundary\r\n" +
            "Content-Disposition: form-data; name=\"files[0]\"; filename=\"tmux.log\"\r\n" +
            "Content-Type: text/plain\r\n\r\n").toByteArray()
        val tail = "\r\n--$boundary--\r\n".toByteArray()
        open("$root/api/v1/tmux/upload", "POST").run {
            setRequestProperty("Authorization", "Bearer $token")
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
            setFixedLengthStreamingMode(head.size + file.length() + tail.size)
            outputStream.use { out ->
                out.write(head)
                file.inputStream().use { it.copyTo(out) }
                out.write(tail)
            }
            answer()
            disconnect()
        }
        return "openpilot/jetlink/$name $device/"
    }
}
