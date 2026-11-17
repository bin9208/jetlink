package io.zoompilot.jetlink.update

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.core.content.FileProvider
import io.zoompilot.jetlink.BuildConfig
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

/**
 * Jetlink Carrot's own updates, from its GitHub releases (bin9208/jetlink):
 * the main app follows the full releases (`v0.9.0-carrot.N`), the test app
 * the pre-releases tagged `-test`. Sideloaded, so no store does this.
 */
object AppUpdate {
    const val REPO = "bin9208/jetlink"
    private const val API = "https://api.github.com/repos/$REPO/releases?per_page=30"
    private const val PREFS = "app_update"
    private const val CHECKED_AT = "checked_at"
    private const val SKIPPED = "skipped_build"
    /** At most this often unless asked: GitHub allows 60 calls an hour unsigned. */
    private const val INTERVAL_MS = 6 * 3600 * 1000L

    /** "stable" or "test", as the build's flavor says. */
    val channel: String get() = BuildConfig.CHANNEL

    data class Release(
        val tag: String,
        val build: Int,
        val name: String,
        val notes: String,
        val apkUrl: String,
        val apkBytes: Long,
        val page: String,
    )

    /** The carrot build a tag names: 25 for `v0.9.0-carrot.25` and `v0.9.0-carrot.25-test`. */
    fun buildOf(tag: String): Int? = Regex("""carrot\.(\d+)""").find(tag)?.groupValues?.get(1)?.toIntOrNull()

    /** Whether a release belongs to this app's channel. */
    fun inChannel(tag: String, prerelease: Boolean, channel: String = this.channel): Boolean =
        if (channel == "test") prerelease && tag.endsWith("-test") else !prerelease && !tag.endsWith("-test")

    /** The newest release of the channel in the API's answer, if any is newer than `current`. */
    fun newest(json: String, current: Int, channel: String = this.channel): Release? {
        fun JsonObject.text(key: String) = this[key]?.jsonPrimitive?.contentOrNull.orEmpty()
        fun JsonObject.flag(key: String) = this[key]?.jsonPrimitive?.booleanOrNull ?: false
        var best: Release? = null
        for (element in Json.parseToJsonElement(json).jsonArray) {
            val r = element.jsonObject
            if (r.flag("draft")) continue
            val tag = r.text("tag_name")
            if (!inChannel(tag, r.flag("prerelease"), channel)) continue
            val build = buildOf(tag) ?: continue
            if (build <= current || (best != null && build <= best.build)) continue
            val apk = r["assets"]?.jsonArray?.map { it.jsonObject }?.firstOrNull { it.text("name").endsWith(".apk") } ?: continue
            val url = apk.text("browser_download_url")
            if (url.isEmpty()) continue
            best = Release(
                tag, build, r.text("name").ifEmpty { tag }, r.text("body"), url,
                apk["size"]?.jsonPrimitive?.longOrNull ?: 0L, r.text("html_url"),
            )
        }
        return best
    }

    /**
     * The newer release of this channel, or null. Network on the caller's
     * thread; `force` skips the interval and a release skipped before.
     */
    fun check(context: Context, force: Boolean = false): Release? {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val now = System.currentTimeMillis()
        if (!force && now - prefs.getLong(CHECKED_AT, 0) < INTERVAL_MS) return null
        val connection = (URL(API).openConnection() as HttpURLConnection).apply {
            connectTimeout = 10_000
            readTimeout = 15_000
            setRequestProperty("Accept", "application/vnd.github+json")
            setRequestProperty("User-Agent", "JetlinkCarrot")
        }
        val text = try {
            if (connection.responseCode !in 200..299) return null
            connection.inputStream.bufferedReader().use { it.readText() }
        } finally {
            connection.disconnect()
        }
        prefs.edit().putLong(CHECKED_AT, now).apply()
        val release = newest(text, BuildConfig.CARROT_BUILD) ?: return null
        if (!force && prefs.getInt(SKIPPED, 0) == release.build) return null
        return release
    }

    /** "Later": this release is not offered again on its own. */
    fun skip(context: Context, release: Release) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putInt(SKIPPED, release.build).apply()
    }

    /** Downloads the APK into the cache, reporting 0..1; returns the file. */
    fun download(context: Context, release: Release, progress: (Double) -> Unit): File {
        val dir = File(context.cacheDir, "updates").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() }
        val file = File(dir, "jetlink-carrot-${release.build}.apk")
        var url = URL(release.apkUrl)
        // GitHub answers a release asset with a redirect to its storage
        var connection: HttpURLConnection
        var hops = 0
        while (true) {
            connection = (url.openConnection() as HttpURLConnection).apply {
                connectTimeout = 15_000
                readTimeout = 60_000
                instanceFollowRedirects = false
                setRequestProperty("User-Agent", "JetlinkCarrot")
            }
            val code = connection.responseCode
            if (code in 300..399 && hops < 5) {
                url = URL(url, connection.getHeaderField("Location"))
                connection.disconnect()
                hops++
                continue
            }
            if (code !in 200..299) throw IllegalStateException("download answered HTTP $code")
            break
        }
        val total = connection.contentLengthLong.takeIf { it > 0 } ?: release.apkBytes
        connection.inputStream.use { input ->
            file.outputStream().use { output ->
                val buffer = ByteArray(1 shl 16)
                var done = 0L
                var last = 0L
                while (true) {
                    val n = input.read(buffer)
                    if (n < 0) break
                    output.write(buffer, 0, n)
                    done += n
                    if (total > 0 && done - last >= total / 100) {
                        progress(done.toDouble() / total)
                        last = done
                    }
                }
            }
        }
        connection.disconnect()
        progress(1.0)
        return file
    }

    /** Whether Android lets this app install APKs; if not, the setting to allow it. */
    fun canInstall(context: Context): Boolean = context.packageManager.canRequestPackageInstalls()

    fun allowInstallSettings(context: Context): Intent =
        Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:${context.packageName}"))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)

    /** Android's installer for the downloaded APK: an update over this app, same key. */
    fun installIntent(context: Context, apk: File): Intent {
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", apk)
        return Intent(Intent.ACTION_VIEW)
            .setDataAndType(uri, "application/vnd.android.package-archive")
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)
    }
}
