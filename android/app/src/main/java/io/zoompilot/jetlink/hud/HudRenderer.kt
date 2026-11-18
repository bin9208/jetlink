package io.zoompilot.jetlink.hud

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.BitmapFactory
import android.graphics.Paint
import android.graphics.Typeface
import io.zoompilot.jetlink.server.Snapshot
import java.io.ByteArrayOutputStream
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale
import kotlin.math.roundToInt

/**
 * Draws the HUD's picture for a TURZX panel and encodes it as the panel
 * takes it. Until the comma sends its driving data this is Jetlink's own
 * status: the time, the comma's link, the model and how fast it runs.
 *
 * Drawn landscape, [size] as the comma's renderer sees the panel, then
 * turned into the panel's portrait frame exactly as the comma's upload ends
 * up ([Portrait]). The panel's own 180-degree setting and the mirror cover a
 * panel mounted the other way.
 */
class HudRenderer(private val size: TurzxPacket.Size) {
    private val landscape = Bitmap.createBitmap(size.width, size.height, Bitmap.Config.ARGB_8888)
    private val portrait = Bitmap.createBitmap(size.height, size.width, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(landscape)
    private val jpeg = ByteArrayOutputStream(256 * 1024)
    private val landscapePixels = IntArray(size.width * size.height)
    private val portraitPixels = IntArray(size.width * size.height)

    private val h = size.height.toFloat()
    private val w = size.width.toFloat()
    private val big = text(h * 0.40f, Typeface.BOLD, Color.WHITE)
    private val medium = text(h * 0.15f, Typeface.BOLD, Color.WHITE)
    private val small = text(h * 0.095f, Typeface.NORMAL, GREY)
    private val dot = Paint(Paint.ANTI_ALIAS_FLAG)

    /** The frame for [snapshot] at [now], as JPEG bytes in the panel's orientation. */
    fun render(snapshot: Snapshot, now: LocalDateTime, mirror: Boolean, quality: Int = JPEG_QUALITY): ByteArray {
        draw(snapshot, now)
        landscape.getPixels(landscapePixels, 0, size.width, 0, 0, size.width, size.height)
        Portrait.rotate(landscapePixels, size.width, size.height, mirror, portraitPixels)
        portrait.setPixels(portraitPixels, 0, size.height, 0, 0, size.height, size.width)
        jpeg.reset()
        portrait.compress(Bitmap.CompressFormat.JPEG, quality, jpeg)
        return jpeg.toByteArray()
    }

    private fun draw(snapshot: Snapshot, now: LocalDateTime) {
        canvas.drawColor(BACKGROUND)
        val margin = h * 0.12f
        val baseline = h * 0.56f

        // the time, left
        big.textAlign = Paint.Align.LEFT
        canvas.drawText(now.format(CLOCK), margin, baseline, big)
        small.textAlign = Paint.Align.LEFT
        canvas.drawText(now.format(DATE), margin, baseline + h * 0.2f, small)

        // the comma's link and the model, middle
        val (label, color) = when {
            snapshot.connected -> "코마 연결됨" to GOOD
            snapshot.link.state == "disconnected" -> "코마 연결 끊김" to BAD
            else -> "코마 기다리는 중" to WAIT
        }
        val middle = w * 0.36f
        dot.color = color
        val radius = h * 0.045f
        canvas.drawCircle(middle + radius, baseline - h * 0.055f, radius, dot)
        medium.textAlign = Paint.Align.LEFT
        canvas.drawText(label, middle + radius * 3.2f, baseline, medium)
        val engine = snapshot.engine
        val model = snapshot.modelName(engine.sha256) ?: when (engine.state) {
            "building", "loading" -> "모델 준비 중"
            "failed" -> "모델 실패"
            else -> "모델 없음"
        }
        val runsOn = engine.accelerator ?: snapshot.server?.device
        canvas.drawText(listOfNotNull(model, runsOn).joinToString(" · "), middle, baseline + h * 0.2f, small)

        // how fast, right
        val recent = snapshot.recent?.takeIf { snapshot.connected && it.frames > 0 }
        val right = w - margin
        medium.textAlign = Paint.Align.RIGHT
        small.textAlign = Paint.Align.RIGHT
        val served = recent?.servedMs
        canvas.drawText(served?.let { "${it.mean.roundToInt()} ms" } ?: "– ms", right, baseline, medium)
        val detail = if (recent != null && served != null) {
            "p99 ${served.p99.roundToInt()} ms · ${"%.0f".format(Locale.US, recent.fps)} fps"
        } else {
            "추론 대기"
        }
        canvas.drawText(detail, right, baseline + h * 0.2f, small)

        // what this screen is, bottom
        small.textAlign = Paint.Align.CENTER
        small.color = DIM
        canvas.drawText("Jetlink HUD · 주행 화면은 코마 쪽 지원 후 표시됩니다", w / 2f, h - margin * 0.45f, small)
        small.color = GREY
    }

    private fun text(px: Float, style: Int, color: Int) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = px
        typeface = Typeface.create(Typeface.DEFAULT, style)
        this.color = color
    }

    companion object {
        /**
         * What a panel shows for [jpeg], a frame [render] made: decoded and
         * turned back to landscape, for the Settings preview without a panel.
         */
        fun preview(jpeg: ByteArray): Bitmap? {
            val frame = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size) ?: return null
            val height = frame.width
            val width = frame.height
            val portraitPixels = IntArray(width * height)
            frame.getPixels(portraitPixels, 0, height, 0, 0, height, width)
            val pixels = IntArray(width * height)
            Portrait.unrotate(portraitPixels, width, height, pixels)
            return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        }

        /** The comma's --usb-jpeg-quality default. */
        const val JPEG_QUALITY = 68

        private val BACKGROUND = Color.rgb(11, 13, 16)
        private val GREY = Color.rgb(170, 176, 186)
        private val DIM = Color.rgb(96, 102, 112)
        private val GOOD = Color.rgb(52, 199, 89)
        private val WAIT = Color.rgb(255, 179, 64)
        private val BAD = Color.rgb(255, 69, 58)
        private val CLOCK = DateTimeFormatter.ofPattern("HH:mm")
        private val DATE = DateTimeFormatter.ofPattern("M월 d일 E", Locale.KOREAN)
    }
}
