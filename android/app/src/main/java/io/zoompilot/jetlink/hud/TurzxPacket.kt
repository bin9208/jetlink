// SPDX-License-Identifier: GPL-3.0-or-later
// Protocol derived from turing-smart-screen-python/library/lcd/lcd_comm_turing_usb.py,
// Copyright (C) 2021 Matthieu Houdebine (mathoudebine), as carrot's comma ports it
// (openpilot-rust rust/crates/cluster/src/usb/packet.rs).
package io.zoompilot.jetlink.hud

import java.time.Duration
import java.time.ZonedDateTime
import javax.crypto.Cipher
import javax.crypto.spec.IvParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * The TURZX USB display's packets, byte for byte as the comma sends them:
 * a 500-byte command, zero-padded to 504 and DES-CBC encrypted with the
 * vendor key (which is also the IV), then two trailer bytes; a frame is a
 * command whose bytes 8..11 carry the payload size, followed by the payload.
 */
object TurzxPacket {
    const val VENDOR_ID = 0x1cbe

    /** Command ids the comma uses. */
    const val INIT = 10
    const val ORIENTATION = 13
    const val BRIGHTNESS = 14
    const val JPEG = 101

    private val KEY = "slv3tuzx".toByteArray(Charsets.US_ASCII)

    /**
     * One 512-byte command. [milliseconds] is the time since local midnight;
     * each field writes the low byte of its value at its offset (< 500).
     */
    fun command(id: Int, milliseconds: Long, fields: List<Pair<Int, Int>> = emptyList()): ByteArray {
        require(milliseconds in 0..0xFFFF_FFFFL) { "command timestamp out of range" }
        val plain = ByteArray(504)
        plain[0] = id.toByte()
        plain[2] = 0x1a
        plain[3] = 0x6d
        for (i in 0 until 4) plain[4 + i] = (milliseconds ushr (8 * i)).toByte()
        for ((offset, value) in fields) {
            require(offset in 0 until 500) { "command field index out of range" }
            plain[offset] = (value and 0xff).toByte()
        }
        val cipher = Cipher.getInstance("DES/CBC/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, SecretKeySpec(KEY, "DES"), IvParameterSpec(KEY))
        val output = ByteArray(512)
        cipher.doFinal(plain, 0, plain.size, output, 0)
        output[510] = 161.toByte()
        output[511] = 26
        return output
    }

    /** A frame: its command, the payload's size big-endian at 8..11, then the payload. */
    fun frame(id: Int, milliseconds: Long, payload: ByteArray): ByteArray {
        val size = payload.size
        val fields = (0 until 4).map { 8 + it to (size ushr (24 - 8 * it)) }
        return command(id, milliseconds, fields) + payload
    }

    /** What the commands carry as their timestamp: milliseconds since local midnight. */
    fun millisecondsToday(now: ZonedDateTime = ZonedDateTime.now()): Long =
        Duration.between(now.toLocalDate().atStartOfDay(now.zone), now).toMillis()

    /** The panel's brightness byte for a percentage: int(percent / 100 * 102), as the comma does. */
    fun brightness(percent: Int): Int = (percent.coerceIn(0, 100) / 100.0 * 102.0).toInt()

    /** The panel's landscape size by USB product id, or null for one the vendor library does not know. */
    fun size(product: Int): Size? = when (product) {
        0x28 -> Size(480, 480)
        0x46 -> Size(960, 320)
        0x50 -> Size(1280, 720)
        0x80 -> Size(1280, 800)
        0x88 -> Size(1920, 480)
        0x92 -> Size(1920, 462)
        0x123 -> Size(1920, 720)
        else -> null
    }

    /** "TURZX 9.2", as the comma's autorun names them. */
    fun name(product: Int): String = when (product) {
        0x92 -> "TURZX 9.2\""
        0x123 -> "TURZX 12.3\""
        else -> "TURZX (pid %04x)".format(product)
    }

    data class Size(val width: Int, val height: Int)
}
