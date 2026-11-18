package io.zoompilot.jetlink.hud

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.security.MessageDigest
import java.time.ZoneId
import java.time.ZonedDateTime

/**
 * The TURZX packets against vectors made outside this code: the plain
 * command as packet.rs builds it, encrypted by `openssl enc -des-cbc -nopad`
 * with the vendor key as key and IV, then the 161/26 trailer.
 */
class TurzxPacketTest {
    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    private fun hex(text: String) = text.chunked(2).map { it.toInt(16).toByte() }.toByteArray()

    @Test
    fun initCommandMatchesOpenssl() {
        val packet = TurzxPacket.command(TurzxPacket.INIT, 12_345_678)
        assertEquals(512, packet.size)
        assertArrayEquals(hex("019a3db35a5ba2ad1c6c82e3dc9bdc9a"), packet.copyOfRange(0, 16))
        assertArrayEquals(hex("0d5211fc95a1f81b000000000000a11a"), packet.copyOfRange(496, 512))
        assertEquals("d214d3f0e60d33454c2e344a09ca077eb2c4d66d69ca2bbbcb9e4faadc02c2db", sha256(packet))
    }

    @Test
    fun jpegFrameCarriesItsSizeBigEndian() {
        val payload = ByteArray(70_000) { it.toByte() }
        val frame = TurzxPacket.frame(TurzxPacket.JPEG, 1000, payload)
        assertEquals(512 + payload.size, frame.size)
        assertEquals("212dc3b625567b2be20d09232be3b91671b8cfa4ca80c7084ae6e705d4930c50", sha256(frame.copyOfRange(0, 512)))
        assertArrayEquals(payload, frame.copyOfRange(512, frame.size))
    }

    @Test
    fun brightnessIsTheCommasByte() {
        assertEquals(0, TurzxPacket.brightness(0))
        assertEquals(81, TurzxPacket.brightness(80))
        assertEquals(102, TurzxPacket.brightness(100))
        assertEquals(102, TurzxPacket.brightness(250))
    }

    @Test
    fun timestampIsMillisecondsSinceLocalMidnight() {
        val now = ZonedDateTime.of(2026, 11, 18, 7, 39, 4, 56_000_000, ZoneId.of("Asia/Seoul"))
        assertEquals(((7 * 60 + 39) * 60 + 4) * 1000L + 56, TurzxPacket.millisecondsToday(now))
    }

    @Test
    fun sizesAreThePanelsLandscapeSizes() {
        assertEquals(TurzxPacket.Size(1920, 462), TurzxPacket.size(0x92))
        assertEquals(TurzxPacket.Size(1920, 720), TurzxPacket.size(0x123))
        assertNull(TurzxPacket.size(0x1234))
    }
}
