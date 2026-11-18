package io.zoompilot.jetlink.hud

import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbEndpoint
import android.hardware.usb.UsbInterface
import android.hardware.usb.UsbManager
import android.util.Log
import java.io.Closeable
import java.io.IOException

/**
 * A TURZX panel on this phone's USB (through a hub, beside the comma), driven
 * the way the comma drives one on its own port (openpilot-rust
 * rust/crates/cluster/src/usb/{connection,io,display,frame}.rs): interface 0
 * claimed, commands written with their halts cleared and the IN endpoint
 * drained first, JPEG frames written without waiting for an answer.
 *
 * Android's bulkTransfer is enough here: unlike the comma's link, nothing
 * arriving is lost when a read times out, since only drained acks come back.
 */
class TurzxDisplay private constructor(
    val device: UsbDevice,
    private val connection: UsbDeviceConnection,
    private val iface: UsbInterface,
    private val output: UsbEndpoint,
    private val input: UsbEndpoint,
) : Closeable {
    val product: Int = device.productId
    val size: TurzxPacket.Size = TurzxPacket.size(product) ?: throw IOException("unknown TURZX product ${product.toString(16)}")
    private var closed = false

    /**
     * The comma's start: init, acknowledged; then brightness and orientation,
     * which the panel takes without answering. Throws when the panel does
     * not answer the init.
     */
    fun initialize(brightness: Int, rotate180: Boolean) {
        commandAck(TurzxPacket.INIT)
        Thread.sleep(200)
        optionalCommand(TurzxPacket.BRIGHTNESS, listOf(8 to TurzxPacket.brightness(brightness)))
        setOrientation(rotate180)
    }

    fun setBrightness(percent: Int) = setting(TurzxPacket.BRIGHTNESS, TurzxPacket.brightness(percent))

    /** 0 or 180 degrees, the only two the panel takes (ClusterHudOrientation 0 or 2). */
    fun setOrientation(rotate180: Boolean) = setting(TurzxPacket.ORIENTATION, if (rotate180) 2 else 0)

    /** One JPEG frame, in the panel's own (portrait) orientation. Throws when the write fails. */
    fun sendJpeg(jpeg: ByteArray) {
        drain(FRAME_DRAINS, FRAME_DRAIN_MS)
        write(TurzxPacket.frame(TurzxPacket.JPEG, TurzxPacket.millisecondsToday(), jpeg))
    }

    /** Dims the panel and lets the device go, as the comma does when its HUD stops. */
    override fun close() {
        if (closed) return
        closed = true
        runCatching { optionalCommand(TurzxPacket.BRIGHTNESS, listOf(8 to 0)) }
        connection.releaseInterface(iface)
        connection.close()
    }

    /** The comma's setting change: an init, 20 ms, then the setting, none of them answered. */
    private fun setting(id: Int, value: Int) {
        optionalCommand(TurzxPacket.INIT)
        Thread.sleep(20)
        optionalCommand(id, listOf(8 to value))
    }

    private fun commandAck(id: Int, fields: List<Pair<Int, Int>> = emptyList()): ByteArray {
        clearHalt()
        drain(3, 20)
        write(TurzxPacket.command(id, TurzxPacket.millisecondsToday(), fields))
        val response = ByteArray(512)
        val count = connection.bulkTransfer(input, response, response.size, ACK_TIMEOUT_MS)
        if (count <= 0) throw IOException("TURZX did not answer command $id")
        return response.copyOf(count)
    }

    /** A command the panel does not answer; a failure is logged and skipped, as on the comma. */
    private fun optionalCommand(id: Int, fields: List<Pair<Int, Int>> = emptyList()) {
        try {
            clearHalt()
            drain(3, 20)
            write(TurzxPacket.command(id, TurzxPacket.millisecondsToday(), fields))
        } catch (error: IOException) {
            Log.w(TAG, "optional TURZX command $id skipped: ${error.message}")
        }
    }

    /** CLEAR_FEATURE(ENDPOINT_HALT) on both endpoints; errors are ignored, as the original does. */
    private fun clearHalt() {
        for (endpoint in listOf(output, input)) {
            connection.controlTransfer(0x02, 0x01, 0, endpoint.address, null, 0, 100)
        }
    }

    /** Reads up to [attempts] stale answers, stopping at the first timeout. */
    private fun drain(attempts: Int, timeoutMs: Int) {
        val buffer = ByteArray(512)
        repeat(attempts) {
            if (connection.bulkTransfer(input, buffer, buffer.size, timeoutMs) < 0) return
        }
    }

    /**
     * One logical write, in pieces of [CHUNK] bytes: a multiple of every bulk
     * packet size, so the panel sees one unbroken transfer, as libusb splits it.
     */
    private fun write(bytes: ByteArray) {
        var offset = 0
        while (offset < bytes.size) {
            val length = minOf(CHUNK, bytes.size - offset)
            val sent = connection.bulkTransfer(output, bytes, offset, length, WRITE_TIMEOUT_MS)
            if (sent <= 0) throw IOException("TURZX write failed at $offset of ${bytes.size} bytes")
            offset += sent
        }
    }

    companion object {
        private const val TAG = "jetlink"
        private const val CHUNK = 16 * 1024
        private const val WRITE_TIMEOUT_MS = 2000
        private const val ACK_TIMEOUT_MS = 2000

        /** Options.frame_drain_attempts / frame_drain_timeout_ms, the comma's defaults. */
        private const val FRAME_DRAINS = 2
        private const val FRAME_DRAIN_MS = 2

        fun isTurzx(device: UsbDevice): Boolean =
            device.vendorId == TurzxPacket.VENDOR_ID && TurzxPacket.size(device.productId) != null

        /** Opens and claims interface 0, as the comma does. Throws if Android or the panel will not. */
        fun open(manager: UsbManager, device: UsbDevice): TurzxDisplay {
            val iface = (0 until device.interfaceCount).map(device::getInterface).firstOrNull { it.id == 0 }
                ?: throw IOException("TURZX has no interface 0")
            val endpoints = (0 until iface.endpointCount).map(iface::getEndpoint)
                .filter { it.type == UsbConstants.USB_ENDPOINT_XFER_BULK }
            val output = endpoints.firstOrNull { it.direction == UsbConstants.USB_DIR_OUT }
                ?: throw IOException("TURZX has no bulk OUT endpoint")
            val input = endpoints.firstOrNull { it.direction == UsbConstants.USB_DIR_IN }
                ?: throw IOException("TURZX has no bulk IN endpoint")
            val connection = manager.openDevice(device) ?: throw IOException("Android would not open the TURZX")
            if (!connection.claimInterface(iface, true)) {
                connection.close()
                throw IOException("another app holds the TURZX")
            }
            return TurzxDisplay(device, connection, iface, output, input)
        }
    }
}
