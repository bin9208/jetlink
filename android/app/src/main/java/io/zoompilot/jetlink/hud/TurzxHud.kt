package io.zoompilot.jetlink.hud

import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Process
import android.os.SystemClock
import android.util.Log
import io.zoompilot.jetlink.server.Snapshot
import io.zoompilot.jetlink.settings.HudSettings
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.IOException
import java.time.LocalDateTime

/** Where the external HUD stands. */
sealed interface HudState {
    /** No TURZX panel on the USB. */
    data object None : HudState

    /** A panel is plugged in and the HUD setting is off. */
    data object Off : HudState

    /** A panel is plugged in, and Android has not let Jetlink use it yet. */
    data object NeedsPermission : HudState

    data class Running(val name: String, val fps: Double) : HudState

    data class Failed(val reason: String) : HudState
}

/**
 * The external HUD on the phone: a TURZX panel on the phone's USB, through a
 * hub that also carries the comma and powers both. Jetlink draws it itself
 * ([HudRenderer]) on a background thread, so the comma neither renders nor
 * encodes it, and the model's frames go first.
 */
class TurzxHud(
    private val context: Context,
    private val snapshot: StateFlow<Snapshot>,
    private val settings: () -> HudSettings,
) {
    private val manager = context.getSystemService(UsbManager::class.java)
    private val hudState = MutableStateFlow<HudState>(HudState.None)
    val hud: StateFlow<HudState> = hudState.asStateFlow()

    private var worker: Worker? = null

    fun find(): UsbDevice? = manager.deviceList.values.firstOrNull(TurzxDisplay::isTurzx)

    /** Starts the HUD if a panel is plugged in and the setting is on; stops it otherwise. Safe to call again. */
    @Synchronized
    fun connect() {
        val device = find()
        when {
            device == null -> {
                stop()
                hudState.value = HudState.None
            }
            !settings().enabled -> {
                stop()
                hudState.value = HudState.Off
            }
            worker?.let { it.device.deviceName == device.deviceName && it.isAlive } == true -> Unit
            !manager.hasPermission(device) -> {
                hudState.value = HudState.NeedsPermission
                requestPermission(device)
            }
            else -> {
                stop()
                worker = Worker(device).also { it.start() }
            }
        }
    }

    /** A device went away; stops the HUD if it was the panel. */
    @Synchronized
    fun detached(device: UsbDevice?) {
        if (device == null || worker?.device?.deviceName == device.deviceName) stop()
        if (find() == null) hudState.value = HudState.None
    }

    @Synchronized
    fun stop() {
        val current = worker ?: return
        worker = null
        current.finish()
        if (hudState.value is HudState.Running) hudState.value = HudState.None
    }

    private fun requestPermission(device: UsbDevice) {
        val intent = Intent(ACTION_PERMISSION).setPackage(context.packageName)
        val pending = PendingIntent.getBroadcast(context, 1, intent, PendingIntent.FLAG_MUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        manager.requestPermission(device, pending)
    }

    /** Opens the panel, then draws and sends a frame every [FRAME_MS] until stopped or the panel is gone. */
    private inner class Worker(val device: UsbDevice) : Thread("jetlink-hud") {
        @Volatile private var running = true

        /** A frame went out since the panel was last opened: a later failure starts the count again. */
        private var delivered = false

        fun finish() {
            running = false
            interrupt()
            join(3000)
        }

        override fun run() {
            // below the model's threads: the HUD can wait, the comma's frames cannot
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
            var failures = 0
            while (running) {
                try {
                    serve()
                    return
                } catch (_: InterruptedException) {
                    return
                } catch (error: IOException) {
                    if (!running) return
                    if (find()?.deviceName != device.deviceName) {
                        hudState.value = HudState.None
                        return
                    }
                    if (delivered) failures = 0
                    delivered = false
                    failures += 1
                    Log.w(TAG, "HUD failed ($failures/$MAX_FAILURES): ${error.message}")
                    if (failures >= MAX_FAILURES) {
                        hudState.value = HudState.Failed(error.message ?: "the TURZX stopped taking frames; unplug and plug it in again")
                        return
                    }
                    try {
                        sleep(RECONNECT_MS)
                    } catch (_: InterruptedException) {
                        return
                    }
                }
            }
        }

        /** One session with the panel; returns when stopped, throws when the panel fails. */
        private fun serve() {
            TurzxDisplay.open(manager, device).use { display ->
                var applied = settings()
                display.initialize(applied.brightness, applied.rotate180)
                val renderer = HudRenderer(display.size)
                val name = TurzxPacket.name(display.product)
                Log.i(TAG, "HUD on $name ${display.size.width}x${display.size.height}")
                hudState.value = HudState.Running(name, 0.0)
                var frames = 0
                var since = SystemClock.elapsedRealtime()
                while (running) {
                    val started = SystemClock.elapsedRealtime()
                    val wanted = settings()
                    if (wanted.brightness != applied.brightness) display.setBrightness(wanted.brightness)
                    if (wanted.rotate180 != applied.rotate180) display.setOrientation(wanted.rotate180)
                    applied = wanted
                    display.sendJpeg(renderer.render(snapshot.value, LocalDateTime.now(), wanted.mirror))
                    frames += 1
                    delivered = true
                    val now = SystemClock.elapsedRealtime()
                    if (now - since >= 5000) {
                        hudState.value = HudState.Running(name, frames * 1000.0 / (now - since))
                        frames = 0
                        since = now
                    }
                    val rest = FRAME_MS - (SystemClock.elapsedRealtime() - started)
                    if (rest > 0) sleep(rest)
                }
            }
        }
    }

    companion object {
        private const val TAG = "jetlink"
        const val ACTION_PERMISSION = "io.zoompilot.jetlink.HUD_PERMISSION"

        /** Five frames a second: a status screen needs no more, and the comma's own HUD runs at ten. */
        private const val FRAME_MS = 200L
        private const val RECONNECT_MS = 1500L
        private const val MAX_FAILURES = 3
    }
}
