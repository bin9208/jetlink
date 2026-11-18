package io.zoompilot.jetlink

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import io.zoompilot.jetlink.server.ServerService

/**
 * What Android opens when the comma or a TURZX HUD panel is plugged in: the
 * device filter matches the jetlink gadget and the panels, and with "Always
 * open" ticked Android grants Jetlink the device without asking again. Hands
 * it to the server's service (the comma's link, the HUD), brings the
 * dashboard up, and goes.
 */
class UsbAttachActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ServerService.start(this, ServerService.ACTION_USB)
        startActivity(Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT))
        finish()
    }
}
