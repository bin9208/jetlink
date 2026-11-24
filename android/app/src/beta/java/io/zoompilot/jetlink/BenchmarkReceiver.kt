package io.zoompilot.jetlink

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import kotlinx.coroutines.launch

/**
 * The test app's benchmark from `adb shell am broadcast` (see the beta
 * manifest): the same run as the Benchmark screen's button, its results in
 * the log as ever. Only the shell can send it (android.permission.DUMP).
 */
class BenchmarkReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val seconds = intent.getIntExtra("seconds", 60).coerceIn(10, 600)
        val pending = goAsync()
        context.graph.scope.launch {
            try {
                val reply = context.graph.server.benchmark(seconds)
                Log.i(TAG, "benchmark from adb: ${seconds} s, ${if (reply.ok) "started" else "refused: ${reply.error}"}")
            } finally {
                pending.finish()
            }
        }
    }

    private companion object {
        const val TAG = "jetlink"
    }
}
