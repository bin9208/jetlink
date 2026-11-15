package io.zoompilot.jetlink.ui.logs

import android.widget.Toast
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.zoompilot.jetlink.AppGraph
import io.zoompilot.jetlink.logs.LogBundle
import io.zoompilot.jetlink.logs.LogUpload
import io.zoompilot.jetlink.ui.Format
import io.zoompilot.jetlink.ui.JetlinkTheme
import io.zoompilot.jetlink.ui.LogLevel
import io.zoompilot.jetlink.ui.components.CardSpacing
import io.zoompilot.jetlink.ui.components.PushedScreen
import io.zoompilot.jetlink.ui.components.shareFile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * What the server and the app have logged, newest at the bottom, following
 * new lines until the reader scrolls up. Warnings orange, errors red.
 */
@Composable
fun LogsScreen(graph: AppGraph, back: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val lines by graph.server.logs.collectAsStateWithLifecycle()
    var uploading by remember { mutableStateOf(false) }
    // A file, not an intent's text, which would have to fit one Binder
    // transaction; with the phone's facts and this process's Android log,
    // where an NPU's refusal is written.
    suspend fun bundle() = withContext(Dispatchers.IO) { LogBundle.write(context, lines, graph.settings.values.value.processor) }
    LogsContent(
        lines = lines,
        back = back,
        share = {
            scope.launch { shareFile(context, bundle(), "text/plain", "Jetlink Logs") }
        },
        upload = {
            uploading = true
            scope.launch {
                val result = runCatching { withContext(Dispatchers.IO) { LogUpload.upload(context, bundle()) } }
                uploading = false
                val message = result.fold({ "Logs sent to $it" }, { "Upload failed: ${it.message}" })
                Toast.makeText(context, message, Toast.LENGTH_LONG).show()
            }
        },
        uploading = uploading,
        clear = graph.server::clearLogs,
    )
}

@Composable
fun LogsContent(
    lines: List<String>,
    back: () -> Unit,
    share: () -> Unit,
    clear: () -> Unit,
    upload: () -> Unit = {},
    uploading: Boolean = false,
) {
    val colors = JetlinkTheme.colors
    val list = rememberLazyListState()
    var follow by remember { mutableStateOf(true) }
    // While the reader scrolls, follow new lines only from the bottom.
    LaunchedEffect(list) {
        snapshotFlow { list.isScrollInProgress to list.canScrollForward }.collect { (scrolling, more) ->
            if (scrolling) follow = !more
        }
    }
    LaunchedEffect(lines.size) {
        if (follow && lines.isNotEmpty()) list.scrollToItem(lines.lastIndex)
    }
    PushedScreen(
        "Logs",
        back,
        actions = {
            IconButton(onClick = upload, enabled = !uploading) { Icon(Icons.Filled.CloudUpload, contentDescription = "Send to the log server") }
            IconButton(onClick = share) { Icon(Icons.Filled.Share, contentDescription = "Share") }
            IconButton(onClick = clear, enabled = lines.isNotEmpty()) { Icon(Icons.Filled.Delete, contentDescription = "Clear") }
        },
    ) { padding ->
        Box(Modifier.padding(padding).consumeWindowInsets(padding).fillMaxSize()) {
            if (lines.isEmpty()) {
                Column(
                    Modifier.align(Alignment.Center),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Icon(Icons.Filled.Description, contentDescription = null, tint = colors.secondaryText, modifier = Modifier.size(48.dp))
                    Text("No Logs", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                }
            } else {
                SelectionContainer {
                    LazyColumn(
                        Modifier.fillMaxSize(),
                        state = list,
                        contentPadding = PaddingValues(horizontal = CardSpacing, vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(2.dp),
                    ) {
                        itemsIndexed(lines) { _, line ->
                            Text(
                                line,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.sp,
                                lineHeight = 14.sp,
                                color = when (Format.logLevel(line)) {
                                    LogLevel.Error -> colors.bad
                                    LogLevel.Warning -> colors.warning
                                    LogLevel.Info -> Color.Unspecified
                                },
                                modifier = Modifier.fillMaxWidth(),
                            )
                        }
                    }
                }
            }
        }
    }
}

@Preview(showBackground = true, heightDp = 500)
@Composable
private fun LogsPreview() {
    JetlinkTheme {
        LogsContent(
            lines = listOf(
                "2026-09-27 12:53:20,114 INFO jetlink.server: listening on port 5599",
                "2026-09-27 12:53:22,871 INFO jetlink.link: comma connected over USB 3",
                "2026-09-27 12:53:24,982 WARNING jetlink.engine: frame 1841 took 61.2 ms",
                "2026-09-27 12:53:25,003 ERROR   jetlink.link: the device went away",
            ),
            back = {},
            share = {},
            clear = {},
        )
    }
}
