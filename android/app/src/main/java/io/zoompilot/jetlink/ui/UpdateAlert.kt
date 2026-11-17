package io.zoompilot.jetlink.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import io.zoompilot.jetlink.BuildConfig
import io.zoompilot.jetlink.update.AppUpdate
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private sealed interface UpdateState {
    data object None : UpdateState
    data class Offer(val release: AppUpdate.Release) : UpdateState
    data class Downloading(val release: AppUpdate.Release, val frac: Double) : UpdateState
    data class Ready(val release: AppUpdate.Release, val apk: File) : UpdateState
    data class Failed(val release: AppUpdate.Release, val reason: String) : UpdateState
}

/** "업데이트가 있어요": a newer release of this app's channel, downloaded and installed from here. */
@Composable
fun UpdateAlert() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf<UpdateState>(UpdateState.None) }
    LaunchedEffect(Unit) {
        val release = withContext(Dispatchers.IO) { runCatching { AppUpdate.check(context) }.getOrNull() }
        if (release != null) state = UpdateState.Offer(release)
    }
    fun start(release: AppUpdate.Release) {
        state = UpdateState.Downloading(release, 0.0)
        scope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    AppUpdate.download(context, release) { frac ->
                        scope.launch { state = UpdateState.Downloading(release, frac) }
                    }
                }
            }
            state = result.fold(
                { UpdateState.Ready(release, it) },
                { UpdateState.Failed(release, it.message ?: it.toString()) },
            )
        }
    }
    fun install(release: AppUpdate.Release, apk: File) {
        if (!AppUpdate.canInstall(context)) {
            // Android asks once whether this app may install updates
            context.startActivity(AppUpdate.allowInstallSettings(context))
            return
        }
        context.startActivity(AppUpdate.installIntent(context, apk))
        state = UpdateState.None
    }
    val channel = if (AppUpdate.channel == "test") " (테스트)" else ""
    when (val s = state) {
        UpdateState.None -> {}
        is UpdateState.Offer -> AlertDialog(
            onDismissRequest = { state = UpdateState.None },
            title = { Text("업데이트가 있어요$channel") },
            text = {
                Column(Modifier.heightIn(max = 360.dp).verticalScroll(rememberScrollState())) {
                    Text("${BuildConfig.VERSION_NAME} → ${s.release.tag.removePrefix("v")}", style = MaterialTheme.typography.titleSmall)
                    if (s.release.apkBytes > 0) Text("%.0f MB".format(s.release.apkBytes / 1e6), style = MaterialTheme.typography.bodySmall)
                    Text(s.release.notes.take(1500), Modifier.padding(top = 8.dp), style = MaterialTheme.typography.bodySmall)
                }
            },
            confirmButton = { TextButton(onClick = { start(s.release) }) { Text("업데이트") } },
            dismissButton = {
                TextButton(onClick = {
                    AppUpdate.skip(context, s.release)
                    state = UpdateState.None
                }) { Text("나중에") }
            },
        )
        is UpdateState.Downloading -> AlertDialog(
            onDismissRequest = {},
            title = { Text("내려받는 중") },
            text = {
                Column {
                    Text("${s.release.tag.removePrefix("v")} ${(s.frac * 100).toInt()}%")
                    LinearProgressIndicator(progress = { s.frac.toFloat() }, modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
                }
            },
            confirmButton = {},
        )
        is UpdateState.Ready -> AlertDialog(
            onDismissRequest = { state = UpdateState.None },
            title = { Text("설치할 준비가 됐어요") },
            text = {
                Text(
                    if (AppUpdate.canInstall(context)) "설치를 누르면 Android 설치 화면이 열립니다."
                    else "처음 한 번은 '이 출처 허용'을 켜야 합니다. 설정에서 켠 뒤 다시 설치를 눌러 주세요."
                )
            },
            confirmButton = { TextButton(onClick = { install(s.release, s.apk) }) { Text("설치") } },
            dismissButton = { TextButton(onClick = { state = UpdateState.None }) { Text("닫기") } },
        )
        is UpdateState.Failed -> AlertDialog(
            onDismissRequest = { state = UpdateState.None },
            title = { Text("내려받지 못했어요") },
            text = { Text(s.reason) },
            confirmButton = { TextButton(onClick = { start(s.release) }) { Text("다시 시도") } },
            dismissButton = { TextButton(onClick = { state = UpdateState.None }) { Text("닫기") } },
        )
    }
}
