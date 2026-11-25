package io.zoompilot.jetlink.ui.status

import io.zoompilot.jetlink.server.Noo
import io.zoompilot.jetlink.ui.Tone
import java.util.Locale

/**
 * The Phone section's NOO tile: whether the comma's frames give the model a
 * maneuver (NOO drives the turn) or not (carrot's ATC does), in words.
 */
object NooText {
    data class Tile(val value: String, val note: String, val tone: Tone)

    fun tile(noo: Noo?, connected: Boolean): Tile = when {
        !connected -> Tile("—", "콤마 연결 안 됨", Tone.Neutral)
        noo == null -> Tile("—", "프레임 기다리는 중", Tone.Neutral)
        !noo.model -> Tile("꺼짐", "NOO 모델 아님 · ATC", Tone.Neutral)
        noo.active -> Tile("작동 중", maneuver(noo.command, noo.distanceM), Tone.Good)
        else -> Tile("대기", "경로 안내 없음 · ATC", Tone.Warning)
    }

    /** "좌회전 · 320 m": the maneuver the model is given and how far. */
    fun maneuver(command: Int?, distanceM: Double?): String {
        val name = command?.let(::command) ?: "안내 중"
        val distance = distanceM?.let(::distance) ?: return name
        return "$name · $distance"
    }

    /** A Valhalla maneuver type, as carrot's navigation input sets them. */
    fun command(type: Int): String = when (type) {
        7 -> "계속"
        8 -> "직진"
        9 -> "약간 우회전"
        10 -> "우회전"
        11 -> "급우회전"
        12, 13 -> "유턴"
        14 -> "급좌회전"
        15 -> "좌회전"
        16 -> "약간 좌회전"
        17 -> "진입로"
        18 -> "우측 진입로"
        19 -> "좌측 진입로"
        20 -> "우측 출구"
        21 -> "좌측 출구"
        22 -> "분기 직진"
        23 -> "우측 분기"
        24 -> "좌측 분기"
        25 -> "합류"
        26 -> "회전교차로"
        27 -> "회전교차로 진출"
        else -> "안내 중"
    }

    fun distance(metres: Double): String = when {
        metres >= 1000 -> String.format(Locale.ROOT, "%.1f km", metres / 1000)
        else -> "${(metres / 10).toInt() * 10} m"
    }
}
