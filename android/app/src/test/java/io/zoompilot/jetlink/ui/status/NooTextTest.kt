package io.zoompilot.jetlink.ui.status

import io.zoompilot.jetlink.server.Noo
import io.zoompilot.jetlink.ui.Tone
import org.junit.Assert.assertEquals
import org.junit.Test

/** The Phone section's NOO tile, for each state the comma's frames can be in. */
class NooTextTest {
    @Test
    fun aRouteAheadIsNoo() {
        val tile = NooText.tile(Noo(model = true, active = true, command = 15, distanceM = 324.0), connected = true)
        assertEquals("작동 중", tile.value)
        assertEquals("좌회전 · 320 m", tile.note)
        assertEquals(Tone.Good, tile.tone)
        assertEquals("우측 출구 · 1.5 km", NooText.maneuver(20, 1520.0))
        assertEquals("안내 중", NooText.maneuver(99, null))
    }

    @Test
    fun withoutARouteTurnsAreAtc() {
        val waiting = NooText.tile(Noo(model = true, active = false), connected = true)
        assertEquals("대기", waiting.value)
        assertEquals(Tone.Warning, waiting.tone)
        val other = NooText.tile(Noo(model = false), connected = true)
        assertEquals("꺼짐", other.value)
        assertEquals("NOO 모델 아님 · ATC", other.note)
    }

    @Test
    fun beforeFramesNothingIsClaimed() {
        assertEquals("콤마 연결 안 됨", NooText.tile(null, connected = false).note)
        assertEquals("프레임 기다리는 중", NooText.tile(null, connected = true).note)
    }
}
