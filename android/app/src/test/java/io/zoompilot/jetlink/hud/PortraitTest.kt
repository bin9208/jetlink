package io.zoompilot.jetlink.hud

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The portrait turn against a model of the comma's raylib path: render
 * texture stored bottom-up, drawn with a positive source height (flipped),
 * rotated -90 degrees into a target at (0, width), read back bottom-up.
 */
class PortraitTest {
    private val width = 6
    private val height = 4
    private val landscape = IntArray(width * height) { it }

    /** The comma's portrait upload, step by step, as cluster_renderer.py does it. */
    private fun commaModel(): IntArray {
        // 1. the scene into a render texture: stored bottom row first
        val texture = IntArray(width * height) { i -> val r = i / width; val c = i % width; landscape[(height - 1 - r) * width + c] }
        // 2. drawn with source (0, 0, w, h): texture row 0 at the top of the dest rect, i.e. flipped
        // 3. rotated -90 about the dest origin at (0, w): local (lx, ly) -> screen (ly, w - lx)
        val screen = IntArray(width * height)
        for (ly in 0 until height) for (lx in 0 until width) {
            val sx = ly
            val sy = width - 1 - lx // pixel centres: (w - lx) covers row w-1-lx
            screen[sy * height + sx] = texture[ly * width + lx]
        }
        // 4. the target stored bottom row first, read back without a flip
        return IntArray(width * height) { i -> val r = i / height; val c = i % height; screen[(width - 1 - r) * height + c] }
    }

    @Test
    fun matchesTheCommasUpload() {
        val portrait = IntArray(width * height)
        Portrait.rotate(landscape, width, height, false, portrait)
        assertArrayEquals(commaModel(), portrait)
    }

    @Test
    fun turnsClockwise() {
        val portrait = IntArray(width * height)
        Portrait.rotate(landscape, width, height, false, portrait)
        // the landscape's top-left corner lands at the portrait's top right
        assertEquals(landscape[0], portrait[height - 1])
        // its top-right corner at the bottom right
        assertEquals(landscape[width - 1], portrait[(width - 1) * height + height - 1])
    }

    @Test
    fun unrotateUndoesRotate() {
        val portrait = IntArray(width * height)
        Portrait.rotate(landscape, width, height, false, portrait)
        val back = IntArray(width * height)
        Portrait.unrotate(portrait, width, height, back)
        assertArrayEquals(landscape, back)
    }

    @Test
    fun mirrorSwapsLeftAndRight() {
        val portrait = IntArray(width * height)
        Portrait.rotate(landscape, width, height, true, portrait)
        val back = IntArray(width * height)
        Portrait.unrotate(portrait, width, height, back)
        for (y in 0 until height) for (x in 0 until width) {
            assertEquals(landscape[y * width + (width - 1 - x)], back[y * width + x])
        }
    }
}
