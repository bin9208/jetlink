package io.zoompilot.jetlink.hud

/**
 * The turn from the HUD's landscape picture to the TURZX's portrait frame,
 * as the comma's upload ends up (carrot cluster_renderer.py `_render_to_image`
 * with portrait_upload, ported as openpilot-rust renderer/output.rs):
 *
 * - the scene is drawn into a raylib render texture, which stores it upside
 *   down (bottom row first);
 * - that texture is drawn with a positive source height, so it lands flipped
 *   vertically, rotated by -90 degrees into a (height x width) target at
 *   (0, width);
 * - that target, upside down in memory again, is read back without a flip.
 *
 * Together: portrait pixel (x, y) is landscape pixel (y, height-1-x), the
 * landscape turned 90 degrees clockwise. Its top-left corner lands top right.
 * Mirror swaps left and right in the landscape picture first.
 */
object Portrait {
    /** [landscape] ([width] x [height], row-major) into [portrait] ([height] wide, [width] tall). */
    fun rotate(landscape: IntArray, width: Int, height: Int, mirror: Boolean, portrait: IntArray) {
        require(landscape.size >= width * height && portrait.size >= width * height)
        for (y in 0 until width) {
            val lx = if (mirror) width - 1 - y else y
            val row = y * height
            for (x in 0 until height) {
                portrait[row + x] = landscape[(height - 1 - x) * width + lx]
            }
        }
    }

    /** The inverse of [rotate] without mirror: what the panel shows, back in landscape. */
    fun unrotate(portrait: IntArray, width: Int, height: Int, landscape: IntArray) {
        require(landscape.size >= width * height && portrait.size >= width * height)
        for (ly in 0 until height) {
            val x = height - 1 - ly
            val row = ly * width
            for (lx in 0 until width) {
                landscape[row + lx] = portrait[lx * height + x]
            }
        }
    }
}
