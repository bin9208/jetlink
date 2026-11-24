package io.zoompilot.jetlink.settings

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Which NPU experiments each app offers, and the main app's default on an 8 Elite (carrot.54). */
class ExperimentsTest {
    @Test
    fun theMainAppOffersTheHot192GuardOnAnEightEliteAlone() {
        assertEquals(listOf(MAIN_APP_EXPERIMENT), offeredExperiments(testApp = false, hexagon = 79).map { it.first })
        assertTrue(offeredExperiments(testApp = false, hexagon = 73).isEmpty())
        assertTrue(offeredExperiments(testApp = false, hexagon = null).isEmpty())
    }

    @Test
    fun theTestAppKeepsItsSweepList() {
        val v79 = offeredExperiments(testApp = true, hexagon = 79).map { it.first }
        assertTrue(MAIN_APP_EXPERIMENT in v79 && "norm-guard-224" in v79 && "fold-vision-prescale" in v79)
        // no norm guard on a v73: every one made N16 far worse on a Tab S9 Ultra
        assertTrue(offeredExperiments(testApp = true, hexagon = 73).none { it.first.startsWith("norm-guard") })
    }

    @Test
    fun theMainAppTurnsTheGuardOnOnceOnAnEightElite() {
        assertEquals(setOf(MAIN_APP_EXPERIMENT), Settings.migratedExperiments(null, stable = true, hexagon = 79))
        assertEquals(setOf("x", MAIN_APP_EXPERIMENT), Settings.migratedExperiments(setOf("x"), stable = true, hexagon = 79))
        assertNull(Settings.migratedExperiments(null, stable = true, hexagon = 73))
        assertNull(Settings.migratedExperiments(null, stable = false, hexagon = 79))
    }
}
