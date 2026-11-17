package io.zoompilot.jetlink.update

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class AppUpdateTest {
    private val json = """
        [
          {"tag_name": "v0.9.0-carrot.27-test", "prerelease": true, "draft": false, "name": "t27", "body": "test",
           "html_url": "h", "assets": [{"name": "jetlink-carrot-test-0.9.0-carrot.27.apk", "browser_download_url": "u27", "size": 5}]},
          {"tag_name": "v0.9.0-carrot.26", "prerelease": false, "draft": false, "name": "s26", "body": "stable",
           "html_url": "h", "assets": [{"name": "jetlink-carrot-0.9.0-carrot.26.apk", "browser_download_url": "u26", "size": 4}]},
          {"tag_name": "v0.9.0-carrot.25", "prerelease": true, "draft": false, "name": "old", "body": "",
           "html_url": "h", "assets": [{"name": "x.apk", "browser_download_url": "u25", "size": 3}]},
          {"tag_name": "v0.9.0-carrot.30", "prerelease": false, "draft": true, "name": "draft", "body": "",
           "html_url": "h", "assets": [{"name": "d.apk", "browser_download_url": "u30", "size": 3}]}
        ]
    """.trimIndent()

    @Test fun eachChannelTakesItsOwnNewestRelease() {
        assertEquals("u26", AppUpdate.newest(json, 25, "stable")?.apkUrl)
        assertEquals("u27", AppUpdate.newest(json, 25, "test")?.apkUrl)
        assertNull(AppUpdate.newest(json, 26, "stable"))
        assertNull(AppUpdate.newest(json, 27, "test"))
    }

    @Test fun buildNumbersComeFromTheTag() {
        assertEquals(25, AppUpdate.buildOf("v0.9.0-carrot.25"))
        assertEquals(27, AppUpdate.buildOf("v0.9.0-carrot.27-test"))
        assertNull(AppUpdate.buildOf("v0.9.0"))
    }
}
