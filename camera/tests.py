from django.test import TestCase

from camera.services.dwell import DwellTracker
from camera.services.geom import point_in_polygon
from camera.services.rtsp import parse_channel, stream_url


class ZoneGeometryTests(TestCase):
    def test_point_in_square(self):
        square = [[0, 0], [1, 0], [1, 1], [0, 1]]
        self.assertTrue(point_in_polygon(0.5, 0.5, square))
        self.assertFalse(point_in_polygon(1.5, 0.5, square))

    def test_dwell_alarms_once_per_visit(self):
        tracker = DwellTracker()
        self.assertEqual(tracker.update({4: True}, 0, 60), [])
        self.assertEqual(tracker.update({4: True}, 59, 60), [])
        self.assertEqual(tracker.update({4: True}, 60, 60), [4])
        self.assertEqual(tracker.update({4: True}, 90, 60), [])
        self.assertEqual(tracker.update({4: False}, 91, 60), [])
        self.assertEqual(tracker.update({4: True}, 100, 60), [])
        self.assertEqual(tracker.update({4: True}, 160, 60), [4])

    def test_channel_numbers(self):
        self.assertEqual(parse_channel("rtsp://user@host:554/Streaming/Channels/102"), (1, "sub"))
        self.assertEqual(parse_channel("rtsp://user@host:554/Streaming/Channels/1201"), (12, "main"))

    def test_blank_image_has_no_face(self):
        import numpy as np

        from camera.services.faces import embed_image

        image = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertIsNone(embed_image(image))

    def test_close_names_are_rejected(self):
        import numpy as np

        from camera.services.faces import best_name

        ashraf = np.zeros(512, dtype=np.float32)
        ashraf[0] = 1
        sharif = np.zeros(512, dtype=np.float32)
        sharif[0] = 0.98
        sharif[1] = 0.199
        sharif /= np.linalg.norm(sharif)
        self.assertEqual(best_name(ashraf, [("Ashraf", ashraf), ("Sharif", sharif)]), "")

    def test_clear_name_is_accepted(self):
        import numpy as np

        from camera.services.faces import best_name

        ashraf = np.zeros(512, dtype=np.float32)
        ashraf[0] = 1
        sharif = np.zeros(512, dtype=np.float32)
        sharif[1] = 1
        self.assertEqual(best_name(ashraf, [("Ashraf", ashraf), ("Sharif", sharif)]), "Ashraf")

    def test_name_locks_after_two_matches(self):
        from camera.services.faces import next_identity

        votes = {}
        current, score = next_identity(votes, "", 0, "Ashraf", 0.61)
        self.assertEqual(current, "")
        current, score = next_identity(votes, current, score, "Ashraf", 0.7)
        self.assertEqual((current, score), ("Ashraf", 0.7))
        current, score = next_identity(votes, current, score, "Sharif", 0.8)
        current, score = next_identity(votes, current, score, "Sharif", 0.9)
        self.assertEqual((current, score), ("Sharif", 0.9))

    def test_old_face_vector_is_ignored(self):
        import numpy as np

        from camera.services.faces import current_vector

        self.assertIsNone(current_vector(np.zeros(128, dtype=np.float32).tobytes()))
        self.assertEqual(current_vector(np.zeros(512, dtype=np.float32).tobytes()).size, 512)

    def test_object_labels_and_defaults(self):
        from camera.services.classes import DEFAULT_ENABLED, catalog, label_for

        self.assertEqual(label_for(62), "Monitor")
        self.assertEqual(label_for(60), "Table")
        self.assertEqual(label_for(63), "Laptop")
        self.assertEqual(set(DEFAULT_ENABLED), {0, 56, 60, 62, 63})
        enabled = set(DEFAULT_ENABLED)
        groups = {group["name"]: group["items"] for group in catalog(enabled)}
        self.assertTrue(groups["People"][0]["enabled"])
        monitor = next(item for item in groups["Electronics"] if item["id"] == 62)
        mouse = next(item for item in groups["Electronics"] if item["id"] == 64)
        self.assertTrue(monitor["enabled"])
        self.assertFalse(mouse["enabled"])

    def test_object_toggle_endpoint(self):
        from camera.services.classes import DEFAULT_ENABLED
        from camera.services.engine import engine

        previous = set(engine.enabled_classes)
        engine.enabled_classes = set(DEFAULT_ENABLED)
        try:
            response = self.client.post("/objects/", {"class_id": "64"})
            self.assertEqual(response.status_code, 200)
            self.assertIn(64, engine.enabled_classes)
            body = response.json()
            mouse = next(
                item
                for group in body["classes"]
                if group["name"] == "Electronics"
                for item in group["items"]
                if item["id"] == 64
            )
            self.assertTrue(mouse["enabled"])
        finally:
            engine.enabled_classes = previous

    def test_stream_url_swaps_channel_only(self):
        from camera.services import rtsp

        original = rtsp.base_rtsp_url
        rtsp.base_rtsp_url = lambda: "rtsp://user:secret@192.168.1.10:554/Streaming/Channels/102"
        try:
            url, channel = stream_url(12, "sub")
            self.assertEqual(channel, "1202")
            self.assertIn("/Streaming/Channels/1202", url)
            self.assertTrue(url.startswith("rtsp://user:secret@192.168.1.10:554/"))
            main_url, main_channel = stream_url(12, "main")
            self.assertEqual(main_channel, "1201")
            self.assertIn("/Streaming/Channels/1201", main_url)
        finally:
            rtsp.base_rtsp_url = original
