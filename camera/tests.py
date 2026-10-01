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


class NvrSettingsTests(TestCase):
    def test_build_rtsp_url_encodes_special_characters(self):
        from camera.models import Nvr
        from camera.services.rtsp import base_rtsp_url, build_rtsp_url

        url = build_rtsp_url("192.168.1.10", "admin", "pass@word")
        self.assertEqual(url, "rtsp://admin:pass%40word@192.168.1.10:554/Streaming/Channels/102")
        Nvr.objects.create(host="10.0.0.5", username="user", password="p@ss")
        self.assertEqual(base_rtsp_url(), "rtsp://user:p%40ss@10.0.0.5:554/Streaming/Channels/102")

    def test_settings_prefills_ip_and_username_from_env(self):
        from camera.services import rtsp

        original = rtsp.load_env_value
        rtsp.load_env_value = (
            lambda key: "rtsp://admin:pass%40word@192.168.1.20:554/Streaming/Channels/1202"
            if key == "RTSP_URL"
            else ""
        )
        try:
            response = self.client.get("/settings/")
        finally:
            rtsp.load_env_value = original
        self.assertContains(response, 'value="192.168.1.20"')
        self.assertContains(response, 'value="admin"')
        self.assertNotContains(response, "pass@word")
        self.assertNotContains(response, "Leave the password blank")

    def test_settings_saves_and_keeps_password_when_blank(self):
        from camera.models import Nvr

        created = self.client.post(
            "/settings/",
            {"host": "192.168.1.20", "username": "admin", "password": "pass@word"},
        )
        self.assertEqual(created.status_code, 302)
        nvr = Nvr.objects.get()
        self.assertEqual((nvr.host, nvr.username, nvr.password), ("192.168.1.20", "admin", "pass@word"))

        updated = self.client.post(
            "/settings/",
            {"host": "192.168.68.80", "username": "operator", "password": ""},
        )
        self.assertEqual(updated.status_code, 302)
        nvr.refresh_from_db()
        self.assertEqual((nvr.host, nvr.username, nvr.password), ("192.168.68.80", "operator", "pass@word"))
        self.assertEqual(Nvr.objects.count(), 1)

        page = self.client.get("/settings/")
        self.assertNotContains(page, "Leave the password blank")
        self.assertContains(page, 'value="192.168.68.80"')
        self.assertContains(page, 'type="password"')
        self.assertContains(page, 'value="pass@word"')

    def test_settings_rejects_bad_ip_and_missing_first_password(self):
        from camera.models import Nvr

        bad_ip = self.client.post(
            "/settings/",
            {"host": "not-an-ip", "username": "admin", "password": "secret"},
        )
        self.assertContains(bad_ip, "Enter a valid IPv4 address.")
        missing = self.client.post(
            "/settings/",
            {"host": "192.168.1.2", "username": "admin", "password": ""},
        )
        self.assertContains(missing, "Password is required.")
        self.assertEqual(Nvr.objects.count(), 0)


class FaceQualityTests(TestCase):
    def test_flat_frame_is_blurrier_than_a_sharp_pattern(self):
        import numpy as np

        from camera.services.faces import MIN_BLUR, blur_score

        flat = np.full((90, 90), 128, dtype=np.uint8)
        sharp = np.zeros((90, 90), dtype=np.uint8)
        sharp[::2, ::2] = 255
        self.assertLess(blur_score(flat), blur_score(sharp))
        self.assertLess(blur_score(flat), MIN_BLUR)

    def test_save_requires_a_large_sharp_front_face(self):
        from camera.services.faces import assess_face

        label_ok, save_ok, quality = assess_face(0.9, 100, 120, 5)
        self.assertTrue(label_ok)
        self.assertTrue(save_ok)
        self.assertGreater(quality, 0)
        self.assertFalse(assess_face(0.4, 100, 120, 0)[0])
        self.assertFalse(assess_face(0.9, 30, 120, 0)[0])
        self.assertTrue(assess_face(0.9, 100, 10, 0)[0])
        self.assertFalse(assess_face(0.9, 100, 10, 0)[1])
        self.assertFalse(assess_face(0.9, 100, 120, 45)[1])

    def test_side_face_is_labeled_but_not_saved(self):
        from camera.services.faces import assess_face

        label_ok, save_ok, quality = assess_face(0.9, 40, 120, 135)
        self.assertTrue(label_ok)
        self.assertFalse(save_ok)
        self.assertEqual(quality, 0)
        label_ok, save_ok, quality = assess_face(0.9, 100, 120, 170)
        self.assertFalse(label_ok)
        self.assertFalse(save_ok)
        self.assertEqual(quality, 0)

    def test_blacklist_alarm_needs_a_stronger_score(self):
        from camera.services.faces import blacklist_alarm_ready

        self.assertFalse(blacklist_alarm_ready("blacklist", 0.48))
        self.assertTrue(blacklist_alarm_ready("blacklist", 0.52))
        self.assertFalse(blacklist_alarm_ready("whitelist", 0.9))

    def test_close_people_are_not_chosen(self):
        import numpy as np

        from camera.services.faces import best_person

        ashraf = np.zeros(512, dtype=np.float32)
        ashraf[0] = 1
        sharif = np.zeros(512, dtype=np.float32)
        sharif[0] = 0.98
        sharif[1] = 0.199
        sharif /= np.linalg.norm(sharif)
        chosen, _score = best_person(ashraf, [("Ashraf", ashraf), ("Sharif", sharif)])
        self.assertIsNone(chosen)
        other = np.zeros(512, dtype=np.float32)
        other[1] = 1
        chosen, score = best_person(ashraf, [("Ashraf", ashraf), ("Other", other)])
        self.assertEqual(chosen, "Ashraf")
        self.assertGreater(score, 0.9)

    def test_two_samples_of_one_person_still_match(self):
        import numpy as np

        from camera.services.faces import best_person

        front = np.zeros(512, dtype=np.float32)
        front[0] = 1
        side = front.copy()
        side[2] = 0.15
        side /= np.linalg.norm(side)
        chosen, score = best_person(front, [("Ashraf", front), ("Ashraf", side)])
        self.assertEqual(chosen, "Ashraf")
        self.assertGreater(score, 0.9)

    def test_front_face_yaw_is_near_zero(self):
        import numpy as np

        from camera.services.faces import yaw_from_kps

        kps = np.array(
            [[30, 40], [70, 40], [50, 55], [35, 75], [65, 75]],
            dtype=np.float32,
        )
        yaw = yaw_from_kps(kps)
        self.assertIsNotNone(yaw)
        self.assertLess(abs(yaw), 8)
        self.assertIsNone(yaw_from_kps(None))

    def test_turned_face_yaw_is_large(self):
        import numpy as np

        from camera.services.faces import SAVE_MAX_YAW, yaw_from_kps

        kps = np.array(
            [[30, 40], [70, 40], [78, 55], [40, 75], [80, 75]],
            dtype=np.float32,
        )
        yaw = yaw_from_kps(kps)
        self.assertIsNotNone(yaw)
        self.assertGreater(abs(yaw), SAVE_MAX_YAW)

    def test_stamp_adds_a_time_bar(self):
        import numpy as np

        from camera.services.captures import stamp_face

        crop = np.zeros((40, 50, 3), dtype=np.uint8)
        stamped = stamp_face(crop)
        self.assertGreater(stamped.shape[0], crop.shape[0])
        self.assertGreaterEqual(stamped.shape[1], crop.shape[1])


class FaceCaptureTests(TestCase):
    def _vector(self, index: int):
        import numpy as np

        vector = np.zeros(512, dtype=np.float32)
        vector[index] = 1
        return vector

    def _crop(self):
        import numpy as np

        image = np.zeros((80, 96, 3), dtype=np.uint8)
        image[:, :] = (40, 90, 140)
        return image

    def test_similar_face_updates_the_same_card(self):
        from camera.models import FaceCapture
        from camera.services.captures import save_visit

        first = save_visit(1, 4, self._vector(0), self._crop(), 0.8, 1.0, None, "", None, None)
        second = save_visit(1, 9, self._vector(0), self._crop(), 0.9, 1.4, None, "", None, None)
        self.assertEqual(first, second)
        self.assertEqual(FaceCapture.objects.count(), 1)
        saved = FaceCapture.objects.get()
        self.assertGreater(saved.quality, 1.2)
        self.assertEqual(saved.camera_number, 1)

    def test_different_face_is_a_new_card(self):
        from camera.models import FaceCapture
        from camera.services.captures import save_visit

        save_visit(1, 4, self._vector(0), self._crop(), 0.8, 1.0, None, "", None, None)
        save_visit(1, 5, self._vector(3), self._crop(), 0.8, 1.0, None, "", None, None)
        self.assertEqual(FaceCapture.objects.count(), 2)

    def test_classify_puts_the_face_on_a_list(self):
        import cv2
        import numpy as np
        from django.core.files.base import ContentFile

        from camera.models import FaceCapture, Person
        from camera.services.faces import to_bytes

        image = np.zeros((48, 48, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        vector = self._vector(0)
        capture = FaceCapture(
            camera_number=2,
            track_id=8,
            embedding=to_bytes(vector),
            det_score=0.9,
            quality=1.1,
        )
        capture.face_crop.save("face.jpg", ContentFile(encoded.tobytes()), save=True)
        twin = FaceCapture(
            camera_number=2,
            track_id=9,
            embedding=to_bytes(vector),
            det_score=0.88,
            quality=1.0,
        )
        twin.face_crop.save("twin.jpg", ContentFile(encoded.tobytes()), save=True)
        response = self.client.post(
            f"/recognition/{capture.pk}/list/",
            {"name": "Ashraf", "list_status": "blacklist"},
        )
        self.assertEqual(response.status_code, 302)
        person = Person.objects.get()
        self.assertEqual(person.name, "Ashraf")
        self.assertEqual(person.list_status, Person.BLACKLIST)
        capture.refresh_from_db()
        twin.refresh_from_db()
        self.assertEqual(capture.person_id, person.pk)
        self.assertEqual(twin.person_id, person.pk)
        self.assertEqual(twin.matched_name, "Ashraf")
        self.assertEqual(person.samples.count(), 1)
        page = self.client.get("/recognition/")
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "Ashraf")
        self.assertContains(page, "blacklist")

    def test_enroll_blacklist_can_switch_to_whitelist(self):
        import cv2
        import numpy as np
        from django.core.files.uploadedfile import SimpleUploadedFile
        from unittest.mock import patch

        from camera.models import Person

        image = np.zeros((48, 48, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        upload = SimpleUploadedFile("ashraf.jpg", encoded.tobytes(), content_type="image/jpeg")
        with patch("camera.views.embed_image", return_value=self._vector(0)):
            response = self.client.post(
                "/people/",
                {"name": "Ashraf", "list_status": "blacklist", "photos": upload},
            )
        self.assertEqual(response.status_code, 302)
        person = Person.objects.get()
        self.assertEqual(person.list_status, Person.BLACKLIST)
        self.assertEqual(person.samples.count(), 1)
        response = self.client.post(f"/people/{person.pk}/list/", {"list_status": "whitelist"})
        self.assertEqual(response.status_code, 302)
        person.refresh_from_db()
        self.assertEqual(person.list_status, Person.WHITELIST)

    def test_recognition_adds_a_second_sample(self):
        import cv2
        import numpy as np
        from django.core.files.base import ContentFile

        from camera.models import FaceCapture, Person
        from camera.services.faces import to_bytes
        from camera.services.gallery import add_sample

        image = np.zeros((48, 48, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        jpeg = encoded.tobytes()
        front = self._vector(0)
        person = Person(name="Ashraf", list_status=Person.WHITELIST, embedding=to_bytes(front))
        person.photo.save("cover.jpg", ContentFile(jpeg), save=True)
        self.assertTrue(add_sample(person, jpeg, front, "front.jpg"))
        side = np.zeros(512, dtype=np.float32)
        side[0] = 0.6
        side[1] = 0.8
        side /= np.linalg.norm(side)
        capture = FaceCapture(
            camera_number=1,
            track_id=3,
            embedding=to_bytes(side),
            det_score=0.9,
            quality=1.2,
        )
        capture.face_crop.save("side.jpg", ContentFile(jpeg), save=True)
        response = self.client.post(
            f"/recognition/{capture.pk}/list/",
            {"name": "Ashraf", "list_status": "blacklist"},
        )
        self.assertEqual(response.status_code, 302)
        person.refresh_from_db()
        self.assertEqual(person.list_status, Person.BLACKLIST)
        self.assertEqual(person.samples.count(), 2)
        first = person.samples.order_by("created_at").first()
        stored = np.frombuffer(bytes(first.embedding), dtype=np.float32)
        self.assertGreater(float(np.dot(stored, front)), 0.99)

    def test_snap_adds_to_existing_profile_without_changing_list(self):
        import cv2
        import numpy as np
        from django.core.files.base import ContentFile

        from camera.models import FaceCapture, Person
        from camera.services.faces import to_bytes
        from camera.services.gallery import add_sample

        image = np.zeros((48, 48, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        jpeg = encoded.tobytes()
        front = self._vector(0)
        person = Person(name="Ashraf", list_status=Person.WHITELIST, embedding=to_bytes(front))
        person.photo.save("cover.jpg", ContentFile(jpeg), save=True)
        self.assertTrue(add_sample(person, jpeg, front, "front.jpg"))
        side = np.zeros(512, dtype=np.float32)
        side[0] = 0.6
        side[1] = 0.8
        side /= np.linalg.norm(side)
        capture = FaceCapture(
            camera_number=1,
            track_id=4,
            embedding=to_bytes(side),
            det_score=0.91,
            quality=1.1,
        )
        capture.face_crop.save("snap.jpg", ContentFile(jpeg), save=True)
        response = self.client.post(f"/recognition/{capture.pk}/sample/", {"person": person.pk})
        self.assertEqual(response.status_code, 302)
        person.refresh_from_db()
        capture.refresh_from_db()
        self.assertEqual(person.list_status, Person.WHITELIST)
        self.assertEqual(person.samples.count(), 2)
        self.assertEqual(capture.person_id, person.pk)
        self.assertEqual(capture.matched_name, "Ashraf")
