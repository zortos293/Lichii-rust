import unittest

from queue_trace import Capture, decode, read_captures


def events(text):
    return [
        (int(at), event, tuple(arg if arg in ("s", "y", "f") else int(arg) for arg in args))
        for line in text.splitlines()
        if line.strip()
        for at, event, *args in (line.split(),)
    ]


def decode_rows(text, dropped=0):
    return list(decode(events(text), dropped, True))


class QueueTraceTest(unittest.TestCase):
    def test_fences_get_only_their_channels_wait_and_service(self):
        rows = decode_rows("""
             0 submit 1 7 1 1
             2 run 1
             4 acquire 2 100 7 0 0
             4 wait 1 s 2 2
             5 stop 1
             6 submit 2 10 1 3
             7 run 2
            10 submit 1 7 2 4
            11 offer 20 100 4 7
            11 latch 2 20 7 1
            13 front 3
            15 stop 2
            16 resume 1 9
            17 run 1
            18 front 1
            20 wait 1 f 1 1
            21 stop 1
            23 fetch_ready 1
            25 resume 1 22
            26 run 1
            27 wait 1 y 5 7
            27 stop 1
            30 sync_ready 0 7
            32 resume 1 28
            32 run 1
            36 front 4
            37 stop 1
            40 publish 20 100 4 7 1
            41 retire 20
        """)
        match = next(row for row in rows if row.get("key") == "s:2")
        self.assertEqual((match["fifo"], match["ready_ns"], match["publication_ns"]), (20, 11, 41))
        by_id = {row["sample"]: row for row in rows if row["type"] == "fence"}
        a, b, c = (by_id[i] for i in (1, 3, 4))
        self.assertEqual((a["semaphore_ns"], a["runnable_ns"], a["front_service_ns"]), (6, 8, 4))
        self.assertEqual(a["runnable_other_ns"], 4)
        self.assertEqual((b["semaphore_ns"], b["runnable_ns"], b["front_service_ns"]), (0, 1, 6))
        self.assertEqual((c["semaphore_ns"], c["fetch_ns"], c["syncpoint_ns"]), (1, 2, 3))
        self.assertEqual(c["wait_ids"], ["s:2", "f:1", "y:5"])
        for sample in (a, b, c):
            time = sum(sample[name] for name in (
                "semaphore_ns", "syncpoint_ns", "fetch_ns", "front_service_ns",
                "runnable_ns", "unattributed_ns",
            ))
            self.assertEqual(time, sample["front_ns"] - sample["submitted_ns"])

    def test_old_fifos_wrong_addresses_and_values_dont_fill_missing_records(self):
        rows = decode_rows("""
             0 submit 1 7 1 1
             1 run 1
             2 acquire 2 100 7 0 0
             2 wait 1 s 2 2
             3 stop 1
             4 offer 19 100 4 7
             5 publish 19 100 4 7 1
             6 retire 19
             8 offer 20 104 4 7
             9 offer 20 100 4 6
            10 latch 2 20 7 1
            11 publish 20 104 4 7 1
            12 publish 20 100 4 6 1
            13 retire 20
            15 resume 1 7
            16 run 1
            17 front 1
            18 stop 1
        """)
        _, wait, fence = rows
        self.assertEqual(wait["source"], "gpu_fifo_partial_record")
        self.assertIsNone(wait["offer_ns"])
        self.assertIsNone(wait["publication_ns"])
        self.assertEqual((fence["quality"], fence["semaphore_ns"], fence["unattributed_ns"]), ("partial", 4, 8))

    def test_written_latch_uses_retirement_not_time_of_first_write(self):
        rows = decode_rows("""
             0 submit 1 7 1 1
             1 run 1
             2 acquire 2 104 7 0 0
             2 wait 1 s 2 2
             3 stop 1
             4 producer 20 8 3
             5 offer 20 100 8 30064771072
             6 publish 20 100 8 30064771072 1
             7 latch 2 20 7 0
             9 retire 20
            15 resume 1 8
            16 run 1
            17 front 1
            18 stop 1
        """)
        _, wait, fence = rows
        self.assertEqual(wait["source"], "gpu_fifo_publication")
        self.assertEqual((wait["publication_ns"], wait["ready_ns"], wait["producer_channel"]), (9, 9, 8))
        self.assertEqual((fence["quality"], fence["semaphore_ns"]), ("complete", 6))

    def test_unobserved_readiness_and_dropped_or_pending_fences_are_explicit(self):
        trace = """
             0 submit 1 7 1 1
             1 run 1
             2 wait 1 s 2 2
             3 stop 1
             4 submit 1 7 2 3
            15 resume 1 10
            16 run 1
            17 front 1
            18 stop 1
        """
        _, wait, ended, pending = decode_rows(trace)
        self.assertEqual(wait["source"], "memory_or_unobserved")
        self.assertEqual((ended["semaphore_ns"], ended["unattributed_ns"]), (7, 5))
        self.assertEqual(pending["quality"], "partial")
        self.assertIsNone(pending["front_ns"])
        status, *fences = decode_rows(trace, dropped=2)
        self.assertEqual(status["dropped"], 2)
        for row in fences:
            self.assertEqual(row["quality"], "dropped")
            self.assertNotIn("runnable_ns", row)

    def test_reads_batched_tracing_fields(self):
        log = [
            "00:00 DEBUG lichii_gpu::queue_trace: GPU queue events \x1b[1mcapture\x1b[0m=1 unix_start_ns=123 data=0,submit,1,3,9,2;5,front,2",
            "00:01 DEBUG lichii_gpu::queue_trace: GPU queue trace status capture=1 unix_start_ns=123 dropped=2 end=true",
        ]
        self.assertEqual(
            read_captures(log),
            {"1": Capture(123, [(0, "submit", (1, 3, 9, 2)), (5, "front", (2,))], 2, True)},
        )

    def test_truncated_event_invalidates_instead_of_guessing(self):
        log = ["GPU queue events capture=1 unix_start_ns=123 data=0,submit,1,3,9,2;5,front"]
        capture = read_captures(log)["1"]
        status, fence = decode(**vars(capture))
        self.assertEqual(status["malformed"], 1)
        self.assertEqual(fence["quality"], "partial")
        self.assertNotIn("runnable_ns", fence)


if __name__ == "__main__":
    unittest.main()
