"""A failing log write must not deadlock a sweep task (disk quota exceeded, 2026-10-04)."""

import io
import subprocess
import sys
import threading
import unittest

from wildmatch.sweep.runner import _pump


class FailingSink:
    name = "full-disk.log"

    def write(self, chunk):
        raise OSError(122, "Disk quota exceeded")

    def flush(self):
        pass


class SweepPumpTests(unittest.TestCase):
    def test_child_finishes_when_a_log_target_fails(self):
        size = 2 * 1024 * 1024  # far more than a pipe buffer holds
        child = subprocess.Popen(
            [sys.executable, "-c", f"import sys; sys.stdout.buffer.write(b'x' * {size})"], stdout=subprocess.PIPE
        )
        good, errors = io.BytesIO(), []
        thread = threading.Thread(target=_pump, args=(child.stdout, (FailingSink(), good), threading.Lock(), errors))
        thread.start()
        thread.join(30)
        self.assertFalse(thread.is_alive(), "pump stopped draining the pipe")
        self.assertEqual(child.wait(30), 0)
        self.assertEqual(len(good.getvalue()), size)
        self.assertEqual(len(errors), 1)
        self.assertIn("Disk quota exceeded", errors[0])


if __name__ == "__main__":
    unittest.main()
