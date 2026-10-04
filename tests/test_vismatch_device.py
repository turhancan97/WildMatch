"""The Vismatch device identity that enters the feature-cache key and the run manifest."""

import inspect
import unittest
from unittest import mock

import torch

from wildmatch.matchers import vismatch


class DeviceIdentityTests(unittest.TestCase):
    def test_cpu(self):
        self.assertEqual(vismatch.device_identity(torch.device("cpu")), "cpu")

    def test_cuda_names_the_gpu_model_and_capability(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)),
            mock.patch.object(torch.cuda, "get_device_name", return_value="NVIDIA GeForce RTX 4090"),
        ):
            self.assertEqual(vismatch.device_identity(torch.device("cuda")), "cuda:NVIDIA GeForce RTX 4090:sm89")

    def test_cache_tag_has_the_device_and_no_checkpoint_path(self):
        source = inspect.getsource(vismatch.run_vismatch_benchmark)
        self.assertIn("device={device_identity(device)}", source)
        self.assertNotIn("checkpoint_path={requested_checkpoint_path}", source)


if __name__ == "__main__":
    unittest.main()
