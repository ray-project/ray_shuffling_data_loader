import math
import tempfile
import unittest
import pytest

import ray

from ray_shuffling_data_loader.data_generation import generate_data
from ray_shuffling_data_loader.shuffle import shuffle_map


class DataLoaderShuffleTest(unittest.TestCase):
    """This test suite validates core RayDMatrix functionality."""

    def setUp(self):
        self.num_rows = 10**4
        self.num_files = 1
        self.num_row_groups_per_file = 1
        self.max_row_group_skew = 0.0
        self.data_dir = tempfile.mkdtemp()

        self.filenames, self.num_bytes = generate_data(
            self.num_rows, self.num_files, self.num_row_groups_per_file,
            self.max_row_group_skew, self.data_dir)

    @classmethod
    def setUpClass(cls):
        ray.init(num_cpus=2)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def testShuffleMap(self):
        num_reducers = 4

        # Calculate mean and SD of rows assigned to each reducer
        p = 1 / num_reducers
        mean = self.num_rows * p
        sd = math.sqrt(self.num_rows * p * (1 - p))

        reducer_parts = shuffle_map.remote(
            filename=self.filenames[0],
            num_reducers=4,
            stats_collector=None,
            epoch=0)

        fetched_parts = ray.get(reducer_parts)

        all_keys = []
        for i, part in enumerate(fetched_parts):
            part_keys = part["key"].to_numpy()

            # 3sd = 99.7% chance of passing
            assert mean - 3 * sd < len(part_keys) < mean + 3 * sd, \
                f"Not enough rows in partition {i}"

            assert len(set(part_keys)) == len(part_keys), \
                f"Keys in partition {i} are not distinct"

            all_keys.extend(part_keys)

        assert len(all_keys) == self.num_rows, "Not all rows were returned."

        assert len(set(all_keys)) == len(all_keys), \
            "Keys in full dataset are not distinct."


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
