import math
import tempfile
import unittest
import pytest

import pandas as pd

import ray

from ray_shuffling_data_loader.data_generation import generate_data
from ray_shuffling_data_loader.shuffle import shuffle_map, shuffle_reduce


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
            num_reducers=num_reducers,
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

    def testShuffleReduce(self):
        num_reducers = 4
        num_shufflers = 2

        reducer_parts = shuffle_map.remote(
            filename=self.filenames[0],
            num_reducers=num_reducers,
            stats_collector=None,
            epoch=0)

        fetched_parts = ray.get(reducer_parts)

        # We cannot get the original references here, so we just push
        # to the object store again as a workaround
        fetched_refs = [ray.put(part) for part in fetched_parts]

        parts_per_shuffler = num_reducers // num_shufflers
        for i in range(num_shufflers):
            unshuffled_refs = fetched_refs[(i * parts_per_shuffler):(
                i + 1 * parts_per_shuffler)]
            unshuffled_parts = fetched_parts[(i * parts_per_shuffler):(
                i + 1 * parts_per_shuffler)]

            shuffled = ray.get(
                shuffle_reduce.remote(
                    0,
                    None,
                    0,
                    *unshuffled_refs,
                ))

            unshuffled = pd.concat(unshuffled_parts, copy=False)

            assert len(unshuffled) == len(shuffled), \
                "Length mismatch between unshuffled and shuffled parts"

            assert set(unshuffled) == set(shuffled), \
                "Key mismatch between unshuffled and shuffled parts"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
