import math
import tempfile
import unittest
from collections import defaultdict

import pytest

import pandas as pd

import ray

from ray_shuffling_data_loader.data_generation import generate_data
from ray_shuffling_data_loader.shuffle import shuffle_map, shuffle_reduce, \
    BatchConsumer, shuffle


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

    def testShuffleEndToEnd(self):
        class EndToEndConsumer(BatchConsumer):
            def __init__(self):
                self.rank_epoch_batches = defaultdict(dict)

            def consume(self, rank, epoch, batches):
                self.rank_epoch_batches[rank][epoch] = ray.get(batches)

            def producer_done(self, rank, epoch):
                pass

            def wait_until_ready(self, epoch):
                return True

            def wait_until_all_epochs_done(self):
                return True

        consumer = EndToEndConsumer()
        num_epochs = 2
        num_reducers = 8
        num_trainers = 4

        shuffle(
            self.filenames,
            batch_consumer=consumer,
            num_epochs=num_epochs,
            num_reducers=num_reducers,
            num_trainers=num_trainers)

        assert len(consumer.rank_epoch_batches) == num_trainers, \
            "Trainer count mismatch"

        assert all(len(consumer.rank_epoch_batches[t]) == num_epochs
                   for t in consumer.rank_epoch_batches), \
            "Epoch count mismatch"

        for tid, epoch_batches in consumer.rank_epoch_batches.items():
            for i in range(len(epoch_batches) - 1):
                assert len(epoch_batches[i]) == len(
                    epoch_batches[+1]) == num_epochs, \
                    "Length mismatch in epoch batches"

                df1 = pd.concat(epoch_batches[i], copy=False)
                df2 = pd.concat(epoch_batches[i], copy=False)

                keys1 = df1["key"].to_numpy()
                keys2 = df2["key"].to_numpy()

                set1 = set(keys1)
                set2 = set(keys2)

                assert len(set1) == len(keys1), \
                    "Keys in dataset are not distinct."

                assert len(set2) == len(keys2), \
                    "Keys in dataset are not distinct."

                assert set1 == set2, \
                    "Shuffled key sets are not equal."


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
