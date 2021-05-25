import argparse
import glob
import os
import timeit

import numpy as np

import ray
from ray_shuffling_data_loader.shuffle import (
    shuffle_with_stats, shuffle_no_stats)
from ray_shuffling_data_loader.stats import (
    process_stats, human_readable_size)

from ray_shuffling_data_loader.data_generation import generate_data

from ray.util.placement_group import placement_group
import time

DEFAULT_DATA_DIR = "/mnt/disk0/benchmark_scratch"
DEFAULT_STATS_DIR = "./results"

DEFAULT_UTILIZATION_SAMPLE_PERIOD = 5.0


@ray.remote(num_cpus=0)
class Sink:
    def __init__(self, consumer_idx):
        self.timestamps = {}
        self.consumer_idx = consumer_idx

    def consume(self, batch):
        self.timestamps[time.time()] = len(batch)
        print(self.consumer_idx, self.timestamps)

    def ping(self):
        self.timestamps[time.time()] = 0

    def collect_stats(self):
        return self.timestamps

def dummy_batch_consumer(consumer_idx, epoch, batches):
    pass


def run_trials(num_epochs,
               filenames,
               num_reducers,
               num_trainers,
               max_concurrent_epochs,
               utilization_sample_period,
               collect_stats=True,
               num_trials=None,
               trials_timeout=None):
    """
    Run shuffling trials.
    """
    print("Using from-memory shuffler.")
    if collect_stats:
        shuffle = shuffle_with_stats
    else:
        shuffle = shuffle_no_stats
    all_stats = []

    pg = placement_group([{'resources': 1} for _ in range(num_trainers)], strategy="SPREAD")
    ray.get(pg.ready())
    sinks = [Sink.options(placement_group=pg).remote(i) for i in range(num_trainers)]
    ray.get([sink.ping.remote() for sink in sinks])
    def batch_consumer(i, epoch, batches):
        if batches:
            for batch in batches:
                sinks[i].consume.remote(batch)

    if num_trials is not None:
        for trial in range(num_trials):
            print(f"Starting trial {trial}.")
            stats, store_stats = shuffle(
                filenames, batch_consumer, num_epochs, num_reducers,
                num_trainers, max_concurrent_epochs, utilization_sample_period)
            duration = stats.duration if collect_stats else stats
            print(f"Trial {trial} done after {duration} seconds.")
            all_stats.append((stats, store_stats))
    elif trials_timeout is not None:
        start = timeit.default_timer()
        trial = 0
        while timeit.default_timer() - start < trials_timeout:
            print(f"Starting trial {trial}.")
            stats, store_stats = shuffle(
                filenames, batch_consumer, num_epochs, num_reducers,
                num_trainers, max_concurrent_epochs, utilization_sample_period)
            duration = stats.duration if collect_stats else stats
            print(f"Trial {trial} done after {duration} seconds.")
            all_stats.append((stats, store_stats))
            trial += 1
    else:
        raise ValueError(
            "One of num_trials and trials_timeout must be specified")

    with open('output.csv', 'a') as f:
        import csv
        writer = csv.writer(f)
        for i, sink in enumerate(sinks):
            records = ray.get(sink.collect_stats.remote())
            for timestamp, num_records in records.items():
                writer.writerow([i, timestamp, num_records])

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffling data loader")
    parser.add_argument("--num-rows", type=int, default=4 * (10**11))
    parser.add_argument("--num-files", type=int, default=100)
    parser.add_argument("--max-row-group-skew", type=float, default=0.0)
    parser.add_argument("--num-row-groups-per-file", type=int, default=1)
    parser.add_argument("--num-reducers", type=int, default=5)
    parser.add_argument("--num-trainers", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--max-concurrent-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--trials-timeout", type=int, default=None)
    parser.add_argument(
        "--utilization-sample-period",
        type=float,
        default=DEFAULT_UTILIZATION_SAMPLE_PERIOD)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--stats-dir", type=str, default=DEFAULT_STATS_DIR)
    parser.add_argument("--clear-old-data", action="store_true")
    parser.add_argument("--use-old-data", action="store_true")
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("--no-epoch-stats", action="store_true")
    parser.add_argument("--no-consume-stats", action="store_true")
    parser.add_argument("--overwrite-stats", action="store_true")
    parser.add_argument("--unique-stats", action="store_true")
    args = parser.parse_args()

    if args.num_row_groups_per_file < 1:
        raise ValueError("Must have at least one row group per file.")

    num_trials = args.num_trials
    trials_timeout = args.trials_timeout
    if num_trials is not None and trials_timeout is not None:
        raise ValueError(
            "Only one of --num-trials and --trials-timeout should be "
            "specified.")

    if num_trials is None and trials_timeout is None:
        num_trials = 3

    if args.clear_old_data and args.use_old_data:
        raise ValueError(
            "Only one of --clear-old-data and --use-old-data should be "
            "specified.")

    data_dir = args.data_dir
    if args.clear_old_data:
        print(f"Clearing old data from {data_dir}.")
        files = glob.glob(os.path.join(data_dir, "*.parquet.snappy"))
        for f in files:
            os.remove(f)

    if args.cluster:
        print("Connecting to an existing Ray cluster.")
        ray.init(address="auto")
    else:
        print("Starting a new local Ray cluster.")
        ray.init()

    num_rows = args.num_rows
    num_row_groups_per_file = args.num_row_groups_per_file
    num_files = args.num_files
    max_row_group_skew = args.max_row_group_skew
    if not args.use_old_data:
        print(f"Generating {num_rows} rows over {num_files} files, with "
              f"{num_row_groups_per_file} row groups per file and at most "
              f"{100 * max_row_group_skew:.1f}% row group skew.")
        filenames, num_bytes = generate_data(num_rows, num_files,
                                             num_row_groups_per_file,
                                             max_row_group_skew, data_dir)
        print(f"Generated {len(filenames)} files containing {num_rows} rows "
              f"with {num_row_groups_per_file} row groups per file, totalling "
              f"{human_readable_size(num_bytes)}.")
    else:
        filenames = [
            os.path.join(data_dir, f"input_data_{file_index}.parquet.snappy")
            for file_index in range(num_files)
        ]
        print("Not generating input data, using existing data instead.")

    num_reducers = args.num_reducers
    num_trainers = args.num_trainers
    batch_size = args.batch_size

    num_epochs = args.num_epochs
    max_concurrent_epochs = args.max_concurrent_epochs
    if max_concurrent_epochs is None or max_concurrent_epochs > num_epochs:
        max_concurrent_epochs = num_epochs
    assert max_concurrent_epochs > 0

    utilization_sample_period = args.utilization_sample_period

    # TODO(Clark): Add warmup trials.

    print("\nRunning real trials.")
    if num_trials is not None:
        print(f"Running {num_trials} shuffle trials with {num_epochs} epochs, "
              f"{num_reducers} reducers, {num_trainers} trainers, and a batch "
              f"size of {batch_size} over {num_rows} rows.")
    else:
        print(f"Running {trials_timeout} seconds of shuffle trials with "
              f"{num_epochs} epochs, {num_reducers} reducers, {num_trainers} "
              f"trainers, and a batch size of {batch_size} over {num_rows} "
              "rows.")
    print(f"Shuffling will be pipelined with at most "
          f"{max_concurrent_epochs} concurrent epochs.")
    collect_stats = not args.no_stats
    all_stats = run_trials(num_epochs, filenames, num_reducers, num_trainers,
                           max_concurrent_epochs, utilization_sample_period,
                           collect_stats, num_trials, trials_timeout)

    if collect_stats:
        process_stats(all_stats, args.overwrite_stats, args.stats_dir,
                      args.no_epoch_stats, args.unique_stats, num_rows,
                      num_files, num_row_groups_per_file, batch_size,
                      num_reducers, num_trainers, num_epochs,
                      max_concurrent_epochs)
    else:
        print("Shuffle trials done, no detailed stats collected.")
        times, _ = zip(*all_stats)
        mean = np.mean(times)
        std = np.std(times)
        throughput_std = np.std(
            [num_epochs * num_rows / time for time in times])
        batch_throughput_std = np.std(
            [(num_epochs * num_rows / batch_size) / time for time in times])
        print(f"\nMean over {len(times)} trials: {mean:.3f}s +- {std}")
        print(f"Mean throughput over {len(times)} trials: "
              f"{num_epochs * num_rows / mean:.2f} rows/s +- "
              f"{throughput_std:.2f}")
        print(f"Mean batch throughput over {len(times)} trials: "
              f"{(num_epochs * num_rows / batch_size) / mean:.2f} batches/s "
              f"+- {batch_throughput_std:.2f}")
