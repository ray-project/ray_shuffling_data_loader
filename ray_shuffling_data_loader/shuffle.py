import timeit
import threading
from typing import Callable, List, Iterable, Union

import pandas as pd
import numpy as np

import ray
from ray_shuffling_data_loader.stats import (TrialStatsCollector,
                                             collect_store_stats, TrialStats)

#
# In-memory shuffling, loads data from disk once per epoch.
#


def shuffle_with_stats(
        filenames: List[str],
        batch_consumer: Callable[[int, int, Iterable[pd.DataFrame]], None],
        num_epochs: int, num_reducers: int, num_trainers: int,
        max_concurrent_epochs: int,
        utilization_sample_period: float) -> (TrialStats, List):
    """
    Shuffle the provided dataset every epoch.
    """
    stats = None
    store_stats = []
    done_event = threading.Event()
    store_stats_collector_thread = threading.Thread(
        target=collect_store_stats,
        args=(store_stats, done_event, utilization_sample_period))
    try:
        store_stats_collector_thread.start()

        print(f"Doing {num_epochs} epochs of shuffling.")

        stats = shuffle(
            filenames,
            batch_consumer,
            num_epochs,
            num_reducers,
            num_trainers,
            max_concurrent_epochs,
            collect_stats=True)
    finally:
        # Signal store stats collector thread that we're done, join the
        # thread.
        done_event.set()
        store_stats_collector_thread.join()

    return stats, store_stats


def shuffle_no_stats(
        filenames: List[str],
        batch_consumer: Callable[[int, int, Iterable[pd.DataFrame]], None],
        num_epochs: int, num_reducers: int, num_trainers: int,
        max_concurrent_epochs: int,
        utilization_sample_period: float) -> (float, None):
    """
    Shuffle the provided dataset every epoch.
    """
    print(f"Doing {num_epochs} epochs of shuffling.")
    duration = shuffle(
        filenames,
        batch_consumer,
        num_epochs,
        num_reducers,
        num_trainers,
        max_concurrent_epochs,
        collect_stats=False)
    return duration, None


def shuffle(filenames: List[str],
            batch_consumer: Callable[[int, int, Iterable[pd.DataFrame]], None],
            num_epochs: int,
            num_reducers: int,
            num_trainers: int,
            max_concurrent_epochs: int,
            collect_stats: bool = True) -> Union[TrialStats, float]:
    if collect_stats:
        stats_collector = TrialStatsCollector.remote(
            num_epochs, len(filenames), num_reducers, num_trainers)
    else:
        stats_collector = None

    start = timeit.default_timer()

    # A list containing reducer output refs for all in-progress epochs.
    in_progress = []
    # We wait for reducer outputs in num_trainers batches given that trainers
    # will consume the reducer outputs in lockstep for a training step, so
    # num_trainers reducer outputs should be released at ~the same time.
    # TODO(Clark): Tweak this heuristic.
    wait_batch = num_trainers
    num_done = 0
    for epoch_idx in range(num_epochs):
        # Throttle epoch pipelining.
        # Get the number of epochs currently in progress.
        num_in_progress_epochs = len(in_progress) // num_reducers
        # Get the number of epochs whose finishing we need to wait for before
        # starting another epoch.
        epochs_to_wait_for = 1 + num_in_progress_epochs - max_concurrent_epochs
        if epochs_to_wait_for > 0:
            # Convert the number of epochs we need to wait for to the number of
            # reducers that we need to wait for.
            reducers_to_wait_for = epochs_to_wait_for * num_reducers
            print(f"Throttling on epoch {epoch_idx}, waiting for "
                  f"{epochs_to_wait_for} epochs, {num_in_progress_epochs} in "
                  "progress.")
            # We wait on the reducers from the first epochs_to_wait_for epochs,
            # ensuring that we give earlier straggler epochs all of the
            # resources they need to finish since epochs are consumed in order.
            refs_to_wait_for = in_progress[:reducers_to_wait_for]
            in_progress = in_progress[reducers_to_wait_for:]
            start_throttle = timeit.default_timer()
            # While throttling, we wait for these refs in num_trainers batches
            # in order to more aggressively free the associated reducer objects
            # from the object store.
            while refs_to_wait_for:
                new_done, refs_to_wait_for = ray.wait(
                    refs_to_wait_for,
                    num_returns=wait_batch,
                    fetch_local=False)
                num_done += wait_batch
                del new_done
            time = timeit.default_timer() - start
            throughput = num_done / time
            print(f"Throughput after throttle: {throughput:.2f} reducer"
                  " chunks/sec")
            if stats_collector is not None:
                stats_collector.epoch_throttle_done.remote(
                    epoch_idx,
                    timeit.default_timer() - start_throttle)

        epoch_reducers = shuffle_epoch(epoch_idx, filenames, batch_consumer,
                                       num_reducers, num_trainers, start,
                                       stats_collector)
        in_progress.extend(epoch_reducers)

    # Block until all epochs are done.
    while in_progress:
        new_done, in_progress = ray.wait(
            in_progress, num_returns=wait_batch, fetch_local=False)
        del new_done

    end = timeit.default_timer()

    if stats_collector is not None:
        stats_collector.trial_done.remote(end - start)

        return ray.get(stats_collector.get_stats.remote())
    else:
        return end - start


def shuffle_epoch(
        epoch: int, filenames: List[str],
        batch_consumer: Callable[[int, int, Iterable[pd.DataFrame]], None],
        num_reducers: int, num_trainers: int, trial_start: float,
        stats_collector: Union[TrialStatsCollector, None]) -> None:
    if stats_collector is not None:
        stats_collector.epoch_start.remote(epoch)
    reducers_partitions = []
    for filename in filenames:
        file_reducer_parts = shuffle_map.options(
            num_returns=num_reducers).remote(
                filename, num_reducers, stats_collector, epoch)
        if not isinstance(file_reducer_parts, list):
            file_reducer_parts = [file_reducer_parts]
        reducers_partitions.append(file_reducer_parts)

    shuffled = []
    for reducer_idx, reducer_partitions in enumerate(
            zip(*reducers_partitions)):
        consumer_batches = shuffle_reduce.remote(
            reducer_idx, stats_collector, epoch, *reducer_partitions)
        shuffled.append(consumer_batches)
    for trainer_idx, batches in enumerate(
            np.array_split(shuffled, num_trainers)):
        consume(trainer_idx, batch_consumer, trial_start, stats_collector,
                epoch, list(batches))
        # Signal to all batch consumers that we're done producing batches for
        # this epoch.
        batch_consumer(trainer_idx, epoch, None)
    return shuffled


@ray.remote
def shuffle_map(filename: str, num_reducers: int,
                stats_collector: Union[TrialStatsCollector, None],
                epoch: int) -> List[List[ray.ObjectRef]]:
    if stats_collector is not None:
        stats_collector.map_start.remote(epoch)
    start = timeit.default_timer()
    # Load file.
    rows = pd.read_parquet(filename)
    assert len(rows) > num_reducers
    end_read = timeit.default_timer()

    # Create random reducer assignment.
    reducer_assignment = np.random.randint(num_reducers, size=len(rows))
    # Partition the rows into a partition per reducer.
    reducer_parts = []
    for reducer_idx in range(num_reducers):
        reducer_part = rows[reducer_assignment == reducer_idx]
        reducer_parts.append(reducer_part)
    if len(reducer_parts) == 1:
        reducer_parts = reducer_parts[0]
    duration = timeit.default_timer() - start
    read_duration = end_read - start
    if stats_collector is not None:
        stats_collector.map_done.remote(epoch, duration, read_duration)
    return reducer_parts


#
# Shared shuffle stages.
#


@ray.remote
def shuffle_reduce(reduce_index: int,
                   stats_collector: Union[TrialStatsCollector, None],
                   epoch: int, *chunks: pd.DataFrame) -> List[pd.DataFrame]:
    if stats_collector is not None:
        stats_collector.reduce_start.remote(epoch)
    start = timeit.default_timer()
    # Concatenate chunks from all mapper partitions.
    batch = pd.concat(chunks)
    # Shuffle the batch.
    batch = batch.sample(frac=1)
    if len(batch) == 1:
        batch = batch[0]
    duration = timeit.default_timer() - start
    if stats_collector is not None:
        stats_collector.reduce_done.remote(epoch, duration)
    return batch


def consume(trainer_idx: int,
            batch_consumer: Callable[[int, int, Iterable[pd.DataFrame]], None],
            trial_start: float,
            stats_collector: Union[TrialStatsCollector, None], epoch: int,
            batches: List[ray.ObjectRef]) -> None:
    if stats_collector is not None:
        stats_collector.consume_start.remote(epoch)
    start = timeit.default_timer()
    trial_time_to_consume = start - trial_start
    batch_consumer(trainer_idx, epoch, batches)
    end = timeit.default_timer()
    duration = end - start
    if stats_collector is not None:
        stats_collector.consume_done.remote(epoch, duration,
                                            trial_time_to_consume)
