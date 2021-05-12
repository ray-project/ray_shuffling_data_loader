import timeit
import threading
from typing import List, Union

import pandas as pd
import numpy as np

import ray
from ray_shuffling_data_loader.stats import (TrialStatsCollector,
                                             collect_store_stats, TrialStats)


class BatchConsumer:
    """
    Interface for consumers of the shuffle outputs.
    """
    def consume(self, rank, epoch, batches):
        """
        Consume the provided batches for the given trainer and epoch.
        """
        raise NotImplementedError(
            "Derived classes must implement consume method.")

    def producer_done(self, rank, epoch):
        """
        Signals to the consumer that we're done producing batches for the
        given trainer and epoch.
        """
        raise NotImplementedError(
            "Derived classes must implement producer_done method.")

    def wait_until_ready(self, epoch):
        """
        Returns once the consumer is ready for this epoch to start.
        """
        raise NotImplementedError(
            "Derived classes must implement wait_until_ready method.")

    def wait_until_all_epochs_done(self):
        """
        Returns once all batches for all epochs have been consumed.
        """
        raise NotImplementedError(
            "Derived classes must implement wait_until_done method.")


#
# In-memory shuffling, loads data from disk once per epoch.
#


def shuffle_with_stats(
        filenames: List[str],
        batch_consumer: BatchConsumer,
        num_epochs: int, num_reducers: int, num_trainers: int,
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
            collect_stats=True)
    finally:
        # Signal store stats collector thread that we're done, join the
        # thread.
        done_event.set()
        store_stats_collector_thread.join()

    return stats, store_stats


def shuffle_no_stats(
        filenames: List[str],
        batch_consumer: BatchConsumer,
        num_epochs: int, num_reducers: int, num_trainers: int,
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
        collect_stats=False)
    return duration, None


def shuffle(filenames: List[str],
            batch_consumer: BatchConsumer,
            num_epochs: int,
            num_reducers: int,
            num_trainers: int,
            collect_stats: bool = True) -> Union[TrialStats, float]:
    if collect_stats:
        stats_collector = TrialStatsCollector.remote(
            num_epochs, len(filenames), num_reducers, num_trainers)
    else:
        stats_collector = None

    start = timeit.default_timer()
    for epoch_idx in range(num_epochs):
        # Wait until consumer is ready for another epoch shuffle to start.
        batch_consumer.wait_until_ready(epoch_idx)

        shuffle_epoch(
            epoch_idx, filenames, batch_consumer, num_reducers, num_trainers,
            start, stats_collector)

    batch_consumer.wait_until_all_epochs_done()
    end = timeit.default_timer()

    if stats_collector is not None:
        stats_collector.trial_done.remote(end - start)

        return ray.get(stats_collector.get_stats.remote())
    else:
        return end - start


def shuffle_epoch(
        epoch: int, filenames: List[str],
        batch_consumer: BatchConsumer,
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
    for rank, batches in enumerate(
            np.array_split(shuffled, num_trainers)):
        consume(rank, batch_consumer, trial_start, stats_collector,
                epoch, list(batches))


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


def consume(rank: int,
            batch_consumer: BatchConsumer,
            trial_start: float,
            stats_collector: Union[TrialStatsCollector, None], epoch: int,
            batches: List[ray.ObjectRef]) -> None:
    if stats_collector is not None:
        stats_collector.consume_start.remote(epoch)
    start = timeit.default_timer()
    trial_time_to_consume = start - trial_start
    batch_consumer.consume(rank, epoch, batches)
    # Signal to batch consumer that we're done producing batches for this
    # epoch.
    batch_consumer.producer_done(rank, epoch)
    end = timeit.default_timer()
    duration = end - start
    if stats_collector is not None:
        stats_collector.consume_done.remote(epoch, duration,
                                            trial_time_to_consume)
