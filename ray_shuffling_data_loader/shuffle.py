import timeit
from typing import List, Union

import pandas as pd
import numpy as np

import ray
from ray_shuffling_data_loader.stats import (TrialStatsCollector, TrialStats)


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


def shuffle(filenames: List[str],
            batch_consumer: BatchConsumer,
            num_epochs: int,
            num_reducers: int,
            num_trainers: int,
            stats_collector: Union[TrialStatsCollector, None] = None,
            ) -> Union[TrialStats, float]:
    """
    Shuffle the provided dataset every epoch.

    Args:
        filenames (str): Paths to input Parquet files.
        batch_consumer (BatchConsumer): Consumer of shuffle outputs.
        num_epochs (int): Number of training epochs.
        num_reducers (int): The number of shuffler reducers.
        num_trainers (int): Number of trainer workers.
        stats_collector(Optional[TrialStatsCollector]): Shuffle stats
            collector.
    """
    start = timeit.default_timer()
    for epoch_idx in range(num_epochs):
        # Wait until consumer is ready for another epoch shuffle to start.
        batch_consumer.wait_until_ready(epoch_idx)

        shuffle_epoch(
            epoch_idx, filenames, batch_consumer, num_reducers, num_trainers,
            stats_collector)

    batch_consumer.wait_until_all_epochs_done()
    end = timeit.default_timer()
    duration = end - start

    if stats_collector is not None:
        stats_collector.trial_done.remote(duration)

    return duration


def shuffle_epoch(
        epoch: int, filenames: List[str],
        batch_consumer: BatchConsumer,
        num_reducers: int, num_trainers: int,
        stats_collector: Union[TrialStatsCollector, None] = None) -> None:
    """
    Shuffle the provided dataset for the specified epoch.

    Args:
        epoch (int): Epoch for which we are shuffling.
        filenames (str): Paths to input Parquet files.
        batch_consumer (BatchConsumer): Consumer of shuffle outputs.
        num_reducers (int): The number of shuffler reducers.
        num_trainers (int): Number of trainer workers.
        stats_collector(Optional[TrialStatsCollector]): Shuffle stats
            collector.
    """
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
        consume(rank, batch_consumer, epoch, list(batches))


@ray.remote
def shuffle_map(filename: str, num_reducers: int,
                stats_collector: Union[TrialStatsCollector, None],
                epoch: int) -> List[List[ray.ObjectRef]]:
    """
    Map (data loading and row selection) stage of the shuffle.

    Args:
        filename (str): Path to input Parquet file.
        num_reducers (int): The number of shuffler reducers.
        stats_collector(Optional[TrialStatsCollector]): Shuffle stats
            collector.
        epoch (int): Epoch for which we are shuffling.

    Returns:
        num_reducers partitions, each randomly sampled (without replacement)
        from rows in provided Parquet file.
    """
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
    """
    Reduce (combine and shuffle) stage of the shuffle.

    Args:
        reduce_idx (int): The index (ID) of this reducer.
        stats_collector(Optional[TrialStatsCollector]): Shuffle stats
            collector.
        epoch (int): Epoch for which we are shuffling.
        *chunks (pd.DataFrame): DataFrame partition, one from each mapper.

    Returns:
        A concatenation and full shuffle of all provided mapper partitions.
    """
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


def consume(
        rank: int, batch_consumer: BatchConsumer, epoch: int,
        batches: List[ray.ObjectRef]) -> None:
    """
    Consume the provided batches. This is the sink of the shuffle.

    Args:
        rank (int): The rank (ID) of the consumer.
        batch_consumer (BatchConsumer): The actual consumer of the shuffle
            outputs.
        epoch (int): Epoch for which we're shuffling.
        batches (List[ray.ObjectRef]): The shuffle outputs from one or more
            reducer.
    """
    batch_consumer.consume(rank, epoch, batches)
    # Signal to batch consumer that we're done producing batches for this
    # epoch.
    batch_consumer.producer_done(rank, epoch)
