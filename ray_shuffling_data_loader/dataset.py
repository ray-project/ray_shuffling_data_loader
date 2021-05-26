from typing import List

import pandas as pd
import ray
from ray._private.utils import get_num_cpus
from ray_shuffling_data_loader.shuffle import shuffle, BatchConsumer
from ray_shuffling_data_loader.batch_queue import BatchQueue

BATCHQUEUE_ACTOR_NAME = "BatchQueue"
REDUCER_CLUSTER_CORE_SHARE = 0.6


class ShufflingDataset:
    """
    A shuffling dataset that yields batches upon iteration.

    This dataset will kick off shuffling for max_concurrent_epochs epochs at
    construction time in the master process (rank 0).

    Args:
        filenames (str): Paths to input Parquet files.
        num_epochs (int): Number of training epochs.
        num_trainers (int): Number of trainer workers.
        batch_size (int): Size of the batches that the iterator should yield.
        rank (int): The worker rank of the current process.
        drop_last (Optional[bool]): Whether to drop the last batch if it's
            incomplete (smaller than batch_size). Default is False.
        num_reducers (Optional[int]): The number of shuffler reducers. Default
            is the number of trainers x the number of cores on the master
            (rank 0) worker x 0.6.
        max_concurrent_epochs (Optional[int]): The maximum number of epochs
            whose shuffling stages should execute concurrently. Default is 2.
    """

    def __init__(self,
                 filenames: List[str],
                 num_epochs: int,
                 num_trainers: int,
                 batch_size: int,
                 rank: int,
                 drop_last: bool = False,
                 num_reducers: int = None,
                 max_concurrent_epochs: int = 2,
                 max_batch_queue_size: int = 0):
        if num_reducers is None:
            num_reducers = int(
                num_trainers * get_num_cpus() * REDUCER_CLUSTER_CORE_SHARE)

        self._batch_size = batch_size

        if rank == 0:
            # rank == 0 --> master process
            # Create the batch queue. Trainers will consume GPU batches
            # through this batch queue.
            self._batch_queue = BatchQueue(
                num_epochs,
                num_trainers,
                max_concurrent_epochs,
                max_batch_queue_size,
                name=BATCHQUEUE_ACTOR_NAME,
                connect=False)
            self._consumer = BatchConsumerQueue(self._batch_queue)
            # Wait until actor has been created.
            self._batch_queue.ready()
            # Kick off shuffle.
            # TODO(Clark): Move the shuffle kickoff to an init() method so the
            # user can better control when the shuffling starts?
            self._shuffle_result = ray.remote(shuffle).remote(
                filenames,
                self._consumer,
                num_epochs,
                num_reducers,
                num_trainers,
                collect_stats=False)
        else:
            # rank != 0 --> worker process
            # Connect to the batch queue.
            self._batch_queue = BatchQueue(
                num_epochs,
                num_trainers,
                max_concurrent_epochs,
                max_batch_queue_size,
                name=BATCHQUEUE_ACTOR_NAME,
                connect=True)
            self._shuffle_result = None

        self._num_epochs = num_epochs
        self._num_trainers = num_trainers
        self._rank = rank
        self._epoch = None
        # Used to check that the user is correctly setting the epoch at the
        # beginning of each epoch.
        self._last_epoch = None

        self._drop_last = drop_last

    def set_epoch(self, epoch):
        """
        Set the current training epoch. This should be called before
        constructing the iterator on this dataset (e.g. before the
        enumerate(train_loader) call).

        Args:
            epoch (int) The epoch number for the training epoch that is about
                to start.
        """
        self._epoch = epoch

    def __iter__(self):
        """
        This iterator yields GPU batches from the shuffling queue.
        """
        if self._epoch is None or self._epoch == self._last_epoch:
            raise ValueError(
                "You must set the epoch on this dataset via set_epoch()"
                "at the beginning of each epoch, before iterating over this "
                "dataset (e.g. via enumerate(ds)).")

        # Batch leftover buffer.
        df_buffer = None
        is_done = False
        pending = []
        outstanding_items = 0
        while True:
            if not pending:
                if is_done:
                    # No more items in pending and is_done flag is set, so
                    # there are no more queue items to consume.
                    break
                if outstanding_items > 0:
                    # Signal to the queue that we're done processing these GPU
                    # batches.
                    self._batch_queue.task_done(
                        self._rank, self._epoch, outstanding_items)
                # Get a batch of batches from the queue.
                for batch in self._batch_queue.get_batch(
                        self._rank, self._epoch):
                    if batch is None:
                        # Set done flag but don't break yet, since we might
                        # still have more items in pending to consume.
                        is_done = True
                    else:
                        pending.append(batch)
                outstanding_items = len(pending) + is_done
                if not pending:
                    # This can happen if the only item in the queue was the
                    # done signal, None.
                    continue

            # This ray.wait() also serves as our prefetching method, where pull
            # requests for all objects in pending will be made at this time.
            ready, pending = ray.wait(
                pending, num_returns=1, fetch_local=True)
            # Fetch the underlying dataframe.
            df = ray.get(ready[0])

            # Don't hold on to object refs for any longer than we need to.
            del ready

            # Get first-slice offset into current dataframe.
            df_buffer_len = len(df_buffer) if df_buffer is not None else 0
            offset = self._batch_size - df_buffer_len
            # If we already have a leftover batch, concatenate it with a
            # front-slice of the current dataframe, attempting to create a
            # full-sized batch.
            # If we don't already have a leftover batch, we consume the first
            # batch in the current dataframe here.
            df_buffer = pd.concat([df_buffer, df[:offset]])
            # If we have a full-sized batch, yield it. Otherwise, hang on to
            # it and yield it in a future round, once we have a full batch.
            if len(df_buffer) == self._batch_size:
                yield df_buffer
                df_buffer = None
            # Yield batches from the current dataframe.
            pos = offset  # Fallback if offset > len(df).
            for pos in range(offset,
                             len(df) - self._batch_size + 1, self._batch_size):
                yield df[pos:pos + self._batch_size]
            # If leftover (incomplete) batch, save for later.
            pos += self._batch_size
            if pos < len(df):
                df_buffer = df[pos:]
        # Yield leftover (incomplete) batch if we're not dropping incomplete
        # batches.
        if df_buffer is not None and not self._drop_last:
            yield df_buffer
        # Signal to the queue that we're done processing the last GPU batch.
        self._batch_queue.task_done(self._rank, self._epoch, outstanding_items)
        self._last_epoch = self._epoch
        if (self._epoch == self._num_epochs - 1
                and self._shuffle_result is not None):
            ray.get(self._shuffle_result)


class BatchConsumerQueue(BatchConsumer):
    def __init__(self, batch_queue: BatchQueue):
        self._batch_queue = batch_queue

    def consume(self, rank: int, epoch: int, batches: List[ray.ObjectRef]):
        self._batch_queue.put_batch(rank, epoch, batches)

    def producer_done(self, rank: int, epoch: int):
        self._batch_queue.producer_done(rank, epoch)

    def wait_until_ready(self, epoch: int):
        self._batch_queue.new_epoch(epoch)

    def wait_until_all_epochs_done(self):
        self._batch_queue.wait_until_all_epochs_done()


if __name__ == "__main__":
    from ray_shuffling_data_loader.stats import human_readable_size
    from ray_shuffling_data_loader.data_generation import generate_data
    print("Starting Ray...")
    # FIXME(Clark): We're setting the idle worker killing time very high in
    # order to work around an idle working killing bug that's causing objects
    # to be lost. We should fix this.
    ray.init(_system_config={"idle_worker_killing_time_threshold_ms": 10**6})
    num_rows = 10**6
    num_files = 10
    num_row_groups_per_file = 1
    max_row_group_skew = 0.0
    data_dir = "data"
    print(f"Generating {num_rows} rows over {num_files} files, with "
          f"{num_row_groups_per_file} row groups per file and at most "
          f"{100 * max_row_group_skew:.1f}% row group skew.")
    filenames, num_bytes = generate_data(num_rows, num_files,
                                         num_row_groups_per_file,
                                         max_row_group_skew, data_dir)
    print(f"Generated {len(filenames)} files containing {num_rows} rows "
          f"with {num_row_groups_per_file} row groups per file, totalling "
          f"{human_readable_size(num_bytes)}.")
    num_epochs = 4
    num_trainers = 1
    batch_size = 20000
    rank = 0
    num_reducers = 8
    print(f"Creating shuffling dataset with {batch_size} batch size, "
          f"{num_epochs} epochs, {num_reducers} reducers, and {num_trainers} "
          "trainers.")
    print(f"Should consume {num_rows // batch_size} batches.")
    ds = ShufflingDataset(
        filenames,
        num_epochs,
        num_trainers,
        batch_size,
        rank,
        num_reducers=num_reducers)

    for epoch in range(num_epochs):
        ds.set_epoch(epoch)
        for batch_idx, batch in enumerate(ds):
            print(f"Consuming batch {batch_idx}!")
    print("Done consuming batches.")
