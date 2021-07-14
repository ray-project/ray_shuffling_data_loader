import pytest
import time

import ray

from ray_shuffling_data_loader.batch_queue import BatchQueue, Empty, Full

from ray.exceptions import GetTimeoutError, RayActorError
from ray.test_utils import wait_for_condition


# Remote helper functions for testing concurrency
@ray.remote
def async_get(queue, **kwargs):
    return queue.get(block=True, **kwargs)


@ray.remote
def async_put(queue, item, **kwargs):
    return queue.put(item=item, block=True, **kwargs)


def test_simple_usage(ray_start_regular_shared):

    q = BatchQueue(num_epochs=1, num_trainers=1, max_concurrent_epochs=1)

    items = list(range(10))

    for item in items:
        q.put(rank=0, epoch=0, item=item)

    for item in items:
        assert item == q.get(rank=0, epoch=0)


def test_get(ray_start_regular_shared):

    q = BatchQueue(num_epochs=1, num_trainers=1, max_concurrent_epochs=1)

    item = 0
    q.put(rank=0, epoch=0, item=item)

    assert q.get(rank=0, epoch=0, block=False) == item

    item = 1
    q.put(rank=0, epoch=0, item=item)
    assert q.get(rank=0, epoch=0, timeout=0.2) == item

    with pytest.raises(ValueError):
        q.get(rank=0, epoch=0, timeout=-1)

    with pytest.raises(Empty):
        q.get_nowait(rank=0, epoch=0)

    with pytest.raises(Empty):
        q.get(rank=0, epoch=0, timeout=0.2)


@pytest.mark.asyncio
async def test_get_async(ray_start_regular_shared):

    q = BatchQueue(num_epochs=1, num_trainers=1, max_concurrent_epochs=1)

    item = 0
    await q.put_async(rank=0, epoch=0, item=item)
    assert await q.get_async(rank=0, epoch=0, block=False) == item

    item = 1
    await q.put_async(rank=0, epoch=0, item=item)
    assert await q.get_async(rank=0, epoch=0, timeout=0.2) == item

    with pytest.raises(ValueError):
        await q.get_async(rank=0, epoch=0, timeout=-1)

    with pytest.raises(Empty):
        await q.get_async(rank=0, epoch=0, block=False)

    with pytest.raises(Empty):
        await q.get_async(rank=0, epoch=0, timeout=0.2)


def test_put(ray_start_regular_shared):

    q = BatchQueue(
        num_epochs=1, num_trainers=1, max_concurrent_epochs=1, maxsize=1)

    item = 0
    q.put(rank=0, epoch=0, item=item, block=False)
    assert q.get(rank=0, epoch=0) == item

    item = 1
    q.put(rank=0, epoch=0, item=item, timeout=0.2)
    assert q.get(rank=0, epoch=0) == item

    with pytest.raises(ValueError):
        q.put(rank=0, epoch=0, item=0, timeout=-1)

    q.put(rank=0, epoch=0, item=0)
    with pytest.raises(Full):
        q.put_nowait(rank=0, epoch=0, item=1)

    with pytest.raises(Full):
        q.put(rank=0, epoch=0, item=1, timeout=0.2)


@pytest.mark.asyncio
async def test_put_async(ray_start_regular_shared):

    q = BatchQueue(
        num_epochs=1, num_trainers=1, max_concurrent_epochs=1, maxsize=1)

    item = 0
    await q.put_async(rank=0, epoch=0, item=item, block=False)
    assert await q.get_async(rank=0, epoch=0) == item

    item = 1
    await q.put_async(rank=0, epoch=0, item=item, timeout=0.2)
    assert await q.get_async(rank=0, epoch=0) == item

    with pytest.raises(ValueError):
        await q.put_async(rank=0, epoch=0, item=0, timeout=-1)

    await q.put_async(rank=0, epoch=0, item=0)
    with pytest.raises(Full):
        await q.put_async(rank=0, epoch=0, item=1, block=False)

    with pytest.raises(Full):
        await q.put_async(rank=0, epoch=0, item=1, timeout=0.2)


def test_concurrent_get(ray_start_regular_shared):
    q = BatchQueue(num_epochs=1, num_trainers=1, max_concurrent_epochs=1)
    future = async_get.remote(queue=q, rank=0, epoch=0)

    with pytest.raises(Empty):
        q.get_nowait(rank=0, epoch=0)

    with pytest.raises(GetTimeoutError):
        ray.get(future, timeout=0.1)  # task not canceled on timeout.

    q.put(rank=0, epoch=0, item=1)
    assert ray.get(future) == 1


def test_concurrent_put(ray_start_regular_shared):
    q = BatchQueue(
        num_epochs=1, num_trainers=1, max_concurrent_epochs=1, maxsize=1)
    q.put(rank=0, epoch=0, item=1)
    future = async_put.remote(q, 2, rank=0, epoch=0)

    with pytest.raises(Full):
        q.put_nowait(rank=0, epoch=0, item=3)

    with pytest.raises(GetTimeoutError):
        ray.get(future, timeout=0.1)  # task not canceled on timeout.

    assert q.get(rank=0, epoch=0) == 1
    assert q.get(rank=0, epoch=0) == 2


def test_batch(ray_start_regular_shared):
    q = BatchQueue(
        num_epochs=1, num_trainers=1, max_concurrent_epochs=1, maxsize=1)

    with pytest.raises(Full):
        q.put_nowait_batch(rank=0, epoch=0, items=[1, 2])

    with pytest.raises(Empty):
        q.get_nowait_batch(rank=0, epoch=0, num_items=1)

    big_q = BatchQueue(
        num_epochs=1, num_trainers=1, max_concurrent_epochs=1, maxsize=100)
    big_q.put_nowait_batch(rank=0, epoch=0, items=list(range(100)))
    assert big_q.get_nowait_batch(
        rank=0, epoch=0, num_items=100) == list(range(100))


def test_qsize(ray_start_regular_shared):

    q = BatchQueue(num_epochs=1, num_trainers=1, max_concurrent_epochs=1)

    items = list(range(10))
    size = 0

    assert q.qsize(rank=0, epoch=0) == size

    for item in items:
        q.put(rank=0, epoch=0, item=item)
        size += 1
        assert q.qsize(rank=0, epoch=0) == size

    for item in items:
        assert q.get(rank=0, epoch=0) == item
        size -= 1
        assert q.qsize(rank=0, epoch=0) == size


def test_shutdown(ray_start_regular_shared):
    q = BatchQueue(num_epochs=1, num_trainers=1, max_concurrent_epochs=1)
    actor = q.actor
    q.shutdown()
    assert q.actor is None
    with pytest.raises(RayActorError):
        ray.get(actor.empty.remote(rank=0, epoch=0))


def test_custom_resources(ray_start_regular_shared):
    current_resources = ray.available_resources()
    assert current_resources["CPU"] == 1.0

    # By default an actor should not reserve any resources.
    q = BatchQueue(num_epochs=1, num_trainers=1, max_concurrent_epochs=1)
    current_resources = ray.available_resources()
    assert current_resources["CPU"] == 1.0
    q.shutdown()

    # Specify resource requirement. The queue should now reserve 1 CPU.
    q = BatchQueue(
        num_epochs=1,
        num_trainers=1,
        max_concurrent_epochs=1,
        actor_options={"num_cpus": 1})

    def no_cpu_in_resources():
        return "CPU" not in ray.available_resources()

    wait_for_condition(no_cpu_in_resources)
    q.shutdown()


def test_pull_from_streaming_batch_queue(ray_start_regular_shared):
    class QueueBatchPuller:
        def __init__(self, batch_size, queue, num_epochs):
            self.batch_size = batch_size
            self.queue = queue
            self.num_epochs = num_epochs

        def __iter__(self):
            pending = []
            is_done = False
            epoch = 0
            while not is_done or pending:
                if not pending:
                    for item in self.queue.get_batch(rank=0, epoch=epoch):
                        if item is None:
                            epoch += 1
                            if epoch >= self.num_epochs:
                                is_done = True
                            break
                        else:
                            pending.append(item)
                ready, pending = ray.wait(pending, num_returns=1)
                yield ray.get(ready[0])

    @ray.remote
    class QueueConsumer:
        def __init__(self, batch_size, queue, num_epochs):
            self.batch_puller = QueueBatchPuller(batch_size, queue, num_epochs)
            self.data = []

        def consume(self):
            for item in self.batch_puller:
                self.data.append(item)
                time.sleep(0.3)

        def get_data(self):
            return self.data

    @ray.remote
    def dummy(x):
        return x

    num_batches = 5
    batch_size = 4
    q = BatchQueue(
        num_epochs=num_batches, num_trainers=1, max_concurrent_epochs=1)
    consumer = QueueConsumer.remote(batch_size, q, num_batches)
    consumer.consume.remote()
    data = list(range(batch_size * num_batches))
    for epoch, idx in enumerate(range(0, len(data), batch_size)):
        time.sleep(1)
        batch = data[idx:idx + batch_size]
        q.put_nowait_batch(
            rank=0, epoch=epoch, items=[dummy.remote(item) for item in batch])
        q.put_nowait(rank=0, epoch=epoch, item=None)
    consumed_data = ray.get(consumer.get_data.remote())
    assert len(consumed_data) == len(data), consumed_data
    assert set(consumed_data) == set(data), set(data)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
