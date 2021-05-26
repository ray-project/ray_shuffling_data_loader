import asyncio
import logging
from typing import Optional, Any, List, Dict
from collections.abc import Iterable
import collections
import time

import ray

logger = logging.getLogger(__name__)


class Empty(Exception):
    pass


class Full(Exception):
    pass


# TODO(Clark): Update docstrings and examples.


class BatchQueue:
    """A first-in, first-out queue implementation on Ray.

    The behavior and use cases are similar to those of the asyncio.Queue class.

    Features both sync and async put and get methods.  Provides the option to
    block until space is available when calling put on a full queue,
    or to block until items are available when calling get on an empty queue.

    Optionally supports batched put and get operations to minimize
    serialization overhead.

    Args:
        maxsize (optional, int): maximum size of the queue. If zero, size is
            unbounded.
        actor_options (optional, Dict): Dictionary of options to pass into
            the QueueActor during creation. These are directly passed into
            QueueActor.options(...). This could be useful if you
            need to pass in custom resource requirements, for example.

    Examples:
        >>> q = Queue()
        >>> items = list(range(10))
        >>> for item in items:
        >>>     q.put(item)
        >>> for item in items:
        >>>     assert item == q.get()
        >>> # Create Queue with the underlying actor reserving 1 CPU.
        >>> q = Queue(actor_options={"num_cpus": 1})
    """

    def __init__(self,
                 num_epochs: int,
                 num_trainers: int,
                 max_concurrent_epochs: int,
                 maxsize: int = 0,
                 name: str = None,
                 connect: bool = False,
                 wait_for_trainers: bool = True,
                 actor_options: Optional[Dict] = None,
                 connect_retries: int = 5) -> None:
        if connect:
            assert actor_options is None
            assert name is not None
            self.actor = connect_queue_actor(name, connect_retries)
        else:
            actor_options = actor_options or {}
            if name is not None:
                actor_options["name"] = name
            self.actor = ray.remote(_QueueActor).options(
                **actor_options).remote(
                    max_concurrent_epochs, num_epochs,
                    num_trainers, maxsize, wait_for_trainers)

    def new_epoch(self, epoch: int):
        return ray.get(self.actor.new_epoch.remote(epoch))

    def producer_done(self, rank: int, epoch: int):
        self.actor.producer_done.remote(rank, epoch)

    def wait_until_all_epochs_done(self):
        ray.get(self.actor.wait_until_all_epochs_done.remote())

    def __len__(self) -> int:
        return ray.get(self.actor.size.remote())

    def size(self, rank: int, epoch: int) -> int:
        """The size of the queue."""
        return ray.get(self.actor.qsize.remote(rank, epoch))

    def qsize(self, rank: int, epoch: int) -> int:
        """The size of the queue."""
        return self.size(rank, epoch)

    def empty(self, rank: int, epoch: int) -> bool:
        """Whether the queue is empty."""
        return ray.get(self.actor.empty.remote(rank, epoch))

    def full(self, rank: int, epoch: int) -> bool:
        """Whether the queue is full."""
        return ray.get(self.actor.full.remote(rank, epoch))

    def put(self,
            rank: int, epoch: int,
            item: Any,
            block: bool = True,
            timeout: Optional[float] = None) -> None:
        """Adds an item to the queue.

        If block is True and the queue is full, blocks until the queue is no
        longer full or until timeout.

        There is no guarantee of order if multiple producers put to the same
        full queue.

        Raises:
            Full: if the queue is full and blocking is False.
            Full: if the queue is full, blocking is True, and it timed out.
            ValueError: if timeout is negative.
        """
        if not block:
            try:
                ray.get(self.actor.put_nowait.remote(rank, epoch, item))
            except asyncio.QueueFull:
                raise Full
        else:
            if timeout is not None and timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                ray.get(self.actor.put.remote(rank, epoch, item, timeout))

    def put_batch(self,
                  rank: int, epoch: int,
                  items: Iterable,
                  block: bool = True,
                  timeout: Optional[float] = None) -> None:
        """Adds an item to the queue.

        If block is True and the queue is full, blocks until the queue is no
        longer full or until timeout.

        There is no guarantee of order if multiple producers put to the same
        full queue.

        Raises:
            Full: if the queue is full and blocking is False.
            Full: if the queue is full, blocking is True, and it timed out.
            ValueError: if timeout is negative.
        """
        if not block:
            try:
                ray.get(self.actor.put_nowait_batch.remote(rank, epoch, items))
            except asyncio.QueueFull:
                raise Full
        else:
            if timeout is not None and timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                ray.get(
                    self.actor.put_batch.remote(rank, epoch, items, timeout))

    async def put_async(self,
                        rank: int, epoch: int,
                        item: Any, block: bool = True,
                        timeout: Optional[float] = None) -> None:
        """Adds an item to the queue.

        If block is True and the queue is full,
        blocks until the queue is no longer full or until timeout.

        There is no guarantee of order if multiple producers put to the same
        full queue.

        Raises:
            Full: if the queue is full and blocking is False.
            Full: if the queue is full, blocking is True, and it timed out.
            ValueError: if timeout is negative.
        """
        if not block:
            try:
                await self.actor.put_nowait.remote(rank, epoch, item)
            except asyncio.QueueFull:
                raise Full
        else:
            if timeout is not None and timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                await self.actor.put.remote(rank, epoch, item, timeout)

    def get(self,
            rank: int, epoch: int,
            block: bool = True,
            timeout: Optional[float] = None) -> Any:
        """Gets an item from the queue.

        If block is True and the queue is empty, blocks until the queue is no
        longer empty or until timeout.

        There is no guarantee of order if multiple consumers get from the
        same empty queue.

        Returns:
            The next item in the queue.

        Raises:
            Empty: if the queue is empty and blocking is False.
            Empty: if the queue is empty, blocking is True, and it timed out.
            ValueError: if timeout is negative.
        """
        if not block:
            try:
                return ray.get(self.actor.get_nowait.remote(rank, epoch))
            except asyncio.QueueEmpty:
                raise Empty
        else:
            if timeout is not None and timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                return ray.get(self.actor.get.remote(rank, epoch, timeout))

    async def get_async(self,
                        rank: int, epoch: int,
                        block: bool = True,
                        timeout: Optional[float] = None) -> Any:
        """Gets an item from the queue.

        There is no guarantee of order if multiple consumers get from the
        same empty queue.

        Returns:
            The next item in the queue.
        Raises:
            Empty: if the queue is empty and blocking is False.
            Empty: if the queue is empty, blocking is True, and it timed out.
            ValueError: if timeout is negative.
        """
        if not block:
            try:
                return await self.actor.get_nowait.remote(rank, epoch)
            except asyncio.QueueEmpty:
                raise Empty
        else:
            if timeout is not None and timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                return await self.actor.get.remote(rank, epoch, timeout)

    def get_batch(self, rank: int, epoch: int) -> Any:
        return ray.get(self.actor.get_batch.remote(rank, epoch))

    def put_nowait(self, rank: int, epoch: int, item: Any) -> None:
        """Equivalent to put(item, block=False).

        Raises:
            Full: if the queue is full.
        """
        return self.put(rank, epoch, item, block=False)

    def put_nowait_batch(self, rank: int, epoch: int, items: Iterable) -> None:
        """Takes in a list of items and puts them into the queue in order.

        Raises:
            Full: if the items will not fit in the queue
        """
        if not isinstance(items, Iterable):
            raise TypeError("Argument 'items' must be an Iterable")

        ray.get(self.actor.put_nowait_batch.remote(rank, epoch, items))

    def get_nowait(self, rank: int, epoch: int) -> Any:
        """Equivalent to get(block=False).

        Raises:
            Empty: if the queue is empty.
        """
        return self.get(rank, epoch, block=False)

    def get_nowait_batch(
            self, rank: int, epoch: int, num_items: int = None) -> List[Any]:
        """Gets items from the queue and returns them in a
        list in order.

        Raises:
            Empty: if the queue does not contain the desired number of items
        """
        if num_items is not None:
            if not isinstance(num_items, int):
                raise TypeError("Argument 'num_items' must be an int")
            if num_items < 0:
                raise ValueError("'num_items' must be nonnegative")

        return ray.get(
            self.actor.get_nowait_batch.remote(rank, epoch, num_items))

    def task_done(self, rank: int, epoch: int, num_items: int = 1):
        self.actor.task_done.remote(rank, epoch, num_items)

    def ready(self):
        ray.get(self.actor.ready.remote())

    def shutdown(self, force: bool = False, grace_period_s: int = 5) -> None:
        """Terminates the underlying QueueActor.

        All of the resources reserved by the queue will be released.

        Args:
            force (bool): If True, forcefully kill the actor, causing an
                immediate failure. If False, graceful
                actor termination will be attempted first, before falling back
                to a forceful kill.
            grace_period_s (int): If force is False, how long in seconds to
                wait for graceful termination before falling back to
                forceful kill.
        """
        if self.actor:
            if force:
                ray.kill(self.actor, no_restart=True)
            else:
                done_ref = self.actor.__ray_terminate__.remote()
                done, not_done = ray.wait([done_ref], timeout=grace_period_s)
                if not_done:
                    ray.kill(self.actor, no_restart=True)
        self.actor = None


def connect_queue_actor(name, num_retries=5):
    """
    Connect to the named actor denoted by `name`, retrying up to
    `num_retries` times. Note that the retry uses exponential backoff.
    If max retries is reached without connecting, an exception is raised.
    """
    retries = 0
    sleep_dur = 1
    last_exc = None
    while retries < num_retries:
        try:
            return ray.get_actor(name)
        except Exception as e:
            retries += 1
            logger.info(
                f"Couldn't connect to queue actor {name}, trying again in "
                f"{sleep_dur} seconds: {retries} / {num_retries}, error: "
                f"{e!s}")
            time.sleep(sleep_dur)
            sleep_dur *= 2
            last_exc = e
    raise ValueError(f"Unable to connect to queue actor {name} after "
                     f"{num_retries} retries. Last error: {last_exc!s}")


class _QueueActor:
    def __init__(
            self, max_epochs, num_epochs, num_trainers, maxsize,
            wait_for_trainers=True):
        self.max_epochs = max_epochs
        self.num_epochs = num_epochs
        self.curr_epochs = collections.deque()
        self.queues = [
            [
                asyncio.Queue(maxsize)
                for _ in range(num_trainers)]
            for _ in range(num_epochs)]
        self.queue_producer_done = [
            [
                asyncio.Event()
                for _ in range(num_trainers)]
            for _ in range(num_epochs)]
        self.maxsize = maxsize
        self.wait_for_trainers = wait_for_trainers

    async def new_epoch(self, epoch: int):
        if len(self.curr_epochs) == self.max_epochs:
            first_epoch = self.curr_epochs.popleft()
            # Wait until queue producers for all trainers are done.
            await asyncio.wait([
                event.wait()
                for event in self.queue_producer_done[first_epoch]])
            if self.wait_for_trainers:
                # Wait until trainers are done with batches.
                await asyncio.wait([
                    queue.join()
                    for queue in self.queues[first_epoch]])
            # TODO(Clark): The queues and events for this epoch should no
            # longer be accessed after this point, so we could set them to
            # None here and save some space.
        self.curr_epochs.append(epoch)

    async def producer_done(self, rank: int, epoch: int):
        await self.queues[epoch][rank].put(None)
        self.queue_producer_done[epoch][rank].set()

    async def wait_until_all_epochs_done(self):
        await asyncio.wait([
            event.wait()
            for event in self.queue_producer_done[self.num_epochs - 1]])
        if self.wait_for_trainers:
            # With the final epoch producer being done, we're guaranteed that
            # no more batches will be added to the queue, so we join on the
            # current queue items.
            await asyncio.wait([
                queue.join() for queue in self.queues[self.num_epochs - 1]])

    def size(self):
        return sum(q.qsize() for queues in self.queues for q in queues)

    def qsize(self, rank: int, epoch: int):
        return self.queues[epoch][rank].qsize()

    def empty(self, rank: int, epoch: int):
        return self.queues[epoch][rank].empty()

    def full(self, rank: int, epoch: int):
        return self.queues[epoch][rank].full()

    async def put(self, rank: int, epoch: int, item, timeout=None):
        try:
            await asyncio.wait_for(self.queues[epoch][rank].put(item), timeout)
        except asyncio.TimeoutError:
            raise Full

    async def put_batch(self, rank: int, epoch: int, items, timeout=None):
        for item in items:
            try:
                await asyncio.wait_for(self.queues[epoch][rank].put(item),
                                       timeout)
            except asyncio.TimeoutError:
                raise Full

    async def get(self, rank: int, epoch: int, timeout=None):
        try:
            return await asyncio.wait_for(self.queues[epoch][rank].get(),
                                          timeout)
        except asyncio.TimeoutError:
            raise Empty

    async def get_batch(self, rank: int, epoch: int):
        batch = [await self.queues[epoch][rank].get()]
        while True:
            try:
                batch.append(self.queues[epoch][rank].get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    def put_nowait(self, rank: int, epoch: int, item):
        self.queues[epoch][rank].put_nowait(item)

    def put_nowait_batch(self, rank: int, epoch: int, items):
        # If maxsize is 0, queue is unbounded, so no need to check size.
        if (self.maxsize > 0
                and len(items) + self.qsize(rank, epoch) > self.maxsize):
            raise Full(f"Cannot add {len(items)} items to queue of size "
                       f"{self.qsize()} and maxsize {self.maxsize}.")
        for item in items:
            self.queues[epoch][rank].put_nowait(item)

    def get_nowait(self, rank: int, epoch: int):
        return self.queues[epoch][rank].get_nowait()

    def get_nowait_batch(self, rank: int, epoch: int, num_items: int = None):
        if num_items is None:
            # If num_items isn't specified, get all items in the queue.
            num_items = self.qsize(rank, epoch)
        if num_items > self.qsize(rank, epoch):
            raise Empty(f"Cannot get {num_items} items from queue of size "
                        f"{self.qsize()}.")
        return [
            self.queues[epoch][rank].get_nowait() for _ in range(num_items)]

    def task_done(self, rank: int, epoch: int, num_items: int = 1):
        for _ in range(num_items):
            self.queues[epoch][rank].task_done()

    def ready(self):
        pass
