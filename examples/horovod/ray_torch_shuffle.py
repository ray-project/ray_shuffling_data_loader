import os
import pickle
import time
import timeit

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import torch
import tempfile
import horovod.torch as hvd
from horovod.ray import RayExecutor

from ray_shuffling_data_loader.torch_dataset import (TorchShufflingDataset)
from ray_shuffling_data_loader.data_generation import (generate_data,
                                                       DATA_SPEC)

import argparse

DEFAULT_DATA_DIR = "s3://shuffling-data-loader-benchmarks/data/"

numpy_to_torch_dtype = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}

# Training settings
parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=250000,
    metavar="N",
    help="input batch size for training (default: 64)")
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=250000,
    metavar="N",
    help="input batch size for testing (default: 1000)")
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)")
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    metavar="LR",
    help="learning rate (default: 0.01)")
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)")
parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help="disables CUDA training")
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    metavar="S",
    help="random seed (default: 42)")
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help=("how many batches to wait before logging training "
          "status"))
parser.add_argument(
    "--fp16-allreduce",
    action="store_true",
    default=False,
    help="use fp16 compression during allreduce")
parser.add_argument(
    "--use-adasum",
    action="store_true",
    default=False,
    help="use adasum algorithm to do reduction")
parser.add_argument(
    "--gradient-predivide-factor",
    type=float,
    default=1.0,
    help=("apply gradient predivide factor in optimizer "
          "(default: 1.0)"))
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--cpus-per-worker", type=int, default=2)
parser.add_argument("--mock-train-step-time", type=float, default=1.0)

# Synthetic training data generation settings.
parser.add_argument("--cache-files", action="store_true", default=False)
parser.add_argument("--num-rows", type=int, default=2 * (10**7))
parser.add_argument("--num-files", type=int, default=25)
parser.add_argument("--max-row-group-skew", type=float, default=0.0)
parser.add_argument("--num-row-groups-per-file", type=int, default=5)
parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)

# Shuffling data loader settings.
parser.add_argument("--num-reducers", type=int, default=32)
parser.add_argument("--max-concurrent-epochs", type=int, default=2)
parser.add_argument("--address", default="auto")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train_main(args, filenames):
    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and not args.no_cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)
    rank = hvd.rank()
    train_dataset = create_dataset(
        filenames,
        batch_size=args.batch_size,
        rank=rank,
        num_epochs=args.epochs,
        world_size=hvd.size(),
        num_reducers=args.num_reducers,
        max_concurrent_epochs=args.max_concurrent_epochs)
    model = Net()
    # By default, Adasum doesn"t need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if torch.cuda.is_available() and not args.no_cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr * lr_scaler, momentum=args.momentum)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = (hvd.Compression.fp16
                   if args.fp16_allreduce else hvd.Compression.none)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)

    def _train(epoch):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        train_dataset.set_epoch(epoch)
        start_epoch = timeit.default_timer()
        last_batch_time = start_epoch
        batch_wait_times = []
        for batch_idx, (data, target) in enumerate(train_dataset):
            batch_wait_times.append(timeit.default_timer() - last_batch_time)
            if torch.cuda.is_available() and not args.no_cuda:
                if isinstance(data, list):
                    data = [t.cuda() for t in data]
                target = target.cuda()
            optimizer.zero_grad()
            # output = model(data)
            if batch_idx % args.log_interval == 0:
                print(
                    f"Processing batch {batch_idx} in epoch {epoch} on worker "
                    f"{rank}.")
            time.sleep(args.mock_train_step_time)
            # TODO(Clark): Add worker synchronization barrier here.
            # loss = F.nll_loss(output, target)
            # loss.backward()
            # optimizer.step()
            last_batch_time = timeit.default_timer()
        epoch_duration = timeit.default_timer() - start_epoch
        avg_batch_wait_time = np.mean(batch_wait_times)
        std_batch_wait_time = np.std(batch_wait_times)
        max_batch_wait_time = np.max(batch_wait_times)
        min_batch_wait_time = np.min(batch_wait_times)
        print(f"\nEpoch {epoch}, worker {rank} stats over "
              f"{len(batch_wait_times)} steps: {epoch_duration:.3f}")
        print(f"Mean batch wait time: {avg_batch_wait_time:.3f}s +- "
              f"{std_batch_wait_time}")
        print(f"Max batch wait time: {max_batch_wait_time:.3f}s")
        print(f"Min batch wait time: {min_batch_wait_time:.3f}s")
        return batch_wait_times

    print(f"Starting training on worker {rank}.")
    batch_wait_times = []
    for epoch in range(args.epochs):
        # TODO(Clark): Don't include stats from first epoch since we already
        # expect that epoch to be cold?
        batch_wait_times.extend(_train(epoch))
    print(f"Done training on worker {rank}.")
    avg_batch_wait_time = np.mean(batch_wait_times)
    std_batch_wait_time = np.std(batch_wait_times)
    max_batch_wait_time = np.max(batch_wait_times)
    min_batch_wait_time = np.min(batch_wait_times)
    print(f"\nWorker {rank} training stats over {args.epochs} epochs:")
    print(f"Mean batch wait time: {avg_batch_wait_time:.3f}s +- "
          f"{std_batch_wait_time}")
    print(f"Max batch wait time: {max_batch_wait_time:.3f}s")
    print(f"Min batch wait time: {min_batch_wait_time:.3f}s")
    # TODO(Clark): Add logic to the dataset abstraction so we don't have to do
    # this.
    if rank == 0:
        print("Waiting in rank 0 worker to let other workers consume queue...")
        time.sleep(10)
        print("Done waiting in rank 0 worker.")


def create_dataset(filenames, *, batch_size, rank, num_epochs, world_size,
                   num_reducers, max_concurrent_epochs):
    print(f"Creating Torch shuffling dataset for worker {rank} with "
          f"{batch_size} batch size, {num_epochs} epochs, {num_reducers} "
          f"reducers, and {world_size} trainers.")
    feature_columns = list(DATA_SPEC.keys())
    feature_types = [
        numpy_to_torch_dtype[dtype] for _, _, dtype in DATA_SPEC.values()
    ]
    label_column = feature_columns.pop()
    label_type = feature_types.pop()
    return TorchShufflingDataset(
        filenames,
        num_epochs,
        world_size,
        batch_size,
        rank,
        num_reducers=num_reducers,
        max_concurrent_epochs=max_concurrent_epochs,
        feature_columns=feature_columns,
        feature_types=feature_types,
        label_column=label_column,
        label_type=label_type)


if __name__ == "__main__":
    args = parser.parse_args()
    from ray_shuffling_data_loader.stats import human_readable_size
    import ray
    print("Connecting to Ray cluster...")
    ray.init(address=args.address)

    num_rows = args.num_rows
    num_files = args.num_files
    num_row_groups_per_file = args.num_row_groups_per_file
    max_row_group_skew = args.max_row_group_skew
    data_dir = args.data_dir

    cache_path = os.path.join(tempfile.gettempdir(), "data_cache")
    filenames = None
    if args.cache_files and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                filenames, num_bytes = pickle.load(f)
        except Exception as exc:
            print(f"Cache load failed - {exc}")

    if not filenames:

        print(f"Generating {num_rows} rows over {num_files} files, with "
              f"{num_row_groups_per_file} row groups per file and at most "
              f"{100 * max_row_group_skew:.1f}% row group skew.")
        filenames, num_bytes = generate_data(num_rows, num_files,
                                             num_row_groups_per_file,
                                             max_row_group_skew, data_dir)
        if args.cache_files:
            with open(os.path.join(tempfile.gettempdir(), "data_cache"),
                      "wb") as f:
                pickle.dump((filenames, num_bytes), f)
    print(f"Generated {len(filenames)} files containing {num_rows} rows "
          f"with {num_row_groups_per_file} row groups per file, totalling "
          f"{human_readable_size(num_bytes)}.")

    print("Create Ray executor")
    num_workers = args.num_workers
    cpus_per_worker = args.cpus_per_worker
    settings = RayExecutor.create_settings(timeout_s=30)
    executor = RayExecutor(
        settings,
        num_workers=num_workers,
        use_gpu=True,
        cpus_per_worker=cpus_per_worker)
    executor.start()
    executor.run(train_main, args=[args, filenames])
    executor.shutdown()

    print("Done consuming batches.")
