import time
import numpy as np
import torch
import ray
import argparse
import os

DEFAULT_DATA_DIR = "s3://shuffling-data-loader-benchmarks/data/"


def create_parser():
    parser = argparse.ArgumentParser(description="Eric Example")
    parser.add_argument("--address")
    parser.add_argument("--num-rows", type=int, default=2 * (10**8))
    parser.add_argument("--num-files", type=int, default=50)
    parser.add_argument("--read-cache", action="store_true", default=False)
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250000,
        metavar="N",
        help="input batch size for training (default: 64)")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def create_torch_iterator(split, batch_size, rank=None):
    print(f"Creating Torch shuffling dataset for worker {rank} with "
          f"{batch_size} batch size.")
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
    DATA_SPEC = {
        "embeddings_name0": (0, 2385, np.int64),
        "embeddings_name1": (0, 201, np.int64),
        "embeddings_name2": (0, 201, np.int64),
        "embeddings_name3": (0, 6, np.int64),
        "embeddings_name4": (0, 19, np.int64),
        "embeddings_name5": (0, 1441, np.int64),
        "embeddings_name6": (0, 201, np.int64),
        "embeddings_name7": (0, 22, np.int64),
        "embeddings_name8": (0, 156, np.int64),
        "embeddings_name9": (0, 1216, np.int64),
        "embeddings_name10": (0, 9216, np.int64),
        "embeddings_name11": (0, 88999, np.int64),
        "embeddings_name12": (0, 941792, np.int64),
        "embeddings_name13": (0, 9405, np.int64),
        "embeddings_name14": (0, 83332, np.int64),
        "embeddings_name15": (0, 828767, np.int64),
        "embeddings_name16": (0, 945195, np.int64),
        "one_hot0": (0, 3, np.int64),
        "one_hot1": (0, 50, np.int64),
        "labels": (0, 1, np.float64),
    }
    feature_columns = list(DATA_SPEC.keys())
    feature_types = [
        numpy_to_torch_dtype[dtype] for _, _, dtype in DATA_SPEC.values()
    ]
    label_column = feature_columns.pop()
    label_type = feature_types.pop()

    torch_iterator = split.to_torch(
         label_column=label_column,
         feature_columns=feature_columns,
         label_column_dtype=label_type,
         feature_column_dtypes=feature_types,
         batch_size=batch_size,
         # prefetch_blocks: int = 0,
         # drop_last: bool = False
    )
    return torch_iterator


def create_dataset(data_dir):
    pipeline = ray.data.read_parquet(data_dir)\
        .repeat().random_shuffle()
    return pipeline

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    print("Connecting to Ray cluster...")
    ray.init(address=args.address)

    data_dir = os.path.join(args.data_dir, f"r{args.num_rows:_}-f{args.num_files}/")
    pipeline = create_dataset(data_dir)
    splits = pipeline.split(args.num_workers, equal=True)

    @ray.remote
    def consume(split, rank=None, batch_size=None):
        torch_iterator = create_torch_iterator(split, batch_size=batch_size, rank=rank)
        for i, (x, y) in enumerate(torch_iterator):
            time.sleep(1)
            if i % 10 == 0:
                print(i)
        return

    tasks = [
        consume.remote(split, rank=idx, batch_size=args.batch_size)
        for idx, split in enumerate(splits)
    ]
    ray.get(tasks)
