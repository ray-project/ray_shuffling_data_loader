# Ray-based Shuffling Data Loader

A Ray-based data loader with per-epoch shuffling and configurable pipelining, for shuffling and loading training data for distributed training of machine learning models.

# Installation

You can install latest master via:

```bash
pip install git+https://github.com/ray-project/ray_shuffling_data_loader.git@main#egg=ray_shuffling_data_loader
```

# Usage

This shuffling data loader exposes a generic `ShufflingDataset` abstraction that takes a list of input files and shuffling configuration, and yields `batch_size`-sized GPU batches via an iterator. This abstraction is framework-agnostic.

We also expose a class deriving from [Torch `IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset), for distributed Torch training use. This dataset abstraction also takes a feature and label column spec, used for converting the Pandas DataFrame GPU batches to Torch tensors.

```python
from shuffling_data_loader_ray.torch_dataset import TorchShufflingDataset

# Input and shuffling configuration for the Torch shuffling dataset.
# Files containing training dataset.
filenames = ["s3://foo/bar"]
# Number of training epochs.
num_epochs = 10
# Number of model training workers.
num_trainers = 1
# Size of a GPU batch.
batch_size = 25000
# Number of reducers for the shuffler to use.
num_reducers = 8
# Maximum number of epoch shuffling runs that are allowed to run concurrently.
max_concurrent_epochs = 2
# The rank of this trainer. This can typically be retrieved via your
# distributed training framework, e.g. `hvd.rank()` for Horovod.
rank = 0

# Spec for feature and label columns, used to convert Pandas DataFrame GPU batches
# into Torch tensors.
# The column names in your Parquet data files.
feature_columns = ["col_0_name", "col_1_name"]
# The Torch types of your columns, e.g. `torch.float64`.
feature_types = [col_0_type, col_1_type]
# The label column name in your Parquet data files.
label_column = "label_col_name"
# The Torch type of your label column, e.g. `torch.float64`.
label_type = label_col_type

# Construct a Torch shuffling dataset that yields shuffled batch_size-sized
# GPU batches, transforming those Pandas DataFrame GPU batches into Torch tensors.
# The shuffling will be kicked off upon construction of this dataset.
ds = TorchShufflingDataset(
    filenames,
    num_epochs,
    num_trainers,
    batch_size,
    rank,
    num_reducers=num_reducers,
    max_concurrent_epochs=max_concurrent_epochs,
    feature_columns=feature_columns,
    feature_types=feature_types,
    label_column=label_column,
    label_type=label_type)

# Train a model on the yielded GPU batches.
for epoch in range(num_epochs):
    # We must set the epoch in order for the dataset to give us the GPU
    # batches for the right epoch.
    ds.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(ds):
        # Do a model training step on this GPU batch.
```
