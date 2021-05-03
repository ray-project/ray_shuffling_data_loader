from setuptools import find_packages, setup

setup(
    name="ray_shuffling_data_loader",
    version="0.1.0",
    author="Ray Team",
    author_email="ray-dev@googlegroups.com",
    description="A Ray-based data loader with pipelined per-epoch shuffling.",
    long_description=(
        "A Ray-based data loader with per-epoch shuffling and configurable "
        "pipelining, for shuffling and loading training data for distributed "
        "training of machine learning models."),
    url="https://github.com/ray-project/ray_shuffling_data_loader",
    install_requires=[
        "ray",
        "numpy",
        "pandas",
        "smart-open",
        "torch",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
)
