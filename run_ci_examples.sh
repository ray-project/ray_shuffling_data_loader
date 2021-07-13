set -e

pushd ray_shuffling_data_loader || exit 1
ray stop || true
echo "================"
echo "Running examples"
echo "================"
echo "running dataset.py" && python dataset.py
echo "running torch_dataset.py" && python torch_dataset.py

popd || exit 1
