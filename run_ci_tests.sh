pushd ray_shuffling_data_loader/tests || exit 1
echo "============="
echo "Running tests"
echo "============="
END_STATUS=0
###
# ADD TEST SCRIPTS HERE
###
if ! python -m pytest -v --durations=0 -x "test_batch_queue.py" ; then END_STATUS=1; fi
###
# END
###
popd || exit 1

if [ "$END_STATUS" = "1" ]; then
  echo "At least one test has failed, exiting with code 1"
fi
exit "$END_STATUS"