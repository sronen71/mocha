#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

./build/tools/compute_image_mean examples/plankton/plankton_train_lmdb \
  examples/plankton/plankton_mean.binaryproto

echo "Done."

