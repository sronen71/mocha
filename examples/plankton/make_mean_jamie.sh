#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

TRAIN_FULL="_train"
SUFFIX="_256"
./build/tools/compute_image_mean examples/plankton/plankton${TRAIN_FULL}_lmdb${SUFFIX} \
  examples/plankton/plankton_mean${TRAIN_FULL}${SUFFIX}.binaryproto

echo "Done."
