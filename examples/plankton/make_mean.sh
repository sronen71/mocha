#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

./build/tools/compute_image_mean examples/starvision/starvision_train_lmdb \
  examples/starvision/starvision_mean.binaryproto

echo "Done."
