#!/usr/bin/env sh
# Create the starvision lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

EXAMPLE=examples/plankton
DATA=/home/shai/mocha/data/plankton
TOOLS=build/tools

TRAIN_DATA_ROOT=/home/shai/plankton/train/
VAL_DATA_ROOT=/home/shai/plankton/train/
TEST_DATA_ROOT=/home/shai/plankton/test/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=FALSE # when using crops of 256x256
if $RESIZE; then
  RESIZE_HEIGHT=48
  RESIZE_WIDTH=48
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi
echo "RESIZE $RESIZE_HEIGHT $RESIZE_WIDTH"

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/plankton_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $EXAMPLE/plankton_val_lmdb

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray \
    $TEST_DATA_ROOT \
    $DATA/test.txt \
    $EXAMPLE/plankton_test_lmdb


echo "Done."
