#!/usr/bin/env sh
# Create the starvision lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

# Set SEMI=TRUE to creat semi-supervised data set

set SEMI=TRUE

EXAMPLE=examples/plankton
DATA=/home/shai/mocha/data/plankton
TOOLS=build/tools

TRAIN_DATA_ROOT=/home/shai/plankton/train/
VAL_DATA_ROOT=/home/shai/plankton/train/
TEST_DATA_ROOT=/home/shai/plankton/test/
SEMI_DATA_ROOT=/home/shai/plankton/

RESIZE_HEIGHT=0
RESIZE_WIDTH=0

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



if $SEMI; then
    echo "Creating semi-supervised lmdb..."

    GLOG_logtostderr=1 $TOOLS/convert_imageset \
        --resize_height=$RESIZE_HEIGHT \
        --resize_width=$RESIZE_WIDTH \
        --shuffle \
        --gray \
        $SEMI_DATA_ROOT \
        $DATA/semi.txt \
        $EXAMPLE/plankton_semi_lmdb
fi



echo "creating full train lmdb..."



GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray \
    $TRAIN_DATA_ROOT \
    $DATA/fulltrain.txt \
    $EXAMPLE/plankton_full_lmdb




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
