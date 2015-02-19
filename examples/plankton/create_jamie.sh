#!/usr/bin/env sh
# Create the starvision lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

EXAMPLE=examples/plankton
DATA=data/plankton
TOOLS=build/tools

TRAIN_DATA_ROOT=/home/jamie/data/Plankton/train/
VAL_DATA_ROOT=/home/jamie/data/Plankton/train/
TEST_DATA_ROOT=/home/jamie/data/Plankton/test/
DO_FULL=true
DO_TEST=false

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false # when using crops of 256x256
if $RESIZE; then
  RESIZE_HEIGHT=48
  RESIZE_WIDTH=48
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi
echo "RESIZE $RESIZE_HEIGHT $RESIZE_WIDTH"
SUFFIX="_256"
SEED_SUFFIX="_seed2"


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


if $DO_FULL; then
  echo "Creating full lmdb..."
  GLOG_logtostderr=1 $TOOLS/convert_imageset \
      --resize_height=$RESIZE_HEIGHT \
      --resize_width=$RESIZE_WIDTH \
      --shuffle \
      --gray \
      $TRAIN_DATA_ROOT \
      $DATA/full${SUFFIX}${SEED_SUFFIX}.txt \
      $EXAMPLE/plankton_full_lmdb${SUFFIX}${SEED_SUFFIX}
fi


echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray \
    $TRAIN_DATA_ROOT \
    $DATA/train${SUFFIX}${SEED_SUFFIX}.txt \
    $EXAMPLE/plankton_train_lmdb${SUFFIX}${SEED_SUFFIX}

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --gray \
    $VAL_DATA_ROOT \
    $DATA/val${SUFFIX}${SEED_SUFFIX}.txt \
    $EXAMPLE/plankton_val_lmdb${SUFFIX}${SEED_SUFFIX}

if $DO_TEST; then
  echo "Creating test lmdb..."

  GLOG_logtostderr=1 $TOOLS/convert_imageset \
      --resize_height=$RESIZE_HEIGHT \
      --resize_width=$RESIZE_WIDTH \
      --shuffle \
      --gray \
      $TEST_DATA_ROOT \
      $DATA/test${SUFFIX}.txt \
      $EXAMPLE/plankton_test_lmdb${SUFFIX}
fi
echo "Done."
