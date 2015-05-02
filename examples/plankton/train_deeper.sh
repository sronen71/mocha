#!/usr/bin/env sh

./build/tools/caffe train -solver=examples/plankton/inet_solver1.prototxt
./build/tools/caffe train -solver=examples/plankton/inet_solver2.prototxt -weights=examples/plankton/inet_iter_37000.caffemodel
./build/tools/caffe train -solver=examples/plankton/inet_solver3.prototxt -weights=examples/plankton/inet_iter_24000.caffemodel
