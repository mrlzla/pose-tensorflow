#!/bin/sh

# Listed here https://github.com/tensorflow/models/tree/master/slim#pre-trained-models

curl http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz | tar xvz
curl http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz | tar xvz
curl http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz | tar xvz
