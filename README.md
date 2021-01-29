# Simple Football Player detector

This repository contains a simple football player detector which aims to be

- small: it has a small memory footprint on both CPU and GPU
- fast: it can easily run real-time (on an NVIDIA 1080 Ti it achieves 100 fps on 1080 frames)
- accurate: it uses the [Max-Margin Object Detection](https://arxiv.org/abs/1502.00046) loss to achieve high accuracy with few training examples

## Details

It has been implemented entirely in C++, using the amazing [Deep Learning API](http://blog.dlib.net/2016/06/a-clean-c11-deep-learning-api.html) from [dlib](http://dlib.net/).

### Training set

The training data for this detector comes from:

https://pspagnolo.jimdofree.com/download/

I manually extracted every other frame in sequences 1 and 6 and manually annotated them myself.

### Network architecture

The network architecture can be divided into 3 parts:

- downsampling: 3 convolutional blocks with big kernel size to increase the receptive field
    - 16 7x7 conv with stride = 2, batch norm, relu
    - 16 5x5 conv with stride = 2, batch norm, relu
    - 16 3x3 conv with stride = 2, batch norm, relu
- feature extractor: 3 convolutional blocks with small kernel size for fast inference
    - 32 3x3 conv with stride = 2, batch norm, relu
    - 32 3x3 conv with stride = 2, batch norm, relu
    - 32 3x3 conv with stride = 2, batch norm, relu
- final convolution that serves as the sliding window over the extracted features
    - N 9x9 conv with stride = 1, where N = num_classes * 5 (class present + 4 for bounding box regression)

In our case the model has the follwing detector windows, found by custering the bounding boxes in the training set:

- detector_windows: (person: 56x105, 92x118, 32x73, 62x60, 65x181, 23x115)

This means it's only able to find players whose bounding box match those with an intersection over union (IoU) > 0.5.
In order to make a better detector, a bigger dataset would be needed to find different aspect ratios and sizes.
We can see that the dataset does not contain players in an horizontal position, so the detector won't be able to find them.

## Performance

These are the performance details When processing 1920x1080 frames:

- Speed: around 100 fps (or 10 ms per frame)
- VRAM: around 400 MiB on the GPU
