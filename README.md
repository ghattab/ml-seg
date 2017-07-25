# Cell segmentation task using Deep Learning (DL)
Two experiments are included in this study using the Network in Network structure (NIN) and the Fully Convoluted Network (FCN), respectively.
The NIN is used as is from Lin et al. 2013.
Whereas, the FCN is adapted from Long et al. 2015 and is extended from a LeNet as seen in LeCun et al. 1998. 
The convolutionalization step transforms fully connected layers into convolution layers, which enables a classification net to output a heatmap.

## Usage 

* `patchk.py` creates image patches and clusters them using the K-means method
* `image-patches/` folder that encompasses the resulting patches
* `buildDataset` creates a HDF5 dataset provided the list of patches 
* Files with the `big` prefix handle the model's training
* Files with the `surgery` suffix comprehend the 2-dimensional convolutions and convolutionalization

## License
```
The MIT License (MIT)

Copyright (c) Georges Hattab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
