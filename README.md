[image1]: ./runs/1528867024.4458013/uu_000007.png "1"
[image2]: ./runs/1528867024.4458013/um_000005.png "2"
[image3]: ./runs/1528867024.4458013/um_000030.png "3"
[image4]: ./runs/1528867024.4458013/um_000083.png "4"
[image5]: ./runs/1528867024.4458013/um_000075.png "5"
[image6]: ./runs/1528867024.4458013/uu_000058.png "6"
[image7]: ./runs/1528867024.4458013/uu_000094.png "6"
[image8]: ./runs/1528867024.4458013/uu_000099.png "6"

# Semantic Segmentation
### Introduction

The implementation labels the pixels of a road in images using a "Fully Convolutional Network (FCN) based on the VGG-16 image classifier architecture".

Following points are worth noting:

1. Started with the base code provided at https://github.com/udacity/CarND-Semantic-Segmentation .
2. Used the suggested [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).
3. Added the TODO implementation code in [main.py](https://github.com/MyCodeBits/Term3-Udacity-CarND-Semantic-Segmentation/blob/master/main.py) which encompassed:
- Using pre-trained VGG-16 network and converting it to a fully convolutional network by converting the final fully connected layer to a 1x1 convolution. Two classes were used : road and not-road.
- Performance gains using : skip connections, 1x1 convolutions and sampling.
- Loss fn. as cross entropy.
- Optimizer: Adam optimizer.
4. Ran the Model training with Adam Optimizer and inference using AWS GPU.


### Run

Following are the dependencies:

- [Python 3](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)

Ran the following command to run the project:

```
python main.py
```


This creates an output folder '[runs/1528867024.4458013](https://github.com/MyCodeBits/Term3-Udacity-CarND-Semantic-Segmentation/blob/master/runs/1528867024.4458013/)' which has 291 images having their road pixels colored/tagged <span style="color:green">GREEN<em></em></span>.

##### Some outputs images



[um_000003.png](./runs/1528867024.4458013/um_000003.png)
![alt text][image1]

[um_000005.png](./runs/1528867024.4458013/um_000005.png)
![alt text][image2]

[um_000030.png](./runs/1528867024.4458013/um_000030.png)
![alt text][image3]

[um_000083.png](./runs/1528867024.4458013/um_000083.png)
![alt text][image4]

[um_000075.png](./runs/1528867024.4458013/um_000075.png)
![alt text][image5]

[uu_000058.png](./runs/1528867024.4458013/uu_000058.png)
![alt text][image6]

[uu_000094.png](./runs/1528867024.4458013/uu_000094.png)
![alt text][image7]

[uu_000099.png](./runs/1528867024.4458013/uu_000099.png)
![alt text][image8]


#### Run Output:

```
ubuntu@ip-172-31-7-191:~/CarND-Semantic-Segmentation$ python main.py
TensorFlow Version: 1.2.1
2018-06-13 04:29:32.288621: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-06-13 04:29:32.289019: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 0 with properties:
name: Tesla M60
major: 5 minor: 2 memoryClockRate (GHz) 1.1775
pciBusID 0000:00:1e.0
Total memory: 7.43GiB
Free memory: 7.36GiB
2018-06-13 04:29:32.289045: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0
2018-06-13 04:29:32.289054: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   Y
2018-06-13 04:29:32.290040: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla M60, pci bus id: 0000:00:1e.0)
2018-06-13 04:29:32.387917: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla M60, pci bus id: 0000:00:1e.0)
Default GPU Device: /gpu:0
2018-06-13 04:29:32.389441: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla M60, pci bus id: 0000:00:1e.0)
Tests Passed
Tests Passed
2018-06-13 04:29:56.261752: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla M60, pci bus id: 0000:00:1e.0)
2018-06-13 04:29:57.131297: I tensorflow/compiler/xla/service/platform_util.cc:58] platform CUDA present with 1 visible devices
2018-06-13 04:29:57.131355: I tensorflow/compiler/xla/service/platform_util.cc:58] platform Host present with 16 visible devices
2018-06-13 04:29:57.136172: I tensorflow/compiler/xla/service/service.cc:198] XLA service 0x43fecb0 executing computations on platform Host. Devices:
2018-06-13 04:29:57.136195: I tensorflow/compiler/xla/service/service.cc:206]   StreamExecutor device (0): <undefined>, <undefined>
2018-06-13 04:29:57.137240: I tensorflow/compiler/xla/service/platform_util.cc:58] platform CUDA present with 1 visible devices
2018-06-13 04:29:57.137261: I tensorflow/compiler/xla/service/platform_util.cc:58] platform Host present with 16 visible devices
2018-06-13 04:29:57.139842: I tensorflow/compiler/xla/service/service.cc:198] XLA service 0x44322e0 executing computations on platform CUDA. Devices:
2018-06-13 04:29:57.139861: I tensorflow/compiler/xla/service/service.cc:206]   StreamExecutor device (0): Tesla M60, Compute Capability 5.2
Tests Passed
2018-06-13 04:30:16.206988: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla M60, pci bus id: 0000:00:1e.0)
Tests Passed
Tests Passed
Downloading pre-trained vgg model...
997MB [00:19, 52.2MB/s]
Extracting model...
2018-06-13 04:30:44.431629: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla M60, pci bus id: 0000:00:1e.0)
Training NN ...

EPOCH No. 1 ...
Training Loss: = 1.653
Training Loss: = 9.652
....
....
....
....
EPOCH No. 3 ...
Training Loss: = 0.232
Training Loss: = 0.199
Training Loss: = 0.138
Training Loss: = 0.164

....
....
....
....
EPOCH No. 22 ...
Training Loss: = 0.073
Training Loss: = 0.052
Training Loss: = 0.312
Training Loss: = 0.055
....
....
....
....
EPOCH No. 30 ...
Training Loss: = 0.026
Training Loss: = 0.029
Training Loss: = 0.020
Training Loss: = 0.023
....
....
....
....
....
....
....
EPOCH No. 50 ...
Training Loss: = 0.031
```


### Rubric Answers

 - Does the project load the pretrained vgg model? **Yes**
 - Does the project learn the correct features from the images? **Yes**
 - Does the project optimize the neural network? **Yes**
 - Does the project train the neural network? **Yes. Loss is already shown above in RUN section.**
 - Does the project train the model correctly? **Yes, it does.**
 - Does the project use reasonable hyperparameters? **Yes,**
      - keep_prob: 0.5
      - epochs : 50
      - batch size : 5, and
      - learning rate : 0.5.
 - Does the project correctly label the road? **Yes**



- - - -


# Base README.md from UDACITY for reference


### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)

 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow.
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy.
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well.

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
