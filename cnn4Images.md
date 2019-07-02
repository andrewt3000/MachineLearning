# Convolutional Neural Networks for image processing
[Convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNNs) are variations of neural networks that contain convolutional layers. Typically used for grids of input such as images.   

### CNN HyperParameters

Types of hidden layers in cnns: convolutional layers, max pooling layers, fully connected layers.  

1. Convolutional Layers - layers that have filters (aka kernels) that are convolved around the image. Then perform a Relu activation.    
  CNN Parmeters: All parameters can vary by layer.  
    1. Number of filters (aka kernels) (K) - typically a power of 2. eg. 32, 64, 128. Typically increases at deeper layers.  

    2. Size of filter (F) - typically odd square numbers. typical values are 3x3, 5x5, up to 11x11.  

    3. Stride (S) - How much to shift the filter.  
  
    4. Padding (P) - zero padding on edges of input volume. Padding can make output volume same size as input volume when padding =  (F - 1)/2. Examples: 3x3 filters / padding = 1, 5x5 filters / padding 2, 7x7 filters / padding 3.

2. Max pool layer: max pooling is when you take a section and subsample only the largest value in the pool.  
  hyperparameter: size of the section to subsample. example 2x2.

3. Fully connected layers are the same as hidden layers in a vanilla neural network.  

#### Calculate output volume of convolutional layer
The depth is the number of filters.  
The width (height is the same assuming image is square) is W2=(W1−F+2P)/S+1  
W1 is original width of image. F is filter witdth, P is padding, S is stride. example: F=3,S=1,P=1  
[Source](http://cs231n.github.io/convolutional-networks/)  


### Computer Vision tasks
**Object classification** - identifying an object in an image.  
**Classification + Localization** - drawing a bounding box around an object. One or fixed number of objects. Train a CNN to classify but also train box coordinates as a regression problem finding (x, y, w, h)    
**Semantic segmentation** - Label every pixel between classes. Don’t distinguish between different instances. Use a cnn, downsample and then upsample for efficiency.     
**Object detection** - drawing a bounding box around an unlimited number of objects. Get regions of interest, and run algorithms such as R-CNN, fast R-CNN, faster R-CNN. There are also new single pass models such as [Yolo](https://pjreddie.com/darknet/yolo/), [V2](https://pjreddie.com/darknet/yolov2/), [V3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)  
 and [SSD](https://arxiv.org/abs/1512.02325).  
**Instance segmentation** - Segment between different instances of objects. State of the art is Mask R-CNN.  
**Image cpationing** - describing the objects in an image.  

<img src="https://github.com/andrewt3000/MachineLearning/blob/master/img/cv.png" />  

### Object Classification: Case Studies
[Case Studies](http://cs231n.github.io/convolutional-networks/#case)  

#### LeNet-5
[Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)  
LeCun et al., 1998 .  Input 32x32 images.  
Architecture: 6 5x5 filters, stride 1. Pooling layers 2x2 stride 2. (conv pool conv pool conv fc)  

[ImageNet](http://www.image-net.org/) [[paper](http://www.image-net.org/papers/imagenet_cvpr09.pdf)] is a dataset of 1.2 million images with 1,000 labels.  

#### AlexNet
[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  
Krizhevsky et al. (2012)  breakthrough paper for convolutional neural networks. First use of Relu (rectified linear units)  
top-1 and top-5 error rates of 37.5% and 17.0% on ImageNet.  
Architecture: First conv layer: 96 11x11 filters, stride 4. 5 convolutional layers, 3 fully connected layers.  

#### ZFNet
[Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901v3.pdf)  
Zeiler and Fergus, 2013 -   
14.8% top 5 error on ImageNet (eventually got to 11% with company clarify)  
Architecture: First conv layer: 7x7 filter, stride 2.  

#### VGGNet
[VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)  
Simonyan et al. 2014.  .  
7.3% top 5 error on ImageNet.   
Architecture: 16 - 19 layers. 3x3 convolutional filters, stride 1, pad 1. 2x2 max pool, stride 2.  
Smaller filters, deeper layers. Typically there are more filters as you go deeper. 

#### Inception
[Going deeper with convolutions](https://arxiv.org/pdf/1409.4842v1.pdf)
Szegedy et al., 2014  GoogleNet. Inception.  
Architecture: 22 layers.  

#### resnet
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf)  
He et al., 2015 aka .  Uses skip connections (residual net). That diminishes vanishing/exploding gradient problems that occur in deep nets.  
3.57% top 5 error on ImageNet  
Architecture: 152 layers. [ResNet model code](https://github.com/KaimingHe/deep-residual-networks).

#### Inception v4
[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)  
Szegedy et al. 2016 
Updated inception to use resnets.  
3.08% top 5 error on ImageNet  [blog post](https://research.googleblog.com/2016/08/improving-inception-and-image.html)  
Architecture: 75 layers  

#### DenseNet
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)  
Huang et al. 2018  
Breakthrough: "For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers."  

#### Additional Resources

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)   
[2017 videos](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  

[A Beginner's Guide To Understanding Convolutional Neural Networks](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)  
[An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

