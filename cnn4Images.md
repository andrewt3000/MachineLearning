# Convolutional Neural Networks for image processing

[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/syllabus.html)   
Class taught by Andrej Karpathy at Stanford. [videos](https://www.youtube.com/playlist?list=PLLvH2FwAQhnpj1WEB-jHmPuUeQ8mX-XXG) were taken down for [legal concerns](https://twitter.com/karpathy/status/727618058471112704?lang=en).   

[A Beginner's Guide To Understanding Convolutional Neural Networks](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)  
[An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

### CNN Parameters and terminology

Types of hidden layers in cnns: convolutional layers, max pooling layers, fully connected layers.  

1. Convolutional Layers - layers that have filters (aka kernels) that are convolved around the image. Then perform a Relu activation.    
  CNN Parmeters: All parameters can vary by layer.  
  1. Number of filters (aka kernels) (K) - typically a power of 2. eg. 32, 64, 128. 

  2. Size of filter (F) - typically odd square numbers. typical values are 3x3, 5x5, up to 11x11.  

  3. Stride (S) - How much to shift the filter.  
  
  4. Padding (P) - zero padding on edges of input volume. Padding can make output volume same size as input volume when padding =  (F - 1)/2. Examples: 3x3 filters / padding = 1, 5x5 filters / padding 2, 7x7 filters / padding 3.

2. Max pool layer: max pooling is when you take a section and subsample only the largest value in the pool.
  parameter: size of the section to subsample. example 2x2.

3. Fully connected layers are the same as hidden layers in a vanilla neural network.  

#### Calculate output volume of convolutional layer
The depth is the number of filters.  
The width (height is the same assuming image is square) is W2=(W1−F+2P)/S+1  
W1 is original width of image. F is filter witdth, P is padding, S is stride. example: F=3,S=1,P=1  
[Source](http://cs231n.github.io/convolutional-networks/)  

### Image applications
Object classification - identifying an object in an image.  
Localization - drawing a bounding box around an object. One or fixed number of objects.  
Detection - drawing a bounding box around an unlimited number of objects.  
Semantic segmentation - Label every pixel between classes. Don’t distinguish between different instances.  
Instance segmentation - Segment between different instances of objects.  
Image cpationing - describing the objects in an image.  

### Object Classification. State of the art research papers
[Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)  
LeCun et al., 1998 LeNet-5.  Input 32x32 images.  
Architecture: 6 5x5 filters, stride 1. Pooling layers 2x2 stride 2. (conv pool conv pool conv fc)  

[ImageNet](http://www.image-net.org/) [[paper](http://www.image-net.org/papers/imagenet_cvpr09.pdf)] is a dataset of 1.2 million images with 1,000 labels.  

[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  
Krizhevsky et al. (2012) AlexNet breakthrough paper for convolutional neural networks. First use of Relu (rectified linear units)  
top-1 and top-5 error rates of 37.5% and 17.0% on ImageNet.  
Architecture: First conv layer: 96 11x11 filters, stride 4. 5 convolutional layers, 3 fully connected layers.  

[Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901v3.pdf)  
Zeiler and Fergus, 2013 - ZFNet.  
14.8% top 5 error on ImageNet (eventually got to 11% with company clarify)  
Architecture: First conv layer: 7x7 filter, stride 2.  

[VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/pdf/1409.1556.pdf)  
Simonyan et al. 2014.  VGGNet.  
7.3% top 5 error on ImageNet.   
Architecture: 16 - 19 layers. 3x3 convolutional filters, stride 1, pad 1. 2x2 max pool, stride 2.    

[Going deeper with convolutions](https://arxiv.org/pdf/1409.4842v1.pdf)
Szegedy et al., 2014  GoogleNet. Inception.  
Architecture: 22 layers.  

[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf)  
He et al., 2015 aka resnet.  Uses skip connections (residual net). That diminishes vanishing/exploding gradient problems that occur in deep nets.  
3.57% top 5 error on ImageNet  
Architecture: 152 layers. [ResNet model code](https://github.com/KaimingHe/deep-residual-networks).

[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)  
Szegedy et al. 2016 
3.08% top 5 error on ImageNet  [blog post](https://research.googleblog.com/2016/08/improving-inception-and-image.html)  
Architecture: 75 layers  
