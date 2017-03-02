# 500pxChallenge

This challenge requires the fundamentals of Deep Learning, more specifically how to generate Adversarial Images. More information on the subject can be found be found [here](http://karpathy.github.io/2015/03/30/breaking-convnets/)__

Using Tensorflow, I was able to learn Deep MNIST 
[here](https://www.tensorflow.org/versions/r0.11/tutorials/mnist/pros/#deep-mnist-for-experts).__

The challenge was to select ten images of the digit '2', also classified as 2 and modify the images so that the network incorrectly classfies them as 6. The result is an image containing 10 rows and 3 columns, with the rows coresponding to the selected images of '2', and the columns representing the original image, delta, and the adversarial image. This is saved in result.jpg__

![alt result](https://github.com/dxy159/500pxChallenge/blob/master/result.jpg)


