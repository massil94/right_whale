# autoencoder + cnn ?
* http://stackoverflow.com/questions/24752655/unsupervised-pre-training-for-convolutional-neural-network-in-theano

# Use CNN for detection ?
* https://github.com/rbgirshick/rcnn

## Deep Learning Resources
* List of references: http://www.jeremydjacksonphd.com/?p=28
* Good tutorial from Stanford: http://ufldl.stanford.edu/tutorial/

## AMI for AWS EC2
ami-03e67874, available in EU-Ireland

## How to run ipython notebook
there is a pyzmq-related problem on the AMI, you can solve it this way:
```
# switch to root
sudo su
# install gnureadline
sudo apt-get install libncurses5-dev
pip install gnureadline
# uninstall and reinstall ipython
pip uninstall ipython
pip install "ipython[all]"
```

## TODO

### Hierarchical models
* Build new features using AutoEncoder to compress the information
* Cluster whales into groups of similars whales (K-means)
* Classification. K can be a parameter of Classif.

Idea: Try to 'transfer' prediction probas to best representant of each class

Goal: multiclass identification 

### Using semi supervised learning

### win the first prize
