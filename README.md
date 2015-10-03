## Deep Learning Resources
http://www.jeremydjacksonphd.com/?p=28

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

## right_whale
Kaggle Right Whale Recognition challenge https://www.kaggle.com/c/noaa-right-whale-recognition

## TODO

### Data prep
Split between 
6926 test

### Using hierarchical models: 
* Build new features using AutoEncoder to compress the information
* Cluster whales into groups of similars whales (K-means)
* Classification. K can be a parameter of Classif.

Try to 'give' prediction probas to best representant of each class

Goal: multiclass identification 

### Using semi supervised learning
