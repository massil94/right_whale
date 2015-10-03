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
* Python labelling tool: slot
```
git clone https://github.com/cvhciKIT/sloth.git
```
* Resize images to 256x256

### Hierarchical models: 
* Build new features using AutoEncoder to compress the information
* Cluster whales into groups of similars whales (K-means)
* Classification. K can be a parameter of Classif.

Idea: Try to 'transfer' prediction probas to best representant of each class

Goal: multiclass identification 

### Using semi supervised learning
