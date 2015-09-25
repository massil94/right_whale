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
0-Data prep
Split between 
6926 test

I-Using hierarchical models: 
Step 0: Build new features using AutoEncoder to compress the information
Step 1: Cluster whales into groups of similars whales (K-means)
Step 2: Classification. K can be a parameter of Classif.

Goal: multiclass identification 


II-Using semi supervised learning
