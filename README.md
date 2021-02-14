# AdvReg4Intpltn

This pytorch repo trys to reproduce the result of "[Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer](http://arxiv.org/abs/1807.07543)" by David Berthelot, Colin Raffel, Aurko Roy, and Ian Goodfellow. The original repo is https://github.com/brain-research/acai .

The architure of DNN is borrowed from the pytorch version repo

https://gist.github.com/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0

The clustering accuracy and single layer classifier accuracy are

Model | clustering accuracy | single layer classifier accuracy
-------------------- | --------------------- | ---------------------
ae | 0.846 | 0.963
acae | 0.913 | 0.972
