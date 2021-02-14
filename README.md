# AdvReg4Intpltn

This pytorch repo trys to reproduce the result of "[Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer](http://arxiv.org/abs/1807.07543)" by David Berthelot, Colin Raffel, Aurko Roy, and Ian Goodfellow. The original repo is https://github.com/brain-research/acai .

The architure of DNN is different from the original repo. Here we use a very simple encoder, decoder and critic NN. I guess this is the reason for performance gap.

The clustering accuracy and single layer classifier accuracy are

Model | clustering accuracy | single layer classifier accuracy
-------------------- | --------------------- | ---------------------
ae | 0.655 | 0.887 
acae | 0.6807 | 0.902
