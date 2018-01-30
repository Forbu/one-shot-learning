# one-shot-learning

Pytorch implementation of the one shot learning paper : 
https://arxiv.org/pdf/1605.06065.pdf

First part : 
Implement (ie modify the current NTM code) the Least Recently Used Access feature.
With CNN to handle the image data and LSTM to handle the image representation and the label (controller).

Second part :
Training on different dataset :
Omniglot (rare image from different alphabet).
Imagenet.

What would be interesting is to use transfert learning (by select the same weights) on the CNN part of the network to get a "good" representation of the feature via a previous cumbersome model.

#### Références : 
I (strongly) inspire myself from :
https://github.com/loudinthecloud/pytorch-ntm (github repo)
for the NTM implementation (that I modify to create the one-shot learning framework with Least Recently Used Access feature).

The original paper on one shot learning using the same initial dataset (paper) :
https://arxiv.org/pdf/1605.06065.pdf

The Neural Turing Machine paper by Alex Graves (paper):
https://arxiv.org/pdf/1410.5401.pdf

Other interesting ressources like the Differential neural computer (paper) :
https://www.nature.com/articles/nature20101.pdf

Other one-shot method via Hierarchical Bayesian Program Learning (paper) :
https://cims.nyu.edu/~brenden/LakeEtAlNips2013.pdf

A nice blog post about "One Shot Learning and Siamese Networks in Keras" :
https://sorenbouma.github.io/blog/oneshot/



