# Paper Presentation: Number of Attention Heads vs. Number of Transformer-Encoders in Computer Vision

Authors of Paper: Tomas Hrycej, Bernhard Bermeitinger and Siegfried Handschuh
Institute of Computer Science, University of St.Gallen (HSG), St.Gallen, Switzerland
{firstname.lastname}@unisg.ch

Paper Link: https://arxiv.org/abs/2209.07221

Presenter: Enya Tan

## Overview

### Background

The transformer model is a sequence-to-sequence model that is widely used in natural language processing tasks such as machine translation, text generation, etc. Recently, more and more researchers have introduced transformer models into the field of computer vision with good results. However, in computer vision tasks, due to the differences between image data and natural language data, the transformer model needs to face some new challenges and issues.

### Introduction

By conducting experiments on multiple computer vision tasks, the author studies the influence of the combination of different attention heads and transformer encoders on the model performance. 

The experimental results show that increasing the number of attention heads and transformer encoders within a certain range can significantly improve the model performance. However, after a certain number, increasing the number of heads and Transformer encoders will not bring more performance improvement, but will bring the risk of performance degradation. Therefore, for different computer vision tasks, there is a trade-off between the number of attention heads and the number of encoders when choosing a model architecture.

The paper indicates the result that if the role of context in images to be classified can be assumed to be small, it is favorable to use multiple transformers with a low number of heads (such as one or two). In classifying objects whose class may heavily depend on the context within the image (i.e., the meaning of a patch being dependent on other patches), the number of heads is equally important as that of transformers.

In conclusion, this paper provides some useful references for transformer model design and optimization in the field of computer vision.

### Main Problem

Regard of the most important choices for implementing a transformer-based processing system in computer vision tasks are:

- the number of attention heads per transformer encoder
- the number of transformer-encoders stacked

The problem is: how to select these numbers? The result substantially depends on them but it is difficult to make recommendations for these choices.

[**Question 1: Any thoughts about the affect of changing number of attention heads and transformer-encoders? What do you think will happen?**]

### Approach

The main approach to find the result in this paper is to implement some of the most commonly used CV datasets and test the them with different combinations of number of attention heads and transformer-encoders using the transformer model.

## Architecture Overview

### Parameter Structure of a Multi-head Transformer

The parameters of a multi-head transformer (in the form of only encoders and no decoders) consist of:
1. matrices transforming token vectors to their compressed form (value in the transformer terminology);
2. matrices transforming token vectors to the feature vectors for similarity measure (key and query), used for context-relevant weighting;
3. matrices transforming the compressed and context-weighted form of tokens back to the original token vector length;
4. parameters of a feedforward network with one hidden layer;

Increasing the number of heads increases the parameter count resulting from the transformation matrices of the attention mechanism, but other parameters such as the feedforward network parameters remain constant. Therefore, the overall parameter count growth is not fully proportional. 

On the other hand, increasing the number of encoders increases the parameter count of all parameters, including the transformation matrices and feedforward network parameters, resulting in a fully proportional increase in total parameter count.

### Measuring The Degree of Overdetermination

The paper then discusses the issue of fitting a parameterized model to a dataset. Given K training examples that require fitting M outputs, this means there are MK equations to be solved. In this process, P free parameters are sought to achieve the best fit. Therefore, we have a system of MK equations with P variables. If MK = P, the system has a unique and exact solution. If MK < P, the system is underdetermined with infinitely many solutions. If MK > P, the system is overdetermined and cannot be solved exactly, only approximately. In overdetermined cases, the degree of overfitting depends on the ratio of the number of constraints to the number of parameters. This ratio can be denoted as:

<div align=center><img src="https://github.com/eyttt51/transformers-paper-presentation-EnyaTan/blob/main/figures/f1.png"/></div>

Q refers to the ratio of the number of constraints (MK) to the number of free parameters (P) in the model. MK represents the number of output values to be fitted for K training examples, while P represents the number of free parameters in the model that are sought for the best fit.

Here, Q is a metric that measures the degree of overconstraint in a system. When Q < 1, the system is underdetermined; when Q = 1, the system is just constrained; and when Q > 1, the system is overdetermined. In the case of an overdetermined system, the degree of overfitting depends on the ratio of the number of constraints to the number of parameters. If we have too many parameters in an overdetermined system, we may overfit our model and learn noise, which can lead to poor performance on new data.

<div align=center><img src="https://github.com/eyttt51/transformers-paper-presentation-EnyaTan/blob/main/figures/f2.png"/></div>

E is the MSE, σ^2 is the noise variance, P is the number of model parameters, M is the number of training samples, and K is the number of outputs.

This equation suggests that the MSE decreases with an increasing number of training samples (M), decreasing number of model parameters (P), or increasing number of outputs (K). However, the decrease in MSE is limited by the quality factor (Q) of the system. When Q is high, the decrease in MSE with more training data is slower, indicating a higher risk of overfitting.

<div align=center><img src="https://github.com/eyttt51/transformers-paper-presentation-EnyaTan/blob/main/figures/f.png"/></div>

The formula represents the prediction error, where σ^2 is the variance of noise, with Q representing the degree of overdetermination of the system. In parameter fitting, the prediction error consists of the imprecision of the model and noise. If both the training and testing sets represent the statistical population, the level of noise is identical to that in the training set. The values of the constants c and Q in the formula are influenced by factors such as the size of the training set, the number of model parameters, and the model structure.

This discusses the variation of mean squared error (MSE) for the training and testing sets with changes in the number of training samples and model parameters. It also emphasizes the importance of having a model structure that is expressive enough to capture the input-output relationship of the real system and the impact of the number of parameters.

For instance, if we are training a neural network for image classification with a small amount of training data and a large number of parameters, the model may overfit the training data and perform poorly on the testing set. Conversely, if we have a large amount of training data but a small number of parameters, the model may underfit the training data and fail to capture the input-output relationship of the real system. Therefore, in choosing the number of model parameters, people need to balance model complexity and interpretability with model performance on the testing set.

<div align=center><img src="https://github.com/eyttt51/transformers-paper-presentation-EnyaTan/blob/main/figures/f4.png"/></div>

The x-axis of the figs is the determination ratio Q, in logarithmic scale (so that the value 100 corresponds to Q = 1).

### Computing Results Using Different Datasets

This part describes a series of model fitting experiments performed in computer vision tasks to investigate the impact of the number of attention heads and transformer encoders on model performance. The authors used several popular image datasets and optimized models with different numbers of heads and encoders for each task. Some models that used a large number of both attention heads and encoders had too many parameters and did not perform well. The authors present some results, including models with four encoders and any number of attention heads, and models with four attention heads and any number of encoders. In the following, a cross-section of the results is presented:

- four transformer-encoders and any number of attention heads;
- four attention heads and any number of transformer-encoders.

Also, it describes a series of model experiments in computer vision tasks. The author optimized the model by varying the number of attention heads and Transformer encoder. The performance of the model was tested on both the training and test sets. The results showed a relationship between the performance of the model and the number of parameters - the fewer parameters, the better the performance. The author used simple data augmentation techniques to help the model better learn, such as random flipping, rotation, and cropping. Additionally, the author used the AdamW algorithm to optimize the model's performance during training.

### Results

**1. Dataset MNIST**

<div align=center><img src="https://github.com/eyttt51/transformers-paper-presentation-EnyaTan/blob/main/figures/d1.png"/></div>

The results are substantially more sensitive to the lack of transformer-encoders: the rightmost configurations with four heads but one or two transformer-encoders have a poor performance. By contrast, using only one or two heads leads only to a moderate performance loss. In other words, it is more productive to stack more transformer-encoders than to use many heads. This is not surprising for simple images such as those of digits. The context-dependency of image patches can be expected to be rather low and to require only a simple attention mechanism with a moderate number of heads.

**2. Dataset CIFAR-100**

<div align=center><img src="https://github.com/eyttt51/transformers-paper-presentation-EnyaTan/blob/main/figures/d2.png"/></div>

The results are more sensitive to the lack of transformer-encoders than to that of heads. How far a high number of transformer-encoders would be helpful, cannot be assessed because of getting then into the region of Q < 1. With this training set size, a reduction of some transformer parameters such as key, query, and value width would be necessary.

**3. Dataset CUB-200-2011**

<div align=center><img src="https://github.com/eyttt51/transformers-paper-presentation-EnyaTan/blob/main/figures/d3.png"/></div>

The cross-entropies for the training and the test sets are mostly consistent due to the high determination ratio Q. There are relatively small differences between small numbers of heads and transformer-encoders. Both categories seem to be comparable. This suggests, in contrast to the datasets treated above, a relatively large contribution of context to the classification performance — multiple heads are as powerful as multiple transformer-encoders. This is not unexpected in the given domain: the habitat of the bird in the image background may constitute a key contribution to classifying the species.

**4. Dataset places365**

<div align=center><img src="https://github.com/eyttt51/transformers-paper-presentation-EnyaTan/blob/main/figures/d4.png"/></div>

There are hardly any differences between variants with varying heads and those varying transformer-encoders. With a given total number of parameters (and thus a similar ratio Q), both categories seem to be equally important. It can be conjectured that there is a relatively strong contribution of context to the classification performance can be assumed.

**5. Dataset imagenet**

<div align=center><img src="https://github.com/eyttt51/transformers-paper-presentation-EnyaTan/blob/main/figures/d5.png"/></div>

Compared to the other experiments, the determination ratio is very high (103 to 104) which means that the number of parameters in the classification network is too small and even larger stacks of transformer-encoders with more attention heads could decrease the loss even further. Looking at the varying number of attention heads, it can be seen that their number has a low impact on the performance.

### Conclusions

Determining the appropriate number of self-attention heads on one hand and, on the other hand, the number of transformer-encoder layers is an important choice for CV tasks using the Transformer architecture. A key decision concerns the total number of parameters to ensure good generalization performance of the fitted model. The determination ratio Q, as de-fined in section 3, is a reliable measure: values significantly exceeding unity (e.g., Q > 4) lead to test set loss similar to that of the training set. This sets the boundaries within which the number of heads and the number of transformer-encoders can be chosen.

Different CV applications exhibit different sensitivity to varying and combining both numbers.

- If the role of context in images to be classified can be assumed to be small, it is favorable to “invest” the parameters into multiple transformer-encoders. With too few transformer-encoders, the performance will rapidly deteriorate. Simultaneously, a low number of attention heads (such as one or two) is sufficient.
- In classifying objects whose class may heavily depend on the context within the image (i.e., the meaning of a patch being dependent on other patches), the number of attention heads is equally important as that of transformer-encoders.

[**Question 2: In this paper, the authors use datasets from specific domains, such as the bird image classification dataset. Do you think these results can be extended to other computer vision tasks and datasets?**]

### Future Work

Although this study provides a systematic comparison between the number of attention heads and number of consecutive transformer-encoders, the sheer number of different hyperparameters is still underrepresented. Any of the listed hyperparameters in the experiments (section 4) need the same systematic analysis as the current study.

## Critical Analysis

1. Although the experimental results of this paper are meaningful, their generalizability may be subject to some limitations. The experiments were only conducted on a specific image classification task and dataset, which may limit the generalizability of the conclusions. 
2. In addition, the hyperparameter settings used in the paper may not be applicable to all cases, such as different input image sizes and patch sizes. Therefore, caution is needed when extrapolating these results to other domains, and further research is needed to determine the optimal hyperparameter combinations.

## Reference

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, page 21, Vienna, Austria.

Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Dataset, University of Toronto.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. In Advances in Neural Information Processing Systems, volume 25. Curran Associates, Inc.

Lecun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324.

Li, F., Li, S., Fan, X., Li, X., and Chang, H. (2022). Structural Attention Enhanced Continual Meta-Learning for Graph Edge Labeling Based Few-Shot Remote Sensing Scene Classification. Remote Sensing, 14(3):485.

Loshchilov, I. and Hutter, F. (2019). Decoupled Weight Decay Regularization. 1711.05101.

Wah, C., Branson, S., Welinder, P., Perona, P., and Belongie, S. (2011). The Caltech-UCSD Birds-200-2011 Dataset. Dataset CNS-TR-2011-001, California Institute of Technology, Pasadena, CA.

Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., and Torralba, A. (2018). Places: A 10 Million Image Database for Scene Recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(6):1452–1464.

## Video Recording

https://youtu.be/1ixt9XnpN5Y
