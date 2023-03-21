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

(+2 questions)

### Main Problem

Regard of the most important choices for implementing a transformer-based processing system in computer vision tasks are:

- the number of attention heads per transformer encoder
- the number of transformer-encoders stacked

The problem is: how to select these numbers? The result substantially depends on them but it is difficult to make recommendations for these choices.

### Approach

The main approach to find the result is to implement some of the most commonly used CV datasets and test the them with different combinations of number of attention heads and transformer-encoders using transformer model.

## Architecture Overview

### Parameter Structure of a Multi-head Transformer

### Measuring The Degree of Overdetermination

### Computing Results Using Different Datasets

### Results

### Conclusions

### Future Work

## Critical Analysis

## Reference

## Video Recording
