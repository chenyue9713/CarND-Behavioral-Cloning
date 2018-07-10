# **Behavioral Cloning Project**

## Overview

In this project, I will implement deep learning to effciently teach a car to drive automatically in a driving simulator provided by Udacity.
The techniques is called [Behavioral Cloning](https://link.springer.com/content/pdf/10.1007%2F978-1-4899-7687-1_69.pdf), which definition is:

      Behavioral cloning is a method by which human subcognitive skills can be captured and reproduced in a computer program. As the human
      subject performs the skill, his or her actions are recorded along with the situation that gave rise to the action. A log of these records is used as input
      to a learning program. The learning program outputs a set of rules that reproduce the skilled behavior. This method can be used to construct
      automatic control systems for complex tasks for which classical control theory is inadequate. It can also be used for training.

The CNN I used was based on NVIDIA's [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) paper with dropout for avoiding overfit. 
Considering the training efficiency, I train my network on AWS at most of time.

## Model Architecture
