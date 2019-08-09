# Two-stream-action-recognition-keras
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/127003611.svg)](https://zenodo.org/badge/latestdoi/127003611) 

We use spatial and temporal stream cnn under the Keras framework to reproduce published results on UCF-101 action recognition dataset. This is a project from a research internship at the Machine Intelligence team, IBM Research AI, Almaden Research Center, by Wushi Dong (dongws@uchicago.edu).


## References

*  [[1] Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional)

*  [[2] Convolutional Two-Stream Network Fusion for Video Action Recognition](https://github.com/feichtenhofer/twostreamfusion)

*  [[3] Five video classification methods](https://github.com/harvitronix/five-video-classification-methods/blob/master/README.md)

*  [[4] UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://arxiv.org/abs/1212.0402)


## Data

### Spatial input data -> rgb frames
  First, download the dataset from UCF into the `data` folder:
  `cd data && wget http://crcv.ucf.edu/data/UCF101/UCF101.rar`
  
  Then extract it with `unrar e UCF101.rar`. in disk, which costs about 5.9G.
  
  We use split #1 for all of our experiments.

### Motion input data -> stacked optical flows

Download the preprocessed tvl1 optical flow dataset directly from https://github.com/feichtenhofer/twostreamfusion. 
  ```
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003
  cat ucf101_tvl1_flow.zip* > ucf101_tvl1_flow.zip
  unzip ucf101_tvl1_flow.zip
  ```

## Training

### Spatial-stream cnn

*  We classify each video by looking at a single frame. We use ImageNet pre-trained models and transfer learning to retrain Inception on our data. We first fine-tune the top dense layers for 10 epochs and then retrain the top two inception blocks.

### Temporal-stream cnn

*  We train the temporal-stream cnn from scratch. In every mini-batch, we randomly select 128 (batch size) videos from 9537 training videos and futher randomly select 1 optical flow stack in each video. We follow the reference paper and use 10 x-channels and 10 y-channels for each optical flow stack, resulting in a input shape of (224, 224, 20). 

*  Multiple workers are utilized in the data generator for faster training.

### Data augmentation

*  Both streams apply the same data augmentation technique such as corner cropping and random horizontal flipping. Temporally, we pick the starting frame among those early enough to guarantee a desired number of frames. For shorter videos, we looped the video as many times as necessary to satisfy each model’s input interface.

## Testing

*  We fused the two streams by averaging the softmax scores.

*  We uniformly sample a number of frames in each video and the video level prediction is the voting result of all frame level predictions. We pick the starting frame among those early enough to guarantee a desired number of frames. For shorter videos, we looped the video as many times as necessary to satisfy each model’s input interface.

## Results
|Network     |Simonyan et al [[1]](http://papers.nips.cc/paper/5353-two-stream-convolutional) |Ours  |
-------------|:--------------:|:----:|
|Spatial     |72.7%           |73.1% |
|Temporal    |81.0%           |78.8% |
|Fusion      |85.9%           |82.0% |
