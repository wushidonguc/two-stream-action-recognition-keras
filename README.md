# Two-stream-action-recognition-keras
We use spatial and temporal stream cnn under the Keras framework to reproduce published results on UCF-101 action recognition dataset.

## References
*  [[1] Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional)
*  [[2] Five video classification methods](https://github.com/harvitronix/five-video-classification-methods/blob/master/README.md)


## Data
  ### Spatial input data -> rgb frames
  * First, download the dataset from UCF into the `data` folder:
  `cd data && wget http://crcv.ucf.edu/data/UCF101/UCF101.rar`
  Then extract it with `unrar e UCF101.rar`. in disk, which costs about 5.9G.
  we use split #1 for all of our experiments.
  ### Motion input data -> stacked optical flows
  * Download the preprocessed tvl1 optical flow dataset directly from https://github.com/feichtenhofer/twostreamfusion. 
  ```
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003
  cat ucf101_tvl1_flow.zip* > ucf101_tvl1_flow.zip
  unzip ucf101_tvl1_flow.zip
  ```

## Model
  ### Spatial-stream cnn
  * We classify each video by looking at a single frame.
  ### Temporal-stream cnn
  * We follow the reference paper and use 10 x-channels and 10 y-channels for each optical flow stack, resulting in a input shape of (224, 224, 20).
  
## Training
  ### Spatial-stream cnn
  * We use ImageNet pre-trained models and transfer learning to retrain Inception on our data. We first fine-tune the top dense layers for 10 epochs and then retrain the top two inception blocks.
  ### Temporal-stream cnn
  * We train the temporal-stream cnn from scratch. In every mini-batch, we randomly select 128 (batch size) videos from 9537 training videos and futher randomly select 1 optical flow stack in each video. 
  ### Data augmentation
  * Both streams apply the same data augmentation technique such as corner cropping and random horizontal flipping. Temporally, we pick the starting frame among those early enough to guarantee a desired number of frames. For shorter videos, we looped the video as many times as necessary to satisfy each model’s input interface.

## Testing
  * We fused the two streams by averaging the softmax scores.
  * For every 3783 testing videos, we uniformly sample a number of frames in each video and the video level prediction is the voting result of all frame level predictions. We pick the starting frame among those early enough to guarantee a desired number of frames. For shorter videos, we looped the video as many times as necessary to satisfy each model’s input interface.

## Results
|Network     |Simonyan et al  |Ours  |
-------------|:--------------:|:----:|
|Spatial     |72.7%           |73.1% |
|Temporal    |81.0%           |78.8% |
|Fusion      |85.9%           |82.0% |

## TODOs
- [x] Support multiple workers in the data generator for faster training
- [ ] Initialize temporal-stream cnn by ImageNet pre-trained weights
- [ ] Add a demo script
