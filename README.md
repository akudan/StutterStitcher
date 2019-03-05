# StutterStitcher
![](outputs/dog_run.png) ![](outputs/maria_run.png)
### Requirements
* Python 2.7 or 3.x
* OpenCV 3 (+contrib)
* NumPy
* Matplotlib (for display and debugging)
### Usage
```
usage: stutterstitcher.py [-h] [-n NUM_FOREGROUND] [-r RATE] [-s SCALE]
                          [-o OUTPUT] [--show] [-v]
                          data

positional arguments:
  data                  Path to image folder or video file

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_FOREGROUND, --num-foreground NUM_FOREGROUND
                        Number of foreground objects in scene (default:
                        autodetect)
  -r RATE, --rate RATE  Frame resampling rate (e.g. 3 means keep 1 in 3
                        frames) (default: autodetect)
  -s SCALE, --scale SCALE
                        Scale input images by factor (default: 0.5)
  -o OUTPUT, --output OUTPUT
                        Output filename (default: out.png)
  --show                Display results
  -v, --verbose         Display intermediate results and verbose output

```
Rate is really the most important parameter to set, and should generally be 3-4 for image bursts, or 10-25 for video (these are rough estimates)
### Todo
* Foreground object detection
* Sampling rate estimation
### Credits
* [Sean Chen, Ben Stabler, and Andrew Stanley. 2013. *Automatic Generation of Action Sequence Images from Burst Shots*. Stanford University, Stanford, California.](https://stacks.stanford.edu/file/druid:yt916dh6570/Chen_Stabler_Stanley_Action_Sequence_Generation.pdf)