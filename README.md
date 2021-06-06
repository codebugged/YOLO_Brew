## YOLO_Brew
### The most basic implementation of Yolo algorithm using Python framework

#### What is YOLO ?
Yolo is a deep learning algorythm for object detcetion. With yolo we can detect objects at a relatively high speed. With a GPU we would be able to process over 45 frames/second while with a CPU around a frame per second.

#### 3 most used and known frameworks compatible with YOLO and the advantages and disadvantages of each one:

1. Darknet : it’s the framework built from the developer of YOLO and made specifically for yolo.
Advantage: it’s fast, it can work with GPU or CPU
Disadvantage: it olny works with Linux os

2. Darkflow: it’s the adaptation of darknet to Tensorflow (another deep leanring framework).
Advantage: it’s fast, it can work with GPU or CPU, and it’s also compatible with Linux, Windows and Mac.
Disadvantage: the installation it’s really complex, especially on windows

3. Opencv: also opencv has a deep learning framework that works with YOLO. Just make sure you have opencv 3.4.2 at least.
Advantage: it works without needing to install anything except opencv.
Disadvantage: it only works with CPU, so you can’t get really high speed to process videos in real time.


#### To run the algorithm we need three files:

1. Weight file: it’s the trained model, the core of the algorythm to detect the objects.
2. Cfg file: it’s the configuration file, where there are all the settings of the algorythm.
3. Name files: contains the name of the objects that the algorythm can detect.
