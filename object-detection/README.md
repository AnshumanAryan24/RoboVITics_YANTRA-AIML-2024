# Object detection

## Python Libraries used:
1. [OpenCV Python](https://pypi.org/project/opencv-python/): cv2
2. [OS](https://docs.python.org/3/library/os.html): Built-in library
3. [Keyboard](https://pypi.org/project/keyboard/): Built-in library
4. [PyTorch](https://pypi.org/project/torch/): For loading weights

## YOLO Library used:
[Ultralytics](https://pypi.org/project/ultralytics/)
Models used are _yolov5_ and _yolov8x_.
These can be downladed directly from GitHub repository, or are downloaded automatically as cache (one-time install).

## Steps for installation:
1. Open terminal
2. Type following commands:
```
pip install ultralytics
```
This installs other libraries also, including _PyTorch_, _SciPy_, _cv2_, _Pillow_, _pandas_, _numpy_, etc..

Additional installation instructions:
```
pip install opencv-python
```
```
pip install torch
```

# Information about the files:
### obj_detect.py:
1. This file will be used to demonstrate the use of YOLO model.
2. The _torch.hub_ package is used for loading model and for classification - [check here](https://pytorch.org/docs/stable/hub.html#torch.hub.load).
3. The results are rendered into video using the ```render()``` function of the returned object.
Documentation about the returned object was studied from [here](https://docs.ultralytics.com/modes/predict/#inference-sources).

Recommended to check the Ultralytics Documentation for YOLO-V8 [here](https://docs.ultralytics.com/).
