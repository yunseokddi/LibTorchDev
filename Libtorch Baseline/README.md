# Libtorch Start code
## Environment
### Python
- python 3.6.8
- pytorch: 1.7.1 (cpu ver)
- OpenCV: 4.4.0

### C++
- LibTorch: 1.7.1 (cpu ver)
- OpenCV: 3.4.13

## How to apply LibTorch?
Visit my blog
[https://ys-cs17.tistory.com/43](https://ys-cs17.tistory.com/43)

## Process
Convert from python weight to c++ weight using **trace.py** → Load converted weight in C++ code using **detect.cpp**

## directory structure
```
+-- c++_code (c++)
	+--Project1
		+-- detect.cpp
+-- python_code (python)
	+-- detect.py
	+-- trace.py
+-- sample_data
	+-- dog.jpg
	+-- cat.jpg
+-- label.txt
```
## File Description
### detect.cpp
- Load converted weight and run detection
### detect.py
- Same to detect.cpp
### trace.py
- Convert from python weight to c++ weight (Using Pretrained ResNet18)

## Result
### python version
![python](./sample_data/python_output.PNG)
### C++ version
![python](./sample_data/c++_output.PNG)

## Gihub link
[https://github.com/yunseokddi/LibTorchDev/tree/master/Libtorch%20Baseline](https://github.com/yunseokddi/LibTorchDev/tree/master/Libtorch%20Baseline)