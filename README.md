# Defne Eken Medtronic Tech Challange

**This repository contains code for Medtronic Tech Challange**.

**The models are modified YOLOV5 models taken from :https://github.com/spmallick/learnopencv/tree/master/Object-Detection-using-YOLOv5-and-OpenCV-DNN-in-CPP-and-Python and the data used for analysis is from COCO 2017 dataset**

## Install Dependancies
```
pip install -r requirements.txt

```
## Install OpenCV and CMake 
The installment steps for windows will be in "Medtro Installment Steps" document.
If you are using Linux system, you can use the tutorial from: https://learnopencv.com/category/install/
If you are using a Linux please uncomment the dirent.h liberary from requirements. Also please make sure you are using a compiler that supports dirent.h.
The list of compilers that support dirent.h are as follows:
Turbo C++ (DOS)
GCC (Cross-platform)

## Clone repo or download the document
Extract the folder into "C:/"
Please make sure the path to your yolov5.py path will be "C:/medtro/yolov5.py" 

## Project Breakdown:
-Yolov5.cpp: C++ code of the YOLO model taken from YOLOv5 github

-Yolov5.py: python code of YOLO model taken from YOLOv5 github

-YOLO5n_vs_YOLO5m.py code developed to compare YOLOv5n adn YOLOV5m models for the 20 images in the 'data' folder. 

-C_vs_Python.py: code developed to compare python and C++ models for the 20 images in the 'data' folder. (both are using YOLOv5s)

-conf_mat.py :code developed to produce confusion matrix,precision and recall. Results have not been found to be reliable.

Note: The C++ codes are ran using the 'CMakeLists.txt' file please modify the 10th line according to the C++ code you would like to run.

Note: Please make sure you run the 'Cpp.cpp, code before you run 'C_vs_Python.py'

Note: For an accurate comparison between C++ and python, please make sure your C++ code is in release mode (the time taken for debugging effects the inference time). Further details can be found in "Medtro Installment Steps" document.

## Results:
The expected results and short discussion can be found in the "Medtro Results" document.






