
#include <QImage>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "segmentation.h"
#include <iostream>
using namespace cv;
Mat optimal_thresholding(Mat image);

Mat thresholding_local(Mat image, int blockSize,std::string type);
Mat otsu_thresholding(Mat image);
Mat multilevelThresholding(Mat inputImage, int numThresholds);
std::vector<double> calculateInitialThresholds(Mat grayImage, int numThresholds);
