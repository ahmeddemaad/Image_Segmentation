#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "qlabel.h"
#include <QDebug>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>

using namespace std;
using namespace cv;

// Region growing function
Mat regionGrowing(const Mat& anImage, const vector<pair<int, int>>& aSeedSet, unsigned char anInValue = 255, float tolerance = 5);
void mouseCallback(int event, int x, int y, int flags, void* userdata);
void showImg(cv::Mat& img, QLabel* imgLbl, enum QImage::Format imgFormat, int width , int hieght, bool colorTransform = true);
Mat* setImg(Mat img1);
void getImgLbl(QLabel* imgLbl1);
vector<pair<int, int>>* get_seeds();

#endif // SEGMENTATION_H
