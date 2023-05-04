#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "qlabel.h"
#include <QDebug>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

// Region growing function
Mat regionGrowing(const Mat& anImage, const vector<pair<int, int>>& aSeedSet, unsigned char anInValue = 255, float tolerance = 5);
void mouseCallback(int event, int x, int y, int flags, void* userdata);
void showImg(cv::Mat& img, QLabel* imgLbl, enum QImage::Format imgFormat, int width , int hieght, bool colorTransform = true);
Mat* setImg(Mat img1);
void getImgLbl(QLabel* imgLbl1);
void kmeans_euclidean(const Mat& X, int K, Mat& idx, Mat& centroids, int max_iters);
vector<pair<int, int>>* get_seeds();
struct Pixel {
    int x;
    int y;
    cv::Vec3b color;

    bool operator==(const Pixel& other) const {
            return x == other.x && y == other.y && color == other.color;
        }
};
Mat mean_shift_segmentation(cv::Mat image, double bandwidth);
double estimateBandwidth(cv::Mat inputImg);

long calcDistance(Vec3b point1, Vec3b point2);
Vec3b mergeClusters(Vec3b c1, Vec3b c2);
Mat agglomerativeSegmentation(Mat original, int numClusters);


#endif // SEGMENTATION_H
