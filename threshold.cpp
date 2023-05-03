#include "threshold.h"
#include "mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
Mat thresholding_local(Mat image, int blockSize,std::string type)
{
    // The local Otsu's Function

    // image dimensions
    int width = image.cols;
    int height = image.rows;

    // create a copy of the input image to store the output
    Mat outputImage = image.clone();

    // divide the input image into blocks, including boundary pixels
    for (int i = 0; i < height; i += blockSize) {
        for (int j = 0; j < width; j += blockSize) {

            // adjust the block size for the last blocks in each row and column
            int currentBlockSizeX = (j + blockSize > width) ? (width - j) : blockSize;
            int currentBlockSizeY = (i + blockSize > height) ? (height - i) : blockSize;

            // create a ROI (Region of Interest) for the current block
            Rect roi(j, i, currentBlockSizeX, currentBlockSizeY);
            // extract the current block from the input image
            Mat block = image(roi);
            // apply Otsu thresholding to the current block
            Mat thresholdedBlock = Mat::zeros(block.size(), CV_8UC1);
            if(type=="otsu"){
            thresholdedBlock = otsu_thresholding(block);}
            else if(type=="optimal"){

              thresholdedBlock = optimal_thresholding(block);

            }
            // copy the thresholded block to the output image
            thresholdedBlock.copyTo(outputImage(roi));
        }
    }

    return outputImage;
}


Mat otsu_thresholding(Mat image)
{
    // The Otsu's Function

    // computing the image histogram
    int histogram[256] = { 0 };

    // image dimentions
    long width = image.cols, height = image.rows;
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            histogram[image.at<uchar>(row, col)]++;
        }
    }

    // equation of sigma is sigma = Wb * Wf (Mub - Muf)^2
    // b: background
    // f: foreground
    // sigma: threshold

    int bin = 0;

    double sigma;

    double Wb = 0, Wf = 0, Mub = 0, Muf = 0;

    double Wb_nominator = 0, Mub_nominator = 0, Wf_nominator = 0, Muf_nominator = 0;

    int Mub_denominator = 0, Muf_denominator = 0;

    vector<int> sigmas;

    while (bin < 256)
    {
        for (int i = 0; i <= bin; i++)
        {
            Wb_nominator += histogram[i];
            Mub_nominator += (i * histogram[i]);
            Mub_denominator += histogram[i];
        }
        for (int i = bin + 1; i <= 255; i++)
        {
            Wf_nominator += histogram[i];
            Muf_nominator += (i * histogram[i]);
            Muf_denominator += histogram[i];
        }

        Wb = Wb_nominator / (height * width);
        Mub = Mub_nominator / Mub_denominator;
        Wf = Wf_nominator / (height * width);
        Muf = Muf_nominator / Muf_denominator;

        sigma = round(sqrt(Wb * Wf * (Mub - Muf) * (Mub - Muf)));


        sigmas.push_back(sigma);

        Wb_nominator = 0;
        Mub_nominator = 0;
        Wf_nominator = 0;
        Muf_nominator = 0;
        Mub_denominator = 0;
        Muf_denominator = 0;

        bin++;
    }

    double maxSigma = sigmas[0];


    for (int i = 1; i < sigmas.size(); i++)
    {
        if (sigmas[i] > maxSigma) maxSigma = sigmas[i];
    }

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {

            if (image.at<uchar>(row, col) > maxSigma)
                image.at<uchar>(row, col) = 255;
            else
                image.at<uchar>(row, col) = 0;
        }
    }

    return image;
}


Mat optimal_thresholding(Mat image)
{
    // The function
    bool flag = true;

    // image dimentions
    long width = image.cols, height = image.rows;

    // image initial background
    double background = (image.at<uchar>(0, 0) + image.at<uchar>(0, width - 1) + image.at<uchar>(0, width - 1) + image.at<uchar>(height - 1, width - 1)) / 4;

    // initial object
    // Loop over all the pixels and calculate the sum of their values
    long long sum = 0;
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            if (!(row == 0 && col == 0) || !(row == 0 && col == (height - 1)) || !(row == (height - 1) && col == 0) || !(row == (height - 1) && col == (width - 1)))
                sum += image.at<uchar>(row, col);
        }
    }

    // Calculate the average pixel value
    double object = static_cast<double>(sum) / ((width * height) - 4);

    // initial theshold
    double threshold = (object + background) / 2;

    double previousThreshold = threshold;


    // background less than
    while (flag)
    {
        long long sumBackground = 0, sumObject = 0, counterBackground = 0, counterObject = 0;

        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                if (image.at<uchar>(row, col) < threshold)
                {
                    sumBackground += image.at<uchar>(row, col);
                    counterBackground++;
                }
                else
                {
                    sumObject += image.at<uchar>(row, col);
                    counterObject++;
                }
            }
        }

        background = static_cast<double>(sumBackground) / counterBackground;
        object = static_cast<double>(sumObject) / counterObject;

        threshold = (object + background) / 2;

        cout << object << endl << background << endl << threshold << endl;

        if (previousThreshold = threshold) flag = false;

        else previousThreshold = threshold;
    }

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            if (image.at<uchar>(row, col) > threshold)
                image.at<uchar>(row, col) = 255;
            else
                image.at<uchar>(row, col) = 0;
        }
    }

    return image;
}



Mat multilevelThresholding(Mat inputImage, int numThresholds) {
    cv::Mat grayImage;

        if (inputImage.channels() == 1) {
            grayImage = inputImage.clone();
        } else {
            cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
        }

        std::vector<double> thresholds = calculateInitialThresholds(grayImage, numThresholds);

        cv::Mat outputImage(grayImage.size(), CV_8U);
        for (int y = 0; y < grayImage.rows; ++y) {
            for (int x = 0; x < grayImage.cols; ++x) {
                int intensity = grayImage.at<uchar>(y, x);
                int label = 0;

                for (int i = 0; i < numThresholds; ++i) {
                    if (intensity > thresholds[i]) {
                        label++;
                    } else {
                        break;
                    }
                }

                outputImage.at<uchar>(y, x) = static_cast<uchar>((label * 255) / numThresholds);
            }
        }

        return outputImage;
}


std::vector<double> calculateInitialThresholds(Mat grayImage, int numThresholds) {
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&grayImage, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    std::vector<double> thresholds(numThresholds, 0);
    double sum = grayImage.rows * grayImage.cols;
    double mu = 0;
    for (int i = 1; i < 256; ++i) {
        mu += i * hist.at<float>(i);
    }

    double mu1 = 0;
    double q1 = 0;
    double maxSigma = -1;
    for (int i = 0; i < 256; ++i) {
        mu1 += i * hist.at<float>(i);
        q1 += hist.at<float>(i);

        double q2 = sum - q1;
        if (q1 == 0 || q2 == 0) {
            continue;
        }

        double mu2 = (mu - mu1);
        double sigma = q1 * q2 * pow(mu1 / q1 - mu2 / q2, 2);
        if (sigma > maxSigma) {
            maxSigma = sigma;
            thresholds[0] = i;
        }
    }

    for (int t = 1; t < numThresholds; ++t) {
        thresholds[t] = thresholds[t - 1] + 256.0 / (numThresholds + 1);
    }

    return thresholds;
}

