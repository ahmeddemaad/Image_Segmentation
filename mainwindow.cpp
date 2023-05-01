#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "segmentation.h"
#include <iostream>

using namespace std;
using namespace cv;


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Default Settings of The segmentation Page
    ui->set_seedsBtn->show();
    ui->submitBtn->show();
    ui->kmeansBtn->hide();
    ui->horizontalSlider->hide();
}


MainWindow::~MainWindow()
{
    delete ui;
}


// Global Variables
QString imgPath;
Mat img;
vector<pair<int, int>>* seedSet;


//@desc actions done on clicking Upload button
void MainWindow::on_actionUpload_triggered()
{
    // Clearing Previous images
    ui->imginput1->clear();
    ui->imgOutput1->clear();

    // Reading the images & Resizing it to fit Qlabel
    QString imgPath = QFileDialog::getOpenFileName(this, "Open an Image", "..", "Images (*.png *.xpm *.jpg *.bmb)");
    if(imgPath.isEmpty())
        return;
    img = imread(imgPath.toStdString(),IMREAD_COLOR);
    cv::resize(img, img, Size(512, 512));

    // Showing the input image uploaded in RGB format
    showImg(img, ui->imginput1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
    showImg(img, ui->imginput2, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
    showImg(img, ui->imginput3_1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
}


void MainWindow::on_set_seedsBtn_clicked()
{
    ui->imgOutput1->clear();
    if(!get_seeds()->empty())
    {
        seedSet->clear();
    }
    seedSet = get_seeds();
    setImg(img.clone());
    Mat img_with_seeds;
    cvtColor(img, img_with_seeds, COLOR_BGR2RGB);
    getImgLbl(ui->imginput1);
    // Create a window
    namedWindow("set seeds", WINDOW_AUTOSIZE);
    // Show our image inside the created window
    imshow("set seeds", img_with_seeds);
    // Register the callback function
    cv::setMouseCallback("set seeds", mouseCallback, &img_with_seeds);
}



//@desc Actions on clicking on ReigonGrowing Submit button
void MainWindow::on_submitBtn_clicked()
{
    Mat im;
    cvtColor(img.clone(), im, COLOR_BGR2GRAY);
    Mat segmented_image = regionGrowing(im, *seedSet, 255, 2);
    showImg(segmented_image, ui->imgOutput1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
}



//@desc Combox used to Switch different modes between segementation types
void MainWindow::on_comboBox_currentTextChanged(const QString &mode)
{

    if(mode=="kmeans"){
        ui->set_seedsBtn->hide();
        ui->submitBtn->hide();
        ui->kmeansBtn->show();
        ui->horizontalSlider->show();
        ui->imginput1->clear();
        ui->imgOutput1->clear();
    }
    if(mode=="RegionGrowing"){
        ui->set_seedsBtn->show();
        ui->submitBtn->show();
        ui->kmeansBtn->hide();
        ui->horizontalSlider->hide();
        ui->imginput1->clear();
        ui->imgOutput1->clear();
    }
}


//@desc Performing K-mean of the Image on clicking on K-mean submitBtn
void MainWindow::on_kmeansBtn_clicked()
{
       //Reading & Resizing The Image As a Colored RGB image
       Mat image = img.clone();

       // Convert the image to LUV color space
       Mat luv_image;
       cvtColor(image, luv_image, COLOR_BGR2Luv);

       // Convert the image to a 2D matrix for k-means clustering
       Mat data;
       luv_image.convertTo(data, CV_32F);
       data = data.reshape(1, data.rows * data.cols);

       // Perform k-means clustering
       int num_clusters = ui->horizontalSlider->value();
       Mat labels, centers;
       kmeans_euclidean(data, num_clusters, labels, centers, 5);

       // Reshape the labels and centers to match the LUV image size
       labels = labels.reshape(1, luv_image.rows);
       centers = centers.reshape(1, num_clusters);

       // Generate the segmented image
       Mat segmented_image(luv_image.size(), luv_image.type());
       for (int i = 0; i < luv_image.rows; i++) {
           for (int j = 0; j < luv_image.cols; j++) {
               int label = labels.at<int>(i, j);
               segmented_image.at<Vec3b>(i, j) = centers.at<Vec3f>(label, 0);
           }
       }

       // Convert the segmented image back to BGR color space
       Mat bgr_image;
       cvtColor(segmented_image, bgr_image, COLOR_Luv2BGR);

       // Display the original and segmented images
       showImg(bgr_image, ui->imgOutput1, QImage::Format_BGR888, ui->imginput1->width(), ui->imginput1->height());
}


//@desc Number of Clusters used in K-mean Slider
void MainWindow::on_horizontalSlider_sliderMoved(int position)
{
    QString text = QString::number(position);
    ui->label_2->setText(text);
}
