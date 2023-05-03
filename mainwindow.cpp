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
#include "threshold.h"

using namespace std;
using namespace cv;


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Default Settings of The segmentation Page
    ui->set_seedsBtn->show();
    ui->horizontalSlider->hide();
    ui->horizontalSlider_2->hide();
    ui->slider_thres_val->hide();
    ui->label_3->hide();
    ui->set_seedsBtn->setDisabled(true);
    ui->submitBtn->setDisabled(true);
    ui->ThresholdSubmit->setDisabled(true);
    ui->comboBox->setDisabled(true);
    ui->comboBox_2->setDisabled(true);
    ui->local->setDisabled(true);
    ui->global->setDisabled(true);
    ui->label_8->hide();
    ui->label_2->hide();


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


    // Reading the images & Resizing it to fit Qlabel
    QString imgPath = QFileDialog::getOpenFileName(this, "Open an Image", "..", "Images (*.png *.xpm *.jpg *.bmb)");
    if(imgPath.isEmpty())
        return;
    ui->imginput1->clear();
    ui->imgOutput1->clear();
    ui->imgThreshold->clear();
    ui->set_seedsBtn->setDisabled(false);
    ui->submitBtn->setDisabled(false);
    ui->ThresholdSubmit->setDisabled(false);
    ui->comboBox->setDisabled(false);
    ui->comboBox_2->setDisabled(false);
    ui->local->setDisabled(false);
    ui->global->setDisabled(false);

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
    imshow("set seeds", img.clone());
    // Register the callback function
    cv::setMouseCallback("set seeds", mouseCallback, &img_with_seeds);
}



//@desc Actions on clicking on ReigonGrowing Submit button
void MainWindow::on_submitBtn_clicked()
{


    QString mode = ui->comboBox->currentText();
    if(mode == "Region Growing")
    {
        if(get_seeds()->empty())
        {
            return;
        }
        Mat im;
        cvtColor(img.clone(), im, COLOR_BGR2GRAY);
        Mat segmented_image = regionGrowing(im, *seedSet, 255, 2);
        showImg(segmented_image, ui->imgOutput1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
    }
    else if(mode=="kmeans"){
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
        showImg(bgr_image, ui->imgOutput1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
    }
    else if(mode == "Mean Shift"){
        Mat image = img.clone();
        cv::resize(image, image, cv::Size(), 0.25, 0.25); // resize the image to reduce computation time

        double bandwidth = 45;// bandwidth parameter
        qDebug()<<bandwidth;
        Mat output = mean_shift_segmentation(image, bandwidth);
        showImg(output, ui->imgOutput1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
    }


    else if(mode=="Agglomerative"){
        Mat original = img.clone();
        cv::resize(original,original,cv::Size(60,60));

        Mat result = agglomerativeClustering(original , 3);

        showImg(result, ui->imgOutput1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());

    }
}



//@desc Combox used to Switch different modes between segementation types
void MainWindow::on_comboBox_currentTextChanged(const QString &mode)
{

    if(mode=="kmeans" || mode=="Agglomerative"){
        ui->label_2->show();
        ui->set_seedsBtn->hide();
        ui->horizontalSlider->show();
        ui->label_8->show();
    }

    else if(mode=="Region Growing"){
        ui->label_2->hide();
        ui->set_seedsBtn->show();
        ui->horizontalSlider->hide();
        ui->label_8->hide();
    }
    else if(mode=="Mean Shift")
    {
        ui->label_2->hide();
        ui->set_seedsBtn->hide();
        ui->horizontalSlider->hide();
        ui->label_8->hide();
    }

    ui->imginput1->clear();
    ui->imgOutput1->clear();
    showImg(img, ui->imginput1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());

}




//@desc Number of Clusters used in K-mean Slider
void MainWindow::on_horizontalSlider_sliderMoved(int position)
{
    QString text = QString::number(position);
    ui->label_2->setText(text);
}




void MainWindow::on_ThresholdSubmit_clicked()
{
    Mat im;
        Mat output;
        cvtColor(img.clone(), im, COLOR_BGR2GRAY);
      if(ui->comboBox_2->currentIndex()==0){
       if(ui->local->isChecked()){
        output=thresholding_local(im,ui->horizontalSlider_2->value(),"otsu");

       }
       else if(ui->global->isChecked()){
           output=otsu_thresholding(im);

       }

    }
      else if(ui->comboBox_2->currentIndex()==1){


          if(ui->local->isChecked()){
           output=thresholding_local(im,ui->horizontalSlider_2->value(),"optimal");

          }
          else if(ui->global->isChecked()){

              output=optimal_thresholding(im);

          }

       }
        else{
          output=multilevelThresholding(im,ui->horizontalSlider_2->value());


      }


        showImg(output, ui->imgThreshold, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
}


void MainWindow::on_horizontalSlider_2_valueChanged(int value)
{
    ui->slider_thres_val->setText(QString::number(value));

}


void MainWindow::on_global_toggled(bool checked)
{

    ui->horizontalSlider_2->hide();
    ui->slider_thres_val->hide();
    ui->label_3->hide();

}


void MainWindow::on_local_toggled(bool checked)
{

    ui->horizontalSlider_2->show();
    ui->slider_thres_val->show();
    ui->label_3->show();

}


void MainWindow::on_comboBox_2_currentIndexChanged(int index)
{
    if(index == 2){
        ui->horizontalSlider_2->show();
        ui->slider_thres_val->show();
        ui->label_3->show();
        ui->label_3->setText("Threshold Level");
        ui->local->hide();
        ui->global->hide();
    }
    else{
        ui->label_3->setText("Box Size");
        ui->local->show();
        ui->global->show();
        if(ui->local->isChecked()){
            ui->horizontalSlider_2->show();
            ui->slider_thres_val->show();
            ui->label_3->show();

        }
        else{
            ui->horizontalSlider_2->hide();
            ui->slider_thres_val->hide();
            ui->label_3->hide();

        }
    }
}


void MainWindow::on_pushButton_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}


void MainWindow::on_pushButton_3_clicked()
{
    ui->stackedWidget->setCurrentIndex(1);

}

