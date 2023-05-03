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
#include<threshold.h>
using namespace std;
using namespace cv;



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{


    ui->setupUi(this);

}


MainWindow::~MainWindow()
{
    delete ui;
}



QString imgPath;
Mat img;


void MainWindow::on_actionUpload_triggered()
{
    ui->imginput1->clear();
    ui->imgOutput1->clear();

    QString imgPath = QFileDialog::getOpenFileName(this, "Open an Image", "..", "Images (*.png *.xpm *.jpg *.bmb)");

    if(imgPath.isEmpty())
        return;


    img = imread(imgPath.toStdString());
    cv::resize(img, img, Size(512, 512));
    showImg(img, ui->imginput1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
    showImg(img, ui->imginput2, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
    showImg(img, ui->imginput3_1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
}
vector<pair<int, int>>* seedSet;

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
    namedWindow("set seeds", WINDOW_AUTOSIZE); // Create a window
    imshow("set seeds", img_with_seeds); // Show our image inside the created window
    cv::setMouseCallback("set seeds", mouseCallback, &img_with_seeds); // Register the callback function
}


void MainWindow::on_submitBtn_clicked()
{
    Mat im;
    cvtColor(img.clone(), im, COLOR_BGR2GRAY);
    Mat segmented_image = regionGrowing(im, *seedSet, 255, 2);
    showImg(segmented_image, ui->imgOutput1, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());
}


void MainWindow::on_siftBtn_clicked()
{
    Mat im;
    Mat output;
    cvtColor(img.clone(), im, COLOR_BGR2GRAY);
  if(ui->comboBox->currentIndex()==0){
   if(ui->local->isChecked()){
    output=thresholding_local(im,ui->horizontalSlider->value(),"otsu");

   }
   else if(ui->global->isChecked()){
       output=otsu_thresholding(im);

   }
   else{



   }
}
  else if(ui->comboBox->currentIndex()==1){


      if(ui->local->isChecked()){
       output=thresholding_local(im,ui->horizontalSlider->value(),"optimal");

      }
      else if(ui->global->isChecked()){

          output=optimal_thresholding(im);

      }
      else{



      }
   }
    else{
      output=multilevelThresholding(im,ui->horizontalSlider->value());


  }


    showImg(output, ui->imgSift, QImage::Format_RGB888, ui->imginput1->width(), ui->imginput1->height());



}


void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    ui->slider_thres_val->setText(QString::number(value));
}



// Swapping between the 3 pages
void MainWindow::on_pushButton_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
}

void MainWindow::on_pushButton_3_clicked()
{
    ui->stackedWidget->setCurrentIndex(1);
}

void MainWindow::on_pushButton_2_clicked()
{
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_global_toggled(bool checked)
{
        ui->horizontalSlider->hide();
        ui->slider_thres_val->hide();
        ui->label_2->hide();


}


void MainWindow::on_local_toggled(bool checked)
{

    ui->horizontalSlider->show();
    ui->slider_thres_val->show();
    ui->label_2->show();


}


void MainWindow::on_comboBox_currentIndexChanged(int index)
{
    if(ui->comboBox->currentIndex()==2){

        ui->horizontalSlider->show();
        ui->label_multi->setText("num of levels");
        ui->label_2->hide();
        ui->global->hide();
        ui->local->hide();


    }
    else{
          ui->label_multi->hide();
          ui->local->show();
          ui->global->show();
          ui->label_2->show();



    }
}

