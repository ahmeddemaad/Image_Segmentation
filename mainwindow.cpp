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

