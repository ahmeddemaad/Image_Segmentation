#include "segmentation.h"


Mat regionGrowing(const Mat& img, const vector<pair<int, int>>& aSeedSet, unsigned char anInValue , float tolerance)
{
    // boolean array/matrix, same size as image
    // all the pixels are initialised to false
    Mat visited_matrix = Mat::zeros(Size(img.cols, img.rows), CV_8UC1);

    // List of points to visit
    vector<pair<int, int>> point_list = aSeedSet;

    while ( ! point_list.empty() )
    {
        // Get a point from the list
        pair<int, int> this_point = point_list.back();
        point_list.pop_back();

        int x = this_point.first;
        int y = this_point.second;
        unsigned char pixel_value = img.at<unsigned char>(Point(x,y));

        // Visit the point
        visited_matrix.at<unsigned char>(Point(x, y)) = anInValue;

        // for each neighbour of this_point
        for (int j = y - 1; j <= y + 1; ++j)
        {
            // vertical index is valid
            if (0 <= j && j < img.rows)
            {
                for (int i = x - 1; i <= x + 1; ++i)
                {
                    // hozirontal index is valid
                    if (0 <= i && i < img.cols)
                    {
                        unsigned char neighbour_value = img.at<unsigned char>(Point(i, j));
                        unsigned char neighbour_visited = visited_matrix.at<unsigned char>(Point(i, j));

                        if (!neighbour_visited &&
                            fabs(neighbour_value - pixel_value) <= (tolerance / 100.0 * 255.0)) // neighbour is similar to this_point
                        {
                            point_list.push_back(pair<int, int>(i, j));
                        }
                    }
                }
            }
        }
    }

    return visited_matrix;
}
Mat imgSeedes;
vector<pair<int, int>> seed_set;

Mat* setImg(Mat img1){
    imgSeedes = img1;
    cvtColor(imgSeedes, imgSeedes, COLOR_BGR2RGB);
    return &imgSeedes;
}
QLabel* imgLbl;

void getImgLbl(QLabel* imgLbl1){
    imgLbl = imgLbl1;
}

vector<pair<int, int>>* get_seeds(){
    return &seed_set;
}

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        seed_set.push_back(pair<int, int>(x, y));
        cv::Scalar colour(0, 0, 255);
        cv::circle(imgSeedes, Point(x, y), 4, colour, FILLED);
        imshow("set seeds", imgSeedes);
        Mat img = imgSeedes.clone();
        showImg(img, imgLbl, QImage::Format_BGR888, imgLbl->width(), imgLbl->height(), false);

    }
}

void showImg(cv::Mat& img, QLabel* imgLbl, enum QImage::Format imgFormat, int width , int hieght, bool colorTransform)
{
    if(colorTransform){
        cvtColor(img, img, COLOR_BGR2RGB);
    }
    QImage image((uchar*)img.data, img.cols, img.rows, imgFormat);
    QPixmap pix = QPixmap::fromImage(image);
    imgLbl->setPixmap(pix.scaled(width, hieght, Qt::KeepAspectRatio));
}
