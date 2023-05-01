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


//@desc maps the image
void showImg(cv::Mat& img, QLabel* imgLbl, enum QImage::Format imgFormat, int width , int hieght, bool colorTransform)
{
    if(colorTransform){
        cvtColor(img, img, COLOR_BGR2RGB);
    }
    QImage image((uchar*)img.data, img.cols, img.rows, imgFormat);
    QPixmap pix = QPixmap::fromImage(image);
    imgLbl->setPixmap(pix.scaled(width, hieght, Qt::KeepAspectRatio));
}



//@desc K-mean Clustring Using Eculdidean distance as a simlarity metric
void kmeans_euclidean(const Mat& X, int K, Mat& idx, Mat& centroids, int max_iters) {
    int m = X.rows;
    int n = X.cols;

    // Randomly initialize K centroids
    RNG rng;
    centroids = Mat::zeros(K, n, CV_32F);
    for (int i = 0; i < K; i++) {
        int row = rng.uniform(0, m);
        X.row(row).copyTo(centroids.row(i));
    }

    Mat previous_centroids;
    for (int iter = 0; iter < max_iters; iter++) {
        // Assign each data point to the closest centroid
        idx = Mat::zeros(m, 1, CV_32S);
        for (int i = 0; i < m; i++) {
            float min_dist = numeric_limits<float>::max();
            int closest_centroid = 0;
            for (int j = 0; j < K; j++) {
                float dist = norm(X.row(i), centroids.row(j));
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }
            idx.at<int>(i, 0) = closest_centroid;
        }

        // Update centroids
        previous_centroids = centroids.clone();
        for (int i = 0; i < K; i++) {
            Mat points;
            for (int j = 0; j < m; j++) {
                if (idx.at<int>(j, 0) == i) {
                    points.push_back(X.row(j));
                }
            }
            if (!points.empty()) {
                Mat1f sum;
                reduce(points, sum, 0, REDUCE_SUM);
                centroids.row(i) = sum / static_cast<float>(points.rows);
            }
        }

        // Check for convergence
        double delta = norm(centroids, previous_centroids);
        if (delta < 1e-5) {
            break;
        }
    }
}

