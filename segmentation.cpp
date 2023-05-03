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
//    cvtColor(imgSeedes, imgSeedes, COLOR_BGR2RGB);
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
        showImg(img, imgLbl, QImage::Format_RGB888, imgLbl->width(), imgLbl->height());

    }
}


//@desc maps the image
void showImg(cv::Mat& img, QLabel* imgLbl, enum QImage::Format imgFormat, int width , int hieght, bool colorTransform)
{   Mat image1 = img.clone();
    if(colorTransform){

        cvtColor(image1, image1, COLOR_BGR2RGB);
    }
    QImage image((uchar*)image1.data, image1.cols, image1.rows, imgFormat);
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


//mean shift segmentation

Mat mean_shift_segmentation(cv::Mat image, double bandwidth) {
    // convert the input image to the LAB color space
    cv::Mat lab_image;
    cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);

    // initialize the list of pixels to track and the list of segmented pixels
    std::vector<Pixel> pixels;
    std::vector<std::vector<Pixel>> segments;

    // iterate over each pixel in the LAB image and add it to the list of pixels to track
    for (int y = 0; y < lab_image.rows; y++) {
        for (int x = 0; x < lab_image.cols; x++) {
            cv::Vec3b color = lab_image.at<cv::Vec3b>(y, x);
            Pixel pixel = {x, y, color};
            pixels.push_back(pixel);
        }
    }

    // while there are still unassigned pixels
    while (!pixels.empty()) {
        // initialize the mean vector and the total weight
        cv::Vec3d mean(0, 0, 0);
        double total_weight = 0.0;

        // find all pixels within the bandwidth
        std::vector<Pixel> neighbors;
        for (auto& p : pixels) {
            double distance = cv::norm(p.color, pixels[0].color, cv::NORM_L2);
            if (distance < bandwidth) {
                neighbors.push_back(p);
                double weight = std::exp(-std::pow(distance, 2) / (2.0 * std::pow(bandwidth, 2)));
                mean += weight * cv::Vec3d(p.color);
                total_weight += weight;
            }
        }

        // calculate the mean vector
        mean /= total_weight;
        cv::Vec3b mean_color(mean[0], mean[1], mean[2]);

        // assign all neighboring pixels to the same segment
        std::vector<Pixel> segment;
        for (auto& p : neighbors) {
            double distance = cv::norm(p.color, mean_color, cv::NORM_L2);
            if (distance < bandwidth) {
                segment.push_back(p);
            }
        }

        // remove the assigned pixels from the list of pixels to track
        for (auto& p : segment) {
            pixels.erase(std::remove(pixels.begin(), pixels.end(), p), pixels.end());
        }

        // add the segment to the list of segments
        segments.push_back(segment);
    }

    // create a new image with the same dimensions as the input image
    cv::Mat segmented_image(image.size(), image.type(), cv::Scalar(0, 0, 0));

    // assign a random color to each segment
    for (size_t i = 0; i < segments.size(); i++) {
        cv::Vec3b color(rand() % 256, rand() % 256, rand() % 256);
        for (auto& p : segments[i]) {
            segmented_image.at<cv::Vec3b>(p.y, p.x) = color;
        }
    }

    return segmented_image;
}

double estimateBandwidth(cv::Mat inputImg){
    cv::Mat outputImg;

    // Convert input image to Lab color space
    cv::cvtColor(inputImg, outputImg, cv::COLOR_BGR2Lab);

    // Calculate the standard deviation of the input image
    double sigma = cv::mean(cv::mean(cv::Mat(cv::abs(outputImg - cv::Scalar::all(128)))))[0];

    // Calculate the number of pixels in the input image
    float n = static_cast<float>(outputImg.rows * outputImg.cols);

    // Estimate the bandwidth using Silverman's rule of thumb
    double bandwidth = 1.06f * sigma * pow(n, -1.0f/5.0f);

    return bandwidth;
}


