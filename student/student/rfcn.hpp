#ifndef _RFCN_H_
#define _RFCN_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
using namespace cv;
using namespace caffe;

vector<Rect> im_detect(Net &net, cv::Mat &img/*, Rect &box_pre*/);
//Rect im_detect( cv::Mat &img, int &n, Rect &box_pre);
Rect refine(Mat &img, Rect &box_r);
#endif