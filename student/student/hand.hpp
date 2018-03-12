#ifndef _HAND_H_
#define _HAND_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
using namespace cv;
using namespace caffe;


std::tuple<bool, float>raise_or_not(Net &net, Mat &oriImg);
#endif