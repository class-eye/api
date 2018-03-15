#ifndef _FACE_H_
#define _FACE_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
using namespace cv;
using namespace caffe;

std::tuple<bool, float>is_front_face(Net &net, Mat &oriImg, Rect &face_bbox);
std::tuple<bool, float>real_front_face(Net &net, Mat &oriImg, Rect &face_bbox);
#endif