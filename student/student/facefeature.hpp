#ifndef _FACEFEATURE_H_
#define _FACEFEATURE_H_
#include "student.hpp"
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
using namespace cv;
using namespace caffe;
//struct FaceInfo {
//	cv::Rect bbox;
//	std::vector<cv::Point2f> landmark;
//	std::vector<float> feature;
//	string path;
//};
void Extract(Net &net, const Mat& img,FaceInfo& face);

#endif