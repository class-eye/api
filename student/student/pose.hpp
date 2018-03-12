#ifndef _POSE_H_
#define _POSE_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
using namespace cv;
using namespace caffe;

struct PoseInfo{
	vector<vector<float>>all_peaks;
	vector<vector<float>>candicate;
	vector<vector<float>>subset;
};
PoseInfo pose_detect(Net &net, cv::Mat &oriImg,PoseInfo &pose);
#endif