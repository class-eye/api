#include <vector>
#include <algorithm>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include<opencv2/gpu/gpu.hpp>
#include "student/student.hpp"
#include "student/functions.hpp"
#include <numeric>

using namespace cv;
using namespace std;
using namespace caffe;



std::tuple<bool, float>is_front_face(Net &net, Mat &oriImg, Rect &face_bbox){
	shared_ptr<Blob> data = net.blob_by_name("data");
	data->Reshape(1, 3, 32, 32);
	cv::Mat patch = CropPatch(oriImg, face_bbox);
	cv::resize(patch, patch, cv::Size(32, 32));
	vector<cv::Mat> bgr;
	cv::split(patch, bgr);
	bgr[0].convertTo(bgr[0], CV_32F, 1.f / 128.f, -1.f);
	bgr[1].convertTo(bgr[1], CV_32F, 1.f / 128.f, -1.f);
	bgr[2].convertTo(bgr[2], CV_32F, 1.f / 128.f, -1.f);
	const int bias = data->offset(0, 1, 0, 0);
	const int bytes = bias*sizeof(float);
	// this model uses rgb
	memcpy(data->mutable_cpu_data() + 0 * bias, bgr[2].data, bytes);
	memcpy(data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
	memcpy(data->mutable_cpu_data() + 2 * bias, bgr[0].data, bytes);
	net.Forward();
	shared_ptr<Blob> prob = net.blob_by_name("prob");
	float scores = prob->data_at(0, 1, 0, 0);
	if (scores >= 0.5){
		return std::make_tuple(true, scores);
	}
	else return std::make_tuple(false, scores);
}
std::tuple<bool, float>real_front_face(Net &net, Mat &oriImg, Rect &face_bbox){
	shared_ptr<Blob> data = net.blob_by_name("data");
	data->Reshape(1, 3, 32, 32);
	cv::Mat patch = CropPatch(oriImg, face_bbox);
	cv::resize(patch, patch, cv::Size(32, 32));
	vector<cv::Mat> bgr;
	cv::split(patch, bgr);
	bgr[0].convertTo(bgr[0], CV_32F, 1.f / 128.f, -1.f);
	bgr[1].convertTo(bgr[1], CV_32F, 1.f / 128.f, -1.f);
	bgr[2].convertTo(bgr[2], CV_32F, 1.f / 128.f, -1.f);
	/*bgr[0].convertTo(bgr[0], CV_32F, 1.f, 0.0);
	bgr[1].convertTo(bgr[1], CV_32F, 1.f, 0.0);
	bgr[2].convertTo(bgr[2], CV_32F, 1.f, 0.0);*/
	
	const int bias = data->offset(0, 1, 0, 0);
	const int bytes = bias*sizeof(float);

	memcpy(data->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
	memcpy(data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
	memcpy(data->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);
	net.Forward();
	shared_ptr<Blob> prob = net.blob_by_name("prob");
	float scores = prob->data_at(0, 1, 0, 0);
	if (scores >= 0.8){
		return std::make_tuple(true, scores);
	}
	else return std::make_tuple(false, scores);
}
