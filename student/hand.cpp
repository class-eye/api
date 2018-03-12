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

std::tuple<bool,float>raise_or_not(Net &net, Mat &oriImg){
	Mat image;
	cv::resize(oriImg, image, cv::Size(40, 60));
	vector<Mat> bgr;
	cv::split(image, bgr);
	bgr[0].convertTo(bgr[0], CV_32F, 1.f, 0.0);
	bgr[1].convertTo(bgr[1], CV_32F, 1.f, 0.0);
	bgr[2].convertTo(bgr[2], CV_32F, 1.f, 0.0);
	shared_ptr<Blob> data = net.blob_by_name("data");
	data->Reshape(1, 3, image.rows, image.cols);
	const int bias = data->offset(0, 1, 0, 0);
	const int bytes = bias*sizeof(float);
	memcpy(data->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
	memcpy(data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
	memcpy(data->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);
	net.Forward();
	shared_ptr<Blob> probs = net.blob_by_name("prob");
	float scores = probs->data_at(0, 1, 0, 0);
	if (scores >= 0.30){
		return std::make_tuple(true,scores);
	}
	else return std::make_tuple(false, scores);
	//const float* probs_out = probs->cpu_data();
	//cout << probs->channels() << endl;	
}
