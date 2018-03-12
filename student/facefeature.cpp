#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <caffe/caffe.hpp>
#include "student/facefeature.hpp"
#include "student/functions.hpp"
using namespace caffe;
using cv::Mat;
using cv::Rect;
using cv::Point2f;
using std::vector;
using std::string;

static inline Point2f RotatePoint(Point2f p, float theta) {
	Point2f op;
	op.x = p.x*cos(theta) - p.y*sin(theta);
	op.y = p.x*sin(theta) + p.y*cos(theta);
	return op;
}
static Mat GetRefinedFace(const cv::Mat& img, const FaceInfo& face, cv::Size kCropSize){
	cv::Mat_<double> ref_pts = (cv::Mat_<double>(5, 2) <<
		30.2946, 51.6963,
		65.5318, 51.5014,
		48.0252, 71.7366,
		33.5493, 92.3655,
		62.7299, 92.2041);
	cv::Mat_<double> landmark = (cv::Mat_<double>(5, 2) <<
		face.landmark[0].x, face.landmark[0].y,
		face.landmark[1].x, face.landmark[1].y,
		face.landmark[2].x, face.landmark[2].y,
		face.landmark[3].x, face.landmark[3].y,
		face.landmark[4].x, face.landmark[4].y);
	/*cv::imshow("origin", img);
	cv::waitKey();*/
	cv::Matx23d tfm = AlignShapeWithScale(landmark, ref_pts);
	cv::Mat aligned_face;
	cv::warpAffine(img, aligned_face, tfm, kCropSize);
	/*cv::imshow("align", aligned_face);
	cv::waitKey();*/
	return aligned_face;
}

//static Mat GetRefinedFace(const cv::Mat& img, const FaceInfo& face, cv::Size kCropSize, int kEcMcY, int kEcY) {
//	// crop the face region
//	Rect roi = face.bbox;
//	roi.x -= roi.width / 2;
//	roi.y -= roi.height / 2;
//	roi.width *= 2;
//	roi.height *= 2;
//	Mat patch = CropPatch(img, roi);
//	//{
//	//  // draw origin image
//	//  Mat tmp = patch.clone();
//	//  for (int i = 0; i < 5; i++) {
//	//    cv::circle(tmp, Point2f(face.landmark5[i].x - roi.x, face.landmark5[i].y - roi.y), 4, cv::Scalar(0, 0, 0), -1);
//	//  }
//	//  cv::imshow("origin", tmp);
//	//  cv::waitKey();
//	//}
//	// get rotation angle
//	const float Pi = CV_PI;
//	float theta = 0.;
//	if (face.landmark[0].x != face.landmark[1].x) {
//		float tan_theta = (face.landmark[1].y - face.landmark[0].y) / (face.landmark[1].x - face.landmark[0].x);
//		theta = atan(tan_theta);
//	}
//	// rotate image
//	Mat rotMat = cv::getRotationMatrix2D(cv::Point2f(patch.cols / 2.f, patch.rows / 2.f), theta / Pi*180.f, 1.f);
//	cv::warpAffine(patch, patch, rotMat, patch.size());
//	// map landmark
//	vector<CvPoint2D32f> landmark(face.landmark);
//	for (int i = 0; i < 5; i++) {
//		// align to patch region
//		landmark[i].x -= roi.x;
//		landmark[i].y -= roi.y;
//		// align to patch center
//		landmark[i].x -= patch.cols / 2.f;
//		landmark[i].y -= patch.rows / 2.f;
//		landmark[i] = RotatePoint(landmark[i], -theta);
//		// align back to patch region
//		landmark[i].x += patch.cols / 2.f;
//		landmark[i].y += patch.rows / 2.f;
//	}
//	//{
//	//  // draw rotated image
//	//  Mat tmp = patch.clone();
//	//  for (int i = 0; i < 5; i++) {
//	//    cv::circle(tmp, landmark[i], 4, cv::Scalar(0, 0, 0), -1);
//	//  }
//	//  cv::imshow("align", tmp);
//	//  cv::waitKey();
//	//}
//	Point2f eye_center, mouth_center;
//	//eye_center.x = patch.cols / 2;
//	eye_center.x = (landmark[0].x + landmark[1].x) / 2.f;
//	eye_center.y = (landmark[0].y + landmark[1].y) / 2.f;
//	mouth_center.x = (landmark[3].x + landmark[4].x) / 2.f;
//	mouth_center.y = (landmark[3].y + landmark[4].y) / 2.f;
//	// crop face
//	float ec_mc_y = mouth_center.y - eye_center.y;
//	float scale = ec_mc_y / kEcMcY;
//	float y1 = kEcY * scale;
//	float y2 = ec_mc_y;
//	float y3 = (kCropSize.height - kEcY - kEcMcY)*scale;
//	float x1 = (kCropSize.width / 2.f)*scale;
//	cv::Size crop_size(kCropSize.width*scale, kCropSize.height*scale);
//	// **TODO** Vertify which one is better, current use 2.
//	// 1. using x of center of eyes
//	// 2. using center of patch
//	// little hack to use 1.
//	// eye_center.x = patch.cols / 2;
//	Rect crop_roi(eye_center.x - x1, eye_center.y - y1, crop_size.width, crop_size.height);
//	patch = CropPatch(patch, crop_roi);
//	cv::resize(patch, patch, kCropSize);
//	//{
//	//  // draw patch
//	//  Mat tmp = patch.clone();
//	//  cv::line(tmp, cv::Point(kCropSize.width / 2, 0), cv::Point(kCropSize.width / 2, kCropSize.height), cv::Scalar(0, 0, 0), 2);
//	//  cv::line(tmp, cv::Point(0, kEcY), cv::Point(kCropSize.width - 1, kEcY), cv::Scalar(0, 0, 0), 2);
//	//  cv::line(tmp, cv::Point(0, kEcY + kEcMcY), cv::Point(kCropSize.width - 1, kEcY + kEcMcY), cv::Scalar(0, 0, 0), 2);
//	//  cv::imshow("res", tmp);
//	//  cv::moveWindow("res", 100, 100);
//	//  cv::waitKey(0);
//	//}
//	return patch;
//}

void Extract(Net &net, const Mat& img, FaceInfo& face){
	// We need color image
	CV_Assert(img.type() == CV_8UC3);
	// align img
	Mat data = GetRefinedFace(img, face, cv::Size(96, 112));
	/*string jiaozheng = "/home/lw/student_api_no_Hik/jiaozheng/0.jpg";
	imwrite(jiaozheng, data);*/
	/*Mat data;
	cv::resize(img, data, cv::Size(96, 112));*/
	Mat data_flip;
	cv::flip(data, data_flip, 1);
	vector<Mat> bgr;
	vector<Mat> bgr_flip;
	data.convertTo(data, CV_32F, 1.0 / 128, -127.5 / 128);
	data_flip.convertTo(data_flip, CV_32F, 1.0 / 128, -127.5 / 128);
	cv::split(data, bgr);
	cv::split(data_flip, bgr_flip);
	shared_ptr<Blob> input = net.blob_by_name("data");
	input->Reshape(2, 3, 112, 96);
	const int bias = input->offset(0, 1);
	const int kBytes = bias * sizeof(float);
	memcpy(input->mutable_cpu_data() + 0 * bias, bgr[0].data, kBytes);
	memcpy(input->mutable_cpu_data() + 1 * bias, bgr[1].data, kBytes);
	memcpy(input->mutable_cpu_data() + 2 * bias, bgr[2].data, kBytes);
	memcpy(input->mutable_cpu_data() + 3 * bias, bgr_flip[0].data, kBytes);
	memcpy(input->mutable_cpu_data() + 4 * bias, bgr_flip[1].data, kBytes);
	memcpy(input->mutable_cpu_data() + 5 * bias, bgr_flip[2].data, kBytes);
	net.Forward();
	shared_ptr<Blob> feature = net.blob_by_name("fc5");

	//vector<float>features;
	const int kFeatureSize = feature->channels();
	face.feature.resize(2 * kFeatureSize);
	for (int i = 0; i < kFeatureSize; i++) {
		face.feature[i] = feature->data_at(0, i, 0, 0);
	}
	for (int i = 0; i < kFeatureSize; i++) {
		face.feature[kFeatureSize + i] = feature->data_at(1, i, 0, 0);
	}
}

//void Extract(Net &net, const Mat& img, FaceInfo& face) {
//	// we need gray image
//	CV_Assert(img.type() == CV_8UC1);
//	//Mat data = GetRefinedFace(img, face, cv::Size(128, 128), 48, 40);
//	Mat data;
//	cv::resize(img, data, cv::Size(128, 128));
//	data.convertTo(data, CV_32F, 1.f / 255.f);
//	shared_ptr<Blob> input = net.blob_by_name("data");
//	const int kBytes = input->offset(1) * sizeof(float);
//	memcpy(input->mutable_cpu_data(), data.data, kBytes);
//	net.Forward();
//	shared_ptr<Blob> feature = net.blob_by_name("eltwise_fc1");
//	const int kFeatureSize = feature->channels();
//	face.feature.resize(kFeatureSize);
//	for (int i = 0; i < kFeatureSize; i++) {
//		face.feature[i] = feature->data_at(0, i, 0, 0);
//	}
//}

//void Extract(Net &net,const Mat& img, vector<FaceInfo>& faces) {
//  // we need gray image
//  CV_Assert(img.type() == CV_8UC1);
//  for (int i = 0; i < faces.size(); i++) {
//    Extract(net,img, faces[i]);
//  }
//}

