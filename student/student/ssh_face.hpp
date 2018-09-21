#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/caffe.hpp>
#include <memory>
#include "student.hpp"
using namespace caffe;
using namespace std;
struct BBox {
	float x1, y1, x2, y2;
};
struct Config {
	int min_size = 720;
	int max_size = 1280;
	/*int min_size = 1080;
	int max_size = 1920;*/
	float nms_th = 0.3;
	float score_th = 0.4;
	bool square_box = true;
	std::string model_dir = "/home/liaowang/api_student_class/models";
	std::string net_prefix = "ssh";
	int gpu_id = 1;

	void Check() const;
};
class SshFaceDetWorker{
public:
	SshFaceDetWorker(const string& ssh_net, const string &ssh_model,
		const string& real_front_face_net, const string& real_front_face_model,
		const string& face_feature_net, const string& face_feature_model,
		const string& pose_net, const string& pose_model);
	~SshFaceDetWorker();
	
	void SetConfig(Config config);
	/*void Initialize();
	void Deinitialize();*/
	vector<BBox>detect(cv::Mat img);
	void GetStandaredFeats_ssh(vector<BBox>faces, jfda::JfdaDetector &detector, Mat &frame_1080);
	int good_face_ssh(vector<BBox>faces, jfda::JfdaDetector &detector, Mat &image_1080);
	int face_match(vector<BBox>faces, jfda::JfdaDetector &detector, Mat &image_1080);
private:
	Config config_;
	Net* net_;
	Net *real_frontface_net;
	Net *facefeature_net;
	Net *pose_net;

	vector<FaceInfo>standard_faces;
	vector<vector<FaceInfo>>matching;
	int n = 0;
};