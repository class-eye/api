#include <iostream>
#include <string>
//#include <cstring> 
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cv.h"  
#include <caffe/caffe.hpp>
#include <thread>
#include "../student/student/student.hpp"
#include "../student/student/functions.hpp"
#include<tuple>
#include "../student/student/rfcn.hpp"
#include "../student/student/yolov3.hpp"
//#include "../MyDB/MyDB.h"

using namespace std;
using namespace cv;
using namespace caffe;
using namespace fs;


void initValue(int &n, int &max_student_num, vector<Class_Info>&class_info_all, vector<int>&student_valid, vector<vector<Student_Info>>&students_all){
	n = 0;
	max_student_num = 0;
	student_valid.clear();
	for (int i = 0; i < 70; i++){
		students_all[i].clear();
	}
	class_info_all.clear();
}

//-------------------------------------------------OpenCV------------------------------------------------------

void draw_pose(Net &net, Mat &image){
	PoseInfo pose;
	//Mat img;
	//resize(image, img, Size(0, 0), 2 / 3., 2 / 3.);
	pose_detect(net, image, pose);
	int color[18][3] = { { 255, 0, 0 }, { 255, 85, 0 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 170, 0, 255 }, { 170, 0, 255 }, { 255, 0, 170 }, { 255, 0, 170 } };
	int x[18];
	int y[18];
	cout << pose.subset.size() << endl;
	for (int i = 0; i < pose.subset.size(); i++){
		float score = float(pose.subset[i][18]) / pose.subset[i][19];
		if (pose.subset[i][19] >= 4 && score >= 0.4){
			for (int j = 0; j < 8; j++){
				if (pose.subset[i][j] == -1){
					x[j] = 0;
					y[j] = 0;
				}
				else{
					x[j] = pose.candicate[pose.subset[i][j]][0];
					y[j] = pose.candicate[pose.subset[i][j]][1];
				}
			}
		}
		for (int j = 0; j < 8; j++){
			if (!(x[j] || y[j])){
				continue;
			}
			else{
				cv::circle(image, Point2f(x[j], y[j]), 3, cv::Scalar(color[j][0], color[j][1], color[j][2]), -1);
			}
		}
	}
}

int main(){
	//54_chinese_1102_4_0.mp4   202帧   54个人
	//ch01_00000000072000000.mp4  99帧  56个人
	int max_student_num = 35;
	std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_info;
	

	/*Student_analy student("../models/pose_deploy.prototxt", "../models/pose_iter_440000.caffemodel",
						  "../models/handsnet.prototxt", "../models/handsnet_iter_12000.caffemodel", 
						  "../models/face.prototxt", "../models/face.caffemodel", 
						  "../models/facefeature.prototxt", "../models/facefeature.caffemodel",
						  "../models/deploy_simple.prototxt", "../models/eight_net_iter_2626.caffemodel", 0);
						
	jfda::JfdaDetector detector("../models/p.prototxt", "../models/p.caffemodel", "../models/r.prototxt", "../models/r.caffemodel", \
		"../models/o.prototxt", "../models/o.caffemodel", "../models/l.prototxt", "../models/l.caffemodel");
	detector.SetMaxImageSize(3000);
	detector.SetMinSize(20);
	detector.SetStageThresholds(0.5, 0.4, 0.55);*/
	int gpu_device = 1;
	if (caffe::GPUAvailable()){
		cout << "GPU Mode" << endl;
		caffe::SetMode(caffe::GPU, gpu_device);
	}
	std::string  cfg_file = "../yolov3.cfg";
	std::string  weights_file = "../yolov3.weights";
	Detector yolo_detector(cfg_file, weights_file, gpu_device);

	Net net("../models/pose_deploy.prototxt");
	net.CopyTrainedLayersFrom("../models/pose_iter_440000.caffemodel");

	string imgdir = "../input_test/";
	string output = "../output2";
	Rect box(0, 0, 0, 0);

	/*cout << "0 done" << endl;
	pyMyDB pyDB("192.168.66.12", "root", "wst123456","intelleye", 3306);
	cout << "0.6 done" << endl;
	pyDB.updateStudentFacePic(1, 2, "../output2/00.jpg");
	cout << "1 done" << endl;
	pyDB.updateStudentFacePic(4, 5, "../output2/02.jpg");
	cout << "3 done" << endl;
	pyDB.updateClassroomPic(2, "../output2/01.jpg");
	cout << "2 done" << endl;*/
	

#if 1
	Mat img;
	string imgdir1 = "/home/lw/student_api/inputimg/";
	vector<string>imagelist = fs::ListDir(imgdir1, { "jpg" });
	for (int i = 50; i < imagelist.size(); i++){
		string imagep = imgdir1 + imagelist[i];
		cout << imagep << endl;
		Mat image = imread(imagep);
		image.copyTo(img);
		//resize(image, image, Size(0, 0), 2 / 3., 2 / 3.);	
		yolo_detect(yolo_detector, image,i);
		draw_pose(net, img);
		string out = "../yolov3_out/" + to_string(i) + "_pose.jpg";
		imwrite(out, img);
	}
#endif

#if 0
	vector<string>imagelist = fs::ListDir(imgdir, { "jpg" });
	if (!fs::IsExists(output)){
		fs::MakeDir(output);
	}
	int behavior_yes_or_no = 0;
	for (int i = 640; i < imagelist.size(); i++){

		string imagep = imgdir + imagelist[i];
		Mat image = imread(imagep);
		//if (i < 20){
		PoseInfo pose1;
		cout << "processing: " << i << endl;


		if (behavior_yes_or_no != 1){
			behavior_yes_or_no = student.GetStandaredFeats(image, output, max_student_num,0);
		}
		else{
			cout << "n1 Finish" << endl;
			break;
		}
	}
	for (int i = 0; i < imagelist.size(); i++){
		string imagep = imgdir + imagelist[i];
		Mat image = imread(imagep);
		if (behavior_yes_or_no == 1){
			student_info = student.student_detect(detector, image, output, max_student_num);
		}
		student.good_face(detector,image,max_student_num,0);
		student.face_match(detector,image);
		/*vector<vector<Student_Info>>students_all = get<0>(student_info);
		vector<Class_Info>class_info_all = get<1>(student_info);*/
	}
#endif
#if 0
	//-------------------------VIDEO---------------------------------------

	string videopath = "../sw180402_1_.mp4";
	/*string output = "";
	int max_student_num = 0;*/
	VideoCapture capture(videopath);
	/*long frameToStart = 100 * 10;
	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);*/
	if (!capture.isOpened())
	{
		printf("video loading fail");
	}
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	cout << "all " << totalFrameNumber << " frame" << endl;
	
	Mat frame;
	int n = 0;
	int behavior_yes_or_no = 0;
	//std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_info;
	while (true)
	{
		
		if (!capture.read(frame)){
			
			break;
		}
		/*if (n % 25 == 0){
			string output_c = "/home/liaowang/student_api_no_Hik/inputimg/" + to_string(n) + ".jpg";
			imwrite(output_c, frame);
		}*/
		if (behavior_yes_or_no != 1){
			if (n % 10 == 0){
				cout << "处理 " << n/10 << " 帧aaa" << endl;
				behavior_yes_or_no = student.GetStandaredFeats(frame,output, max_student_num, 0);
			}
		}
		else{
			if (n % 10 == 0){
				student_info = student.student_detect(detector, frame, output, max_student_num);
				int finish=student.good_face(detector, frame, max_student_num, 0);

				if (finish == 1){
					student.face_match(detector, frame, max_student_num, 0);
					//student.face_vote(frame, max_student_num);
				}
			}
		}
		n++;
		/*if (n == 2000)
		{
			cout << "------------------------------下一个视频--------------------------------------" << endl;
			break;
		}*/
	}
	capture.release();

	max_student_num = 32;
	string videopath1 = "../sw180402_2.mp4";
	//max_student_num = 52;
	VideoCapture capture1(videopath1);
	/*long frameToStart = 45 * 25;
	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);*/
	if (!capture1.isOpened())
	{
		printf("video loading fail");
	}
	totalFrameNumber = capture1.get(CV_CAP_PROP_FRAME_COUNT);
	cout << "all " << totalFrameNumber << " frame" << endl;
	n = 0;
	behavior_yes_or_no = 0;
	//std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_info;
	student.clear();
	while (true)
	{
		if (!capture1.read(frame)){
			break;
		}
		/*if (n % 25 == 0){
		string output_c = "/home/liaowang/student_api_no_Hik/inputimg/" + to_string(n) + ".jpg";
		imwrite(output_c, frame);
		}*/

		

		if (behavior_yes_or_no != 1){
			if (n % 10 == 0){
				cout << "处理 " << n / 10 << " 帧bbb" << endl;
				behavior_yes_or_no = student.GetStandaredFeats(frame, output, max_student_num, 0);
			}
		}
		else{
			if (n % 10 == 0){
				student_info = student.student_detect(detector, frame, output, max_student_num);
				/*if (finish == 1)*/student.face_match(detector, frame, max_student_num, 0);
				student.face_vote(frame, max_student_num);
			}
		}
		n++;
	
	}
	capture1.release();

#endif
}

