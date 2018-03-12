#include <iostream>
#include <string>
#include <cstring> 
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cv.h"  
#include <caffe/caffe.hpp>
#include <thread>
#include "student/student.hpp"
#include "student/functions.hpp"
#include "incCn/HCNetSDK.h"  
#include "incCn/PlayM4.h" 
#include<tuple>
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

int main(){

	/*vector<Class_Info>class_info_all;
	vector<int>student_valid;
	vector<vector<Student_Info>>students_all(70);
	vector<vector<Student_Info>>ID(70);
	vector<FaceInfo>standard_faces;*/
	int max_student_num = 55;
	//int b = 123;
	std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_info;
	/*int n1 = 0, n2 = 0;
	float score_all = 0;*/


	/*if (caffe::GPUAvailable()) {
		caffe::SetMode(caffe::GPU, 0);
	}
	Net net1("../models/pose_deploy.prototxt");
	net1.CopyTrainedLayersFrom("../models/pose_iter_440000.caffemodel");
	Net net2("../models/handsnet.prototxt");
	net2.CopyTrainedLayersFrom("../models/handsnet_iter_12000.caffemodel");
	Net net3("../models/face.prototxt");
	net3.CopyTrainedLayersFrom("../models/face.caffemodel");
	Net net4("../models/facefeature.prototxt");
	net4.CopyTrainedLayersFrom("../models/facefeature.caffemodel");
*/
	Student_analy student("../models/pose_deploy.prototxt", "../models/pose_iter_440000.caffemodel",
						  "../models/handsnet.prototxt", "../models/handsnet_iter_12000.caffemodel", 
						  "../models/face.prototxt", "../models/face.caffemodel", 
						  "../models/facefeature.prototxt", "../models/facefeature.caffemodel",0);


	jfda::JfdaDetector detector("../models/p.prototxt", "../models/p.caffemodel", "../models/r.prototxt", "../models/r.caffemodel", \
		"../models/o.prototxt", "../models/o.caffemodel", "../models/l.prototxt", "../models/l.caffemodel");
	detector.SetMaxImageSize(3000);
	detector.SetMinSize(20);
	detector.SetStageThresholds(0.5, 0.4, 0.55);
	
	string imgdir = "/home/liaowang/student_api/input_test/";
	string output = "/home/liaowang/api_student_class/output2";

#if 1
	vector<string>imagelist = fs::ListDir(imgdir, { "jpg" });
	if (!fs::IsExists(output)){
		fs::MakeDir(output);
	}
	int a = 0;
	for (int i = 640; i < imagelist.size(); i++){

		string imagep = imgdir + imagelist[i];
		Mat image = imread(imagep);
		//if (i < 20){
		PoseInfo pose1;
		cout << "processing: " << i << endl;
		
		if (a != 1){
			a = student.GetStandaredFeats(image, output, max_student_num);
			
			cout << a << endl;
		}
		else{
			cout << "n1 Finish" << endl;
			break;
		}
	}

	for (int i = 0; i < imagelist.size(); i++){
		string imagep = imgdir + imagelist[i];
		Mat image = imread(imagep);
		
		student_info = student.student_detect( detector, image, output,max_student_num);
		/*vector<vector<Student_Info>>students_all = get<0>(student_info);
		vector<Class_Info>class_info_all = get<1>(student_info);*/
	}
#endif
#if 0
	//-------------------------VIDEO---------------------------------------

	string videopath = "../test_face.mp4";
	/*string output = "";
	int max_student_num = 0;*/
	VideoCapture capture(videopath);
	if (!capture.isOpened())
	{
		printf("video loading fail");
	}
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	cout << "all " << totalFrameNumber << " frame" << endl;
	//long frameToStart = 13500; 
	//capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	Mat frame;
	int n = 0;
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
		
		PoseInfo pose1;
		if (n1 != 1){
			if (n % 25 == 0){
				int nn = n / 25;
				int a = GetStandaredFeats(net1, pose1, frame, nn, n1, score_all, output, max_student_num, students_all, student_valid, class_info_all);
			}
		}
		else{
			cout << "n1 Finish" << endl;
			break;
		}
		n++;
	}
	capture.release();

	VideoCapture capture1(videopath);
	if (!capture1.isOpened())
	{
		printf("video loading fail");
	}
	n = 0;
	while (true)
	{
		
		if (!capture1.read(frame)){
			break;
		}
		
		//cv::resize(frame, frame, Size(0, 0), 2 / 3., 2 / 3.);
		PoseInfo pose;
		if (n % 25 == 0){
			int nn = n / 25;
			student_info = student_detect(net1, net2, net3, net4, detector, frame, nn, pose, output, students_all, ID, student_valid, class_info_all, standard_faces, max_student_num);
		}
		/*vector<vector<Student_Info>>students_all = get<0>(student_info);
		vector<Class_Info>class_info_all = get<1>(student_info);*/
		n++;
	}

#endif
}

