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
#include "../student/student/student.hpp"
#include "../student/student/functions.hpp"
#include<tuple>
//#include "../mysql/MyDB.h"

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

	int max_student_num = 56;
	std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_info;
	

	Student_analy student("../models/pose_deploy.prototxt", "../models/pose_iter_440000.caffemodel",
						  "../models/handsnet.prototxt", "../models/handsnet_iter_12000.caffemodel", 
						  "../models/face.prototxt", "../models/face.caffemodel", 
						  "../models/facefeature.prototxt", "../models/facefeature.caffemodel",
						  "../models/f1.prototxt", "../models/f_iter_3000.caffemodel", 0);
	jfda::JfdaDetector detector("../models/p.prototxt", "../models/p.caffemodel", "../models/r.prototxt", "../models/r.caffemodel", \
		"../models/o.prototxt", "../models/o.caffemodel", "../models/l.prototxt", "../models/l.caffemodel");
	detector.SetMaxImageSize(3000);
	detector.SetMinSize(20);
	detector.SetStageThresholds(0.5, 0.4, 0.55);
	/*string imgdir = "/home/liaowang/student_api/input_test/";
	string output = "/home/liaowang/api_student_class/output2";*/
	string imgdir = "../input_test/";
	string output = "../output2";
	Rect box(0, 0, 0, 0);

	/*cout << "0 done" << endl;
	pyMyDB pyDB("localhost", "root", "liuwentong","eye", 3306);
	cout << "0.5 done" << endl;
	pyDB.updateStudentFacePic(1, 2, "/home/lw/1.jpg");
	cout << "1 done" << endl;
	pyDB.updateStudentFacePic(2, 3, "/home/lw/2.jpg");
	cout << "2 done" << endl;
	pyDB.updateStudentFacePic(4, 5, "/home/lw/3.jpg");
	cout << "3 done" << endl;*/

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
#if 1
	//-------------------------VIDEO---------------------------------------

	string videopath = "../ch01_00000000072000000.mp4";
	/*string output = "";
	int max_student_num = 0;*/
	VideoCapture capture(videopath);
	long frameToStart = 40 * 25;
	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
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
			if (n % 25 == 0){
				cout << "processing " << n/25 << " frame" << endl;
				behavior_yes_or_no = student.GetStandaredFeats(frame,output, max_student_num, 0);
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
	/*long frameToStart = 525 * 25;
	capture1.set(CV_CAP_PROP_POS_FRAMES, frameToStart);*/
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
		if (behavior_yes_or_no == 1){
			if (n % 25 == 0){
				student_info = student.student_detect(detector, frame, output, max_student_num);
			}
		}	
		if (n % 25 == 0){
			int finish=student.good_face(detector, frame, max_student_num, 0);
			
			if(finish==1)student.face_match(detector, frame,0);
		
		}
		n++;
	}

#endif
}

