#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex> 
#include <condition_variable> 
#include <opencv2/opencv.hpp>		
#include "opencv2/core/version.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "student/yolov3.hpp"
#include "student/fs.hpp"
#include "student/Timer.hpp"
using namespace std;
using namespace cv;

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec,
	int current_det_fps = -1, int current_cap_fps = -1)
{
	int const colors[6][3] = { { 1, 0, 1 }, { 0, 0, 1 }, { 0, 1, 1 }, { 0, 1, 0 }, { 1, 1, 0 }, { 1, 0, 0 } };

	for (auto &i : result_vec) {
		if (i.obj_id == 0) {
			cv::Scalar color = cv::Scalar(0, 0, 255);
			cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
			/*std::string obj_name = "person";
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 30, 0)),
				cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
				color, CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);*/
		}
	}
}

//void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
//	for (auto &i : result_vec) {
//		if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
//		std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
//			<< ", w = " << i.w << ", h = " << i.h
//			<< std::setprecision(3) << ", prob = " << i.prob << std::endl;
//	}
//}

std::vector<std::string> objects_names_from_file(std::string const filename) {
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for (std::string line; getline(file, line);) file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}

void yolo_detect(Detector &detector,cv::Mat &image){
	Timer timer;
	timer.Tic();
	std::vector<bbox_t> result_vec = detector.detect(image);
	draw_boxes(image, result_vec);
	int count = 0;
	for (auto i : result_vec){
		if (i.obj_id == 0)count++;
	}
	timer.Toc();
	cout << "Detect: "<<count<<" students"<<"     Cost: " << timer.Elasped() / 1000.0 << " s" << endl;
}