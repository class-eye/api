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
#include "student/functions.hpp"
using namespace std;
using namespace cv;

Rect mapping(Rect &rect_ori, Rect &image_rect){

	Rect map_result;
	map_result.x = rect_ori.x + image_rect.x;
	map_result.y = rect_ori.y + image_rect.y;
	map_result.width = rect_ori.width;
	map_result.height = rect_ori.height;
	return map_result;
}

void mapping(std::vector<bbox_t>&result_vec, Rect &image_rect){
	for (int i = 0; i < result_vec.size(); i++){
		if (result_vec[i].obj_id == 0){
			result_vec[i].x += image_rect.x;
			result_vec[i].y += image_rect.y;
		}
	}
}

void de_dup(vector<bbox_t>&a, vector<bbox_t>&b){
	for (auto itera = a.begin(); itera != a.end();){
		if ((*itera).obj_id == 0){
			Rect recta((*itera).x, (*itera).y, (*itera).w, (*itera).h);	
			for (auto iterb = b.begin(); iterb != b.end(); iterb++){
				//cout << "go" << endl;			
				if ((*iterb).obj_id == 0){
					if (itera != a.end()){
						Rect rectb((*iterb).x, (*iterb).y, (*iterb).w, (*iterb).h);
						float IOU = Compute_IOU_area(recta, rectb);
						if (IOU > 0){
							float area = 0;
							if ((*itera).w*(*itera).h >= (*iterb).w*(*iterb).h){
								area = (*iterb).w*(*iterb).h;
								if (IOU / area > 0.7){
									b.erase(iterb);
									itera--;
									//cout << "delete b" << endl;
									break;
								}
							}
							else{
								area = (*itera).w*(*itera).h;
								if (IOU / area > 0.7){
									a.erase(itera);
									//cout << "delete a" << endl;
									itera--;
									break;
								}

							}

						}
					}
				}
				else continue;
			}
			itera++;
		}
		else itera++;
	}
}

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec,cv::Scalar color)
{
	//int const colors[6][3] = { { 1, 0, 1 }, { 0, 0, 1 }, { 0, 1, 1 }, { 0, 1, 0 }, { 1, 1, 0 }, { 1, 0, 0 } };

	for (auto &i : result_vec) {
		if (i.obj_id == 0) {
			if ((float(i.w) / i.h > 2.5) || (float(i.h) / i.w > 2.5))continue;
			Rect rect_ori = cv::Rect(i.x, i.y, i.w, i.h);
			//Rect rect = mapping(rect_ori, rect_image);
			cv::rectangle(mat_img, rect_ori, color, 2);
			//cv::Scalar color = color;
			//cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
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



void yolo_detect(Detector &detector,cv::Mat &image,int &i){
	Timer timer;
	std::vector<bbox_t> result_vec;
	vector<bbox_t>result_vec1;
	vector<bbox_t>result_vec2;
	vector<bbox_t>result_vec3;
	vector<bbox_t>result_vec4;

	Rect rect1(0, 0, image.size().width / 2+100, image.size().height / 2+100);     //left-top
	Rect rect2(0, image.size().height / 2-100, image.size().width / 2+100, image.size().height / 2 +99);    //left-bottom
	Rect rect3(image.size().width / 2-100, 0, image.size().width / 2 - 1+100, image.size().height / 2+100);        //right-top
	Rect rect4(image.size().width / 2-100, image.size().height / 2-100, image.size().width / 2 - 1+100, image.size().height / 2 - 1+100);   //right-bottom
	Mat image1;
	image(rect1).copyTo(image1);
	Mat image2;
	image(rect2).copyTo(image2);
	Mat image3;
	image(rect3).copyTo(image3);
	Mat image4;
	image(rect4).copyTo(image4);

	string s0 = "../yolov3_out/" + to_string(i) + ".jpg";
	string s1 = "../yolov3_out/" + to_string(i) + "left_top.jpg";
	//string s2 = "../yolov3_out/" + to_string(i) + "left_bottom.jpg";
	string s3 = "../yolov3_out/" + to_string(i) + "right_top.jpg";
	//string s4 = "../yolov3_out/" + to_string(i) + "right_bottom.jpg";

	timer.Tic();
	/*result_vec = detector.detect(image, 0.4);
	draw_boxes(image, result_vec, cv::Scalar(0, 0, 255));
	imwrite(s0, image);*/
	result_vec1 = detector.detect(image1,0.5);
	//draw_boxes(image, result_vec1,cv::Scalar(0,0,255),rect1);
	//imwrite(s1, image1);
	result_vec2 = detector.detect(image2,0.5);
	//draw_boxes(image, result_vec2, cv::Scalar(0, 255, 0),rect2);
	//imwrite(s2, image2);
	result_vec3 = detector.detect(image3, 0.5);
	//draw_boxes(image, result_vec3, cv::Scalar(255, 0, 0),rect3);
	//imwrite(s3, image3);
	result_vec4 = detector.detect(image4, 0.5);
	//draw_boxes(image4, result_vec, cv::Scalar(255, 0, 0));
	//imwrite(s4, image4);
	mapping(result_vec1, rect1);
	mapping(result_vec2, rect2);
	mapping(result_vec3, rect3);
	mapping(result_vec4, rect4);
	/*Mat im;
	image.copyTo(im);*/
	//de_dup(result_vec1,result_vec2);
	de_dup(result_vec1, result_vec3);
	de_dup(result_vec1, result_vec2);
	de_dup(result_vec2, result_vec4);
	de_dup(result_vec3, result_vec4);
	de_dup(result_vec1, result_vec4);
	de_dup(result_vec2, result_vec3);
	
	//de_dup(result_vec2, result_vec3);
	draw_boxes(image, result_vec1, cv::Scalar(0, 0, 255));
	draw_boxes(image, result_vec2, cv::Scalar(0, 255, 0));
	draw_boxes(image, result_vec3, cv::Scalar(255, 0, 0));
	draw_boxes(image, result_vec4, cv::Scalar(0, 255, 255));
	//rectangle(image, rect3, Scalar(0, 255, 255));
	imwrite(s0, image);
	/*int count = 0;
	for (auto i : result_vec){
		if (i.obj_id == 0)count++;
	}*/
	timer.Toc();
	cout << /*"Detect: "<<count<<" students"<<*/" Cost " << timer.Elasped() / 1000.0 << " s" << endl;
}