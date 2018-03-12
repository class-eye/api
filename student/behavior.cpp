#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "student/behavior.hpp"
#include "student/functions.hpp"
#include <cmath>
#include <map> 
#include <thread> 
#include <fstream>
#include "caffe/caffe.hpp"

using namespace cv;
using namespace std;


void detect_discontinuous_behavior(Net &net2, Mat &image, PoseInfo &pose, Student_Info &student_info, int &i, int &v, int x[], int y[]){


	//-----------判断双手垂直（为了站立）--------------------

	if (v == 0){
		float angle_r = CalculateVectorAngle(x[2], y[2], x[3], y[3], x[4], y[4]);
		float angle_l = CalculateVectorAngle(x[5], y[5], x[6], y[6], x[7], y[7]);
		bool Vertical_l = false;
		bool Vertical_r = false;
		if (y[4] != 0 && y[7] != 0){
			if ((y[4] > y[3] && y[3] > y[2]) && (y[7] > y[6] && y[6] > y[5])){
				float longer_limb = max(abs(y[4] - y[2]), abs(y[7] - y[5]));
				float shorter_limb = min(abs(y[4] - y[2]), abs(y[7] - y[5]));
				/*float longer_width = max(abs(x[5] - x[2]), abs(x[7] - x[4]));
				float shorter_width = min(abs(x[5] - x[2]), abs(x[7] - x[4]));*/
				//if (shorter_limb / longer_limb > 0.75/* && shorter_width / longer_width > 0.7*/){
				if (abs(y[4] - y[3]) >= abs(x[4] - x[3]) && abs(y[2] - y[3]) >= abs(x[2] - x[3]) && abs(y[7] - y[6]) >= abs(x[7] - x[6]) && abs(y[6] - y[5]) >= abs(x[6] - x[5])){
					if (/*float(y[4] - y[3]) / float(y[3] - y[2]) > 0.7 && */(angle_r > 135 && angle_l > 115) || (angle_l > 135 && angle_r > 115)){
						Vertical_r = true;
					}
				}
				//if (abs(y[7] - y[6]) >= abs(x[7] - x[6]) && abs(y[6] - y[5]) >= abs(x[6] - x[5])){
				//	if (/*float(y[7] - y[6]) / float(y[6] - y[5]) > 0.7 && */(angle_l > 135 && angle_r > 115)){
				//		Vertical_l = true;
				//	}
				//}
			}
		}
		else if (y[4] == 0 && y[7] != 0){
			if (y[7] > y[6] && y[6] > y[5]){
				float angle_l = CalculateVectorAngle(x[5], y[5], x[6], y[6], x[7], y[7]);
				if (abs(y[7] - y[6]) >= abs(x[7] - x[6]) && abs(y[6] - y[5]) >= abs(x[6] - x[5])){
					if (/*float(y[7] - y[6]) / float(y[6] - y[5]) > 0.7 && */(angle_l >= 145)){
						Vertical_l = true;
					}
				}
			}
		}
		else if (y[4] != 0 && y[7] == 0){
			if (y[4] > y[3] && y[3] > y[2]){
				float angle_r = CalculateVectorAngle(x[2], y[2], x[3], y[3], x[4], y[4]);
				if (abs(y[4] - y[3]) >= abs(x[4] - x[3]) && abs(y[2] - y[3]) >= abs(x[2] - x[3])){
					if (/*float(y[4] - y[3]) / float(y[3] - y[2]) > 0.7 && */(angle_r >= 145)){
						Vertical_r = true;
					}
				}
			}
		}
		if ((Vertical_r || Vertical_l)){
			student_info.arm_vertical = true;
			//cv::putText(image, status6, cv::Point2f(x[1], y[1]), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 255, 255), 1);
		}
	}

//---------------------------------判断举手----------------------------------------------
	int symbol_raise_r = 0;
	int symbol_raise_l = 0;
	if (pose.subset[i][0] != -1 || pose.subset[i][1] != -1){
		if (pose.subset[i][4] != -1 && pose.subset[i][3] != -1 && pose.subset[i][2] != -1){
			if (y[4] <= y[3] && y[3] <= y[2] && (y[3] - y[4] > 10)){
				symbol_raise_r = 1;
			}
			else if (y[2] >= y[4])symbol_raise_r = 1;
			else if (y[3] - y[4] >= abs(y[2] - y[4]))symbol_raise_r = 1;
		}
		if (pose.subset[i][7] != -1 && pose.subset[i][6] != -1 && pose.subset[i][5] != -1){
			if (y[7] <= y[6] && y[6] <= y[5] && (y[6] - y[7] > 10)){
				symbol_raise_l = 1;
			}
			else if (y[5] >= y[7])symbol_raise_l = 1;
			else if (y[6] - y[7] >= abs(y[5] - y[7]))symbol_raise_l = 1;
		}
	}
	Rect train;
	if (symbol_raise_r || symbol_raise_l){    //如果举手
		student_info.raising_hand = true;

		if (symbol_raise_r){

			if (y[4] <= y[3] && y[3] <= y[2] && (y[3] - y[4] > 5)){
				student_info.real_raise = true;
				student_info.scores = 1.0;
			}
			else{
				int xg = (x[4] + x[2] + x[3]) / 3;
				int yg = (y[4] + y[2] + y[3]) / 3;
				int heightg;
				int widthg;
				if (x[1] != 0){
					widthg = MIN(abs(x[1] - x[3]), abs(x[1] - x[2]));
				}
				else { widthg = MAX(abs(x[0] - x[3]), abs(x[0] - x[2])); };
				if (x[0] != 0)heightg = abs(y[0] - y[3]);
				else heightg = abs(y[1] - y[3] + 10);
				if (heightg != 0 && widthg != 0){
					train.x = xg - widthg;
					train.y = yg - heightg*1.1;
					train.height = 1.8* heightg;
					train.width = train.height *1.2 / 1.78;
				}
				refine(train, image);
				if (train.height > 0 && train.width > 0){
					Mat img_hand = image(train);
					std::tuple<bool, float> raiseornot = raise_or_not(net2, img_hand);
					student_info.real_raise = get<0>(raiseornot);
					student_info.scores = get<1>(raiseornot);
				}
			}
			if (y[0] > image.size().height / 2){
				if (y[0] >= y[4]){
					student_info.real_raise = true;
					student_info.scores = 1.0;
				}
			}
			else{
				if (y[0] - y[4] > 3){
					student_info.real_raise = true;
					student_info.scores = 1.0;
				}
			}

		}
		if (symbol_raise_l && student_info.real_raise == false){
			if (y[7] <= y[6] && y[6] <= y[5] && (y[6] - y[7] > 5)){
				student_info.real_raise = true;
				student_info.scores = 1.0;
			}
			else{
				int xg = (x[5] + x[6] + x[7]) / 3;
				int yg = (y[5] + y[6] + y[7]) / 3;
				int heightg;
				int widthg;
				if (x[1] != 0){
					widthg = MIN(abs(x[1] - x[6]), abs(x[1] - x[5]));
				}
				else widthg = MAX(abs(x[0] - x[6]), abs(x[0] - x[5]));
				if (x[0] != 0)heightg = abs(y[0] - y[6]);
				else heightg = abs(y[1] - y[6] + 10);

				if (heightg != 0 && widthg != 0){
					train.y = yg - heightg*1.2;
					train.height = 1.8 * heightg;
					train.width = train.height *1.2 / 1.78;
					train.x = xg - train.width / 2;
				}
				refine(train, image);
				if (train.height > 0 && train.width > 0){
					Mat img_hand = image(train);
					std::tuple<bool, float> raiseornot = raise_or_not(net2, img_hand);
					student_info.real_raise = get<0>(raiseornot);
					student_info.scores = get<1>(raiseornot);
				}
			}
			if (y[0] > image.size().height / 2){
				if (y[0] >= y[7]){
					student_info.real_raise = true;
					student_info.scores = 1.0;
				}
			}
			else{
				if (y[0] - y[7] > 3){
					student_info.real_raise = true;
					student_info.scores = 1.0;
				}
			}
		}
	}

	//-----------------判断扭头 判断转身 判断背身（为了讨论）判断低头------------------------------

	if (pose.subset[i][0] != -1 && pose.subset[i][1] != -1 && pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
		if (x[2] < x[5]){
			if (x[0] >= x[5]){
				student_info.turn_head = true;
			}
			else{
				if (abs(x[0] - x[1]) / abs(x[0] - x[5]) > 6)student_info.turn_head = true;
			}
			if (x[0] <= x[2]){
				student_info.turn_head = true;
			}
			else{
				if (abs(x[0] - x[1]) / abs(x[0] - x[2]) > 6)student_info.turn_head = true;
			}
		}
		else{
			student_info.turn_head = true;
		}
	}
	if (pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
		if (abs(y[2] - y[5]) >= abs(x[2] - x[5])){
			student_info.turn_body = true;
			
		}
		if (x[2] >= x[5])student_info.back = true;
		if (pose.subset[i][0] == -1)student_info.bow_head_tmp = true;
	}
	//-------------------判断低头--------------------------
	if (pose.subset[i][0] != -1 && pose.subset[i][1] != -1){
		if (y[1] > image.size().height / 2){
			if (y[0] - y[1] >= 10)student_info.bow_head_tmp = true;
		}
		else if (y[0] > y[1])student_info.bow_head_tmp = true;
	}

}

void Analys_Behavior(vector<vector<Student_Info>>&students_all, vector<vector<Student_Info>>&ID, vector<int>&student_valid, vector<Class_Info> &class_info_all, Mat &image_1080, Mat &image, int &n){
	Class_Info class_info;
	class_info.cur_frame = n;

	string status1 = "Raising hand";
	string status2 = "Standing";
	string status3 = "2-Disscussion";
	string status3back = "4-Disscussion";
	string status4 = "Dazing";
	string status5 = "Bow Head";

	int num_of_back = 0;
	int num_of_disscuss = 0;
	int num_of_bowhead = 0;

	//Mat image;
	//cv::resize(image_1080, image, Size(0, 0), 2 / 3., 2 / 3.);

	for (int j = 0; j < student_valid.size(); j++){

		if (students_all[student_valid[j]][0].cur_size != students_all[student_valid[j]].size()){
			students_all[student_valid[j]][0].cur_size = students_all[student_valid[j]].size();

			int x2 = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].loc.x;
			int y2 = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].loc.y;

			if (students_all[student_valid[j]].size() > 5){
			
				//-------------------收集5s内的信息---------------------------

				//vector<float>nose_range;
				vector<float>box_range;
				vector<float>nose_y;
				for (int k = students_all[student_valid[j]].size() - 5; k < students_all[student_valid[j]].size(); k++){
					//nose_range.push_back(students_all[student_valid[j]][k].loc.x);
					nose_y.push_back(students_all[student_valid[j]][k].neck_loc.y);
					box_range.push_back(students_all[student_valid[j]][k].body_bbox.width);
				}
				/*float max_nose = *max_element(nose_range.begin(), nose_range.end());
				float min_nose = *min_element(nose_range.begin(), nose_range.end());*/
				float max_nose_y = *max_element(nose_y.begin(), nose_y.end());
				float max_width = (*max_element(box_range.begin(), box_range.end())) * 2 / 3;
				//------------------判断讨论-----------------------------

				
				int count1 = 0;
				if (students_all[student_valid[j]].size() > 10){
					for (int k = students_all[student_valid[j]].size() - 10; k < students_all[student_valid[j]].size(); k++){
						if (students_all[student_valid[j]][k].turn_head == true)count1++;
					}
				}
				if (count1 >= 8 || students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].turn_body == true || students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].back == true){
					students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].disscussion = true;
				}
				if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].back == true){
					num_of_back++;
				}
				if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].disscussion == true){
					num_of_disscuss++;
				}
				//}

				//------------------判断低头-----------------------------------
				int count2 = 0;
				if (students_all[student_valid[j]].size() > 10){
					for (int k = students_all[student_valid[j]].size() - 10; k < students_all[student_valid[j]].size(); k++){
						if (students_all[student_valid[j]][k].bow_head_tmp == true)count2++;
					}
				}
				if (count2 >= 7)students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head_each = true;
				if (count2 >= 4)students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head = true;
				if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].back == true)students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head = false;
				if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head == true){
					num_of_bowhead++;
				}

				//-----------------判断起立(3s内)---------------------------
				float thre1;

				if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].loc.y > image.size().height / 2)thre1 = max_width * 1 / 2;
				else thre1 = max_width * 9 / 16.;
				if (max_nose_y - nose_y[nose_y.size() - 1] > thre1){
					if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].arm_vertical == true)
					{
						students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].standing = true;
					}
				}
				if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 2].standing == true && students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].arm_vertical == true){
					float dis = abs(students_all[student_valid[j]][students_all[student_valid[j]].size() - 2].loc.y - students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].loc.y);
					if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].loc.y < image.size().height / 5)thre1 = 10;
					else thre1 = max_width * 9 / 16.;
					if (dis < thre1)students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].standing = true;
				}

				int thre = 4;
				int cur_y = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].loc.y;
				if (cur_y < 200)thre = 3;
				//-------------------累积能量--------------------------------
				Point2f pre_loc = students_all[student_valid[j]][students_all[student_valid[j]].size() - 2].loc;
				Point2f cur_loc = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].loc;
				float distance = euDistance(pre_loc, cur_loc);
				if (distance < thre)students_all[student_valid[j]][0].energy++;
				else students_all[student_valid[j]][0].energy = 0;
				if (students_all[student_valid[j]][0].energy >= 10)line(image, Point2f(x2, y2), Point2f(x2, y2 - students_all[student_valid[j]][0].energy), cv::Scalar(255, 0, 255), 2, 8, 0);
				//cv::putText(image, to_string(students_all[student_valid[j]][0].energy), cv::Point2f(x2 + 8, y2 - students_all[student_valid[j]][0].energy), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);
				if (students_all[student_valid[j]][0].energy > students_all[student_valid[j]][0].max_energy){
					students_all[student_valid[j]][0].max_energy = students_all[student_valid[j]][0].energy;
					students_all[student_valid[j]][0].cur_frame1 = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].cur_frame1;
				}
				//-------------------判断发呆(Ns内)--------------------------

				//if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head == true){
				//	students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].daze = false;
				//}
				//else {
				//	if (students_all[student_valid[j]][0].energy >= 30){
				//		students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].daze = true;
				//		/*vector<float>nose_range_x;
				//		vector<float>nose_range_y;
				//		for (int k = students_all[student_valid[j]].size() - 30; k < students_all[student_valid[j]].size(); k++){
				//		nose_range_x.push_back(students_all[student_valid[j]][k].loc.x);
				//		nose_range_y.push_back(students_all[student_valid[j]][k].loc.y);
				//		}

				//		float max_nose_x = *max_element(nose_range_x.begin(), nose_range_x.end());
				//		float min_nose_x = *min_element(nose_range_x.begin(), nose_range_x.end());
				//		float max_nose_y = *max_element(nose_range_y.begin(), nose_range_y.end());
				//		float min_nose_y = *min_element(nose_range_y.begin(), nose_range_y.end());
				//		if (abs(nose_range_x[nose_range_x.size() - 1] - max_nose_x) < thre && abs(nose_range_x[nose_range_x.size() - 1] - min_nose_x) < thre && abs(nose_range_y[nose_range_y.size() - 1] - max_nose_y) < thre && abs(nose_range_y[nose_range_y.size() - 1] - min_nose_y) < thre){
				//		students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].daze = true;
				//		}*/
				//	}
				//}
			}
			//---------------------------------------------移动人脸识别-----------------------------------------------------
			if (students_all[student_valid[j]][0].matching_at_end != 100){
				if (ID[students_all[student_valid[j]][0].matching_at_end].empty()){
					ID[students_all[student_valid[j]][0].matching_at_end].assign(students_all[student_valid[j]].begin(), students_all[student_valid[j]].end());
				}
				else{
					ID[students_all[student_valid[j]][0].matching_at_end].push_back(students_all[student_valid[j]].back());
				}
			}
			//--------------------------------------------------------------------------------------------------------------

		}
		else{
			students_all[student_valid[j]][0].miss_frame.push_back(n);
			students_all[student_valid[j]][0].away_from_seat++;

			if (students_all[student_valid[j]][0].away_from_seat >= 15){
				students_all[student_valid[j]][0].energy = 0;
				students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head = false;
				students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head_each = false;
				students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].standing = false;
				students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].raising_hand = false;
				students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].daze = false;
				students_all[student_valid[j]][0].away_from_seat = 0;
			}
		}
	}//for (int j = 0; j < student_valid.size(); j++) end

	//----------------------------------------群体行为---------------------------------------------
	if (num_of_back >= 6){
		class_info.all_disscussion_4 = true;
	}
	else if (num_of_disscuss >= 9){
		class_info.all_disscussion_2 = true;
	}
	if (num_of_bowhead >= 10 && num_of_back < 6 && num_of_disscuss < 9){
		class_info.all_bow_head = true;
		for (int j = 0; j < student_valid.size(); j++){
			for (int k = 1; k < students_all[student_valid[j]].size(); k++){
				if (students_all[student_valid[j]][k].cur_frame1 == n){
					students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head = false;
					students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head_each = false;
				}
			}
		}
	}

	////----------------如果讨论--------------------------
	//if (class_info.all_disscussion_2 == true){
	//	cv::putText(image, status3, cv::Point2f(image.size[1] / 2, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
	//}
	//if (class_info.all_disscussion_4 == true){
	//	cv::putText(image, status3back, cv::Point2f(image.size[1] / 2, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
	//}
	////-----------------如果低头-------------------------
	//if (class_info.all_bow_head == true){
	//	cv::putText(image, status5, cv::Point2f(image.size[1] / 2, 70), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
	//}

	//--------------------------------个体行为------------------------------------------------
	for (int j = 0; j < student_valid.size(); j++){
		int x1 = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].neck_loc.x;
		int y1 = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].neck_loc.y;
		//----------------如果发呆----------------------------
		/*if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].daze == true){
		cv::putText(image, status4, cv::Point2f(x1, y1 + 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
		}*/
		////---------------如果起立-----------------------------
		/*if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].standing == true){
			cv::putText(image, status2, cv::Point2f(x1, y1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
			}*/
		////----------------如果低头----------------------------

		//if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].bow_head_each == true){
		//	cv::putText(image, status5, cv::Point2f(x1, y1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
		//}
		//-----------------如果举手-----------------------------

		if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].back == true){
			students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].raising_hand = false;
		}
		/*if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].raising_hand == true){
			int x2 = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].body_for_save.x;
			int y2 = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].body_for_save.y - 10;
			cv::putText(image, to_string(students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].scores), cv::Point2f(x2, y2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 0.7);
			cv::rectangle(image, students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].body_for_save, Scalar(255, 0, 0), 2, 8, 0);
			if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].real_raise == true){
				cv::rectangle(image, students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].body_for_save, Scalar(0, 255, 0), 2, 8, 0);
			}
		}*/
	}



	/*if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].disscussion == true){
	cv::putText(image, status3, cv::Point2f(x1, y1), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 255, 255), 1);
	}*/
	class_info_all.push_back(class_info);

	/*char buff[100];
	if (num_of_bowhead >= 10){
	sprintf(buff, "bow_head: %d", num_of_bowhead);
	cv::putText(image, buff, cv::Point2f(10, image.size().height - 110), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
	}
	else{
	sprintf(buff, "bow_head: %d", 0);
	cv::putText(image, buff, cv::Point2f(10, image.size().height - 110), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
	}
	if (num_of_back >= 7){
	int disscuss_people = 2 * num_of_back;
	sprintf(buff, "4-students'discussion: %d", disscuss_people);
	cv::putText(image, buff, cv::Point2f(10, image.size().height - 50), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
	sprintf(buff, "2-students'discussion: %d", 0);
	cv::putText(image, buff, cv::Point2f(10, image.size().height - 80), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
	}
	else if (num_of_disscuss >= 10){
	int disscuss_people = num_of_disscuss;
	sprintf(buff, "4-students'discussion: %d", 0);
	cv::putText(image, buff, cv::Point2f(10, image.size().height - 50), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
	sprintf(buff, "2-students'discussion: %d", disscuss_people);
	cv::putText(image, buff, cv::Point2f(10, image.size().height - 80), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
	}
	else{
	int disscuss_front = num_of_disscuss;
	int disscuss_back = 2 * num_of_back;
	sprintf(buff, "4-students'discussion: %d", disscuss_back);
	cv::putText(image, buff, cv::Point2f(10, image.size().height - 50), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
	sprintf(buff, "2-students'discussion: %d", disscuss_front);
	cv::putText(image, buff, cv::Point2f(10, image.size().height - 80), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255), 1);
	}*/
}

void good_face(Net &net3, Net &net4, jfda::JfdaDetector &detector, vector<vector<Student_Info>>&students_all, vector<int>&student_valid, int &n, Mat &image_1080, vector<FaceInfo>&standard_faces, int &max_student_num){
	for (int j = 0; j < student_valid.size(); j++){
		if (students_all[student_valid[j]][0].good_face_features.empty()){
			if (students_all[student_valid[j]][0].cur_size != students_all[student_valid[j]].size()){
				if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].face_bbox.width != 0){
					Rect face_bbox_720 = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].face_bbox;
					Rect face_bbox_1080(face_bbox_720.x * 3 / 2, face_bbox_720.y * 3 / 2, face_bbox_720.width * 3 / 2., face_bbox_720.height * 3 / 2);
					Mat faceimg = image_1080(face_bbox_1080);
					vector<FaceInfoInternal>facem;
					vector<FaceInfo> faces = detector.Detect(faceimg, facem);
					if (faces.size() != 0){
						FaceInfo faceinfo = faces[0];
						std::tuple<bool, float>front_face_or_not = is_front_face(net3, faceimg, faceinfo.bbox);
						bool front_face = get<0>(front_face_or_not);
						float sco = get<1>(front_face_or_not);
						/*if (standard_faces.size() == max_student_num - 1){
							if (sco>0.35){
								front_face = true;
							}
						}*/
						if (front_face == true && sco>0.72){

							string output3 = "/home/liaowang/student_api_no_Hik/output_face/" + to_string(student_valid[j]);
							if (!fs::IsExists(output3)){
								fs::MakeDir(output3);
							}
							faceinfo.path = output3;
							string b = output3 + "/" + "0_"+to_string(sco)+"_standard.jpg";
							cv::imwrite(b, faceimg);

							faceinfo.id = student_valid[j];
							Extract(net4, faceimg, faceinfo);
							students_all[student_valid[j]][0].good_face_features.assign(faceinfo.feature.begin(), faceinfo.feature.end());
							standard_faces.push_back(faceinfo);
						}
					}
				}
			}
		}
	}
}

void face_match(Net &net3, Net &net4, jfda::JfdaDetector &detector, vector<vector<Student_Info>>&students_all, vector<int>&student_valid, int &n, Mat &image_1080, vector<FaceInfo>&standard_faces){
	//vector<int>num={ 9, 33, 5, 50, 2, 34, 30, 16, 3, 38, 8, 40, 42, 24, 49, 15, 4, 25, 35, 0, 31, 17, 39 };
	//vector<int>num = { 2,5,50 };
	Timer timer;
	//timer.Tic();
	for (int j = 0; j < student_valid.size(); j++){
		if (students_all[student_valid[j]][0].have_matched == false){
			if (students_all[student_valid[j]][0].cur_size != students_all[student_valid[j]].size()){
				int save_num = 10;
				//if (find(num.begin(), num.end(), student_valid[j]) != num.end()){	

				if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].face_bbox.width != 0){

					Rect face_bbox_720 = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].face_bbox;
					Rect face_bbox_1080(face_bbox_720.x * 3 / 2, face_bbox_720.y * 3 / 2, face_bbox_720.width * 3 / 2., face_bbox_720.height * 3 / 2);
					Mat faceimg = image_1080(face_bbox_1080);

					if (students_all[student_valid[j]][0].matching.size() < save_num){
						vector<FaceInfoInternal>facem;
						vector<FaceInfo> faces = detector.Detect(faceimg, facem);
						if (faces.size() != 0){
							FaceInfo faceinfo = faces[0];

							std::tuple<bool, float>front_face_or_not = is_front_face(net3, faceimg, faceinfo.bbox);
							bool front_face = get<0>(front_face_or_not);
							float sco = get<1>(front_face_or_not);
							if (front_face == true){
								Extract(net4, faceimg, faceinfo);
								std::multimap<float, int, greater<float>>feat_map;
								for (int i = 0; i < standard_faces.size(); i++){
									float distance = 0;
									featureCompare(standard_faces[i].feature, faceinfo.feature, distance);
									feat_map.insert(make_pair(distance, i));
								}
								if (feat_map.begin()->first > 0.5){
									students_all[student_valid[j]][0].matching.push_back(feat_map.begin()->second);
								}
								/*if (student_valid[j] == 35){
									string path35 = "/home/lw/student_api_no_Hik/output_face/" + to_string(n) + ".jpg";
									cv::imwrite(path35, faceimg(faceinfo.bbox));
									cout << "35 face score: " << feat_map.begin()->first << endl;
									}*/
								/*else{
									cout << feat_map.begin()->first << endl;
									}*/
							}
						}

					}
					else if (students_all[student_valid[j]][0].matching.size() == save_num &&students_all[student_valid[j]][0].have_matched == false){
						/*if (student_valid[j] == 52){
							for (int i = 0; i < save_num; i++){
							cout << students_all[student_valid[j]][0].matching[i] << " ";
							}
							cout << endl;
							}*/

						int max_similar = 0;

						for (int i = 0; i < save_num; i++){
							int num = count(students_all[student_valid[j]][0].matching.begin(), students_all[student_valid[j]][0].matching.end(), students_all[student_valid[j]][0].matching[i]);
							if (num > max_similar && num > 1){
								students_all[student_valid[j]][0].matching_at_end = students_all[student_valid[j]][0].matching[i];
							}
						}
						int used = -1;
						for (int k = 0; k < student_valid.size(); k++){
							if (students_all[student_valid[k]][0].match_history.find(students_all[student_valid[j]][0].matching_at_end) != students_all[student_valid[k]][0].match_history.end()){
								used++;
							}
						}

						if (used != -1){
							students_all[student_valid[j]][0].matching_at_end = 100;
							students_all[student_valid[j]][0].matching.erase(students_all[student_valid[j]][0].matching.begin());
						}

						if (students_all[student_valid[j]][0].matching_at_end != 100){
							students_all[student_valid[j]][0].prev_matching_at_end = students_all[student_valid[j]][0].matching_at_end;
							students_all[student_valid[j]][0].have_matched = true;
							students_all[student_valid[j]][0].match_history.insert(make_pair(students_all[student_valid[j]][0].matching_at_end, student_valid[j]));

							string dir1 = standard_faces[students_all[student_valid[j]][0].matching_at_end].path + "/" + to_string(n) + "_" + to_string(student_valid[j]) + ".jpg";
							cv::imwrite(dir1, faceimg);
						}

					}
				}
				//}
			}
		}
	}
	/*timer.Toc();
	cout << "1 cost " << timer.Elasped() / 1000.0 << " s" << endl;*/
}

void renew_face_match(Net &net4, jfda::JfdaDetector &detector, vector<vector<Student_Info>>&students_all, vector<int>&student_valid, int &n, Mat &image_1080, vector<FaceInfo>&standard_faces){
	//vector<int>num={ 9, 33, 5, 50, 2, 34, 30, 16, 3, 38, 8, 40, 42, 24, 49, 15, 4, 25, 35, 0, 31, 17, 39 };
	//vector<int>num = { 2,5,50 };
	Timer timer;
	//timer.Tic();
	for (int j = 0; j < student_valid.size(); j++){	
		if (students_all[student_valid[j]][0].cur_size != students_all[student_valid[j]].size()){	
			int save_num = 4;
			if (students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].face_bbox.width != 0){

				Rect face_bbox_720 = students_all[student_valid[j]][students_all[student_valid[j]].size() - 1].face_bbox;
				Rect face_bbox_1080(face_bbox_720.x * 3 / 2, face_bbox_720.y * 3 / 2, face_bbox_720.width * 3 / 2., face_bbox_720.height * 3 / 2);
				Mat faceimg = image_1080(face_bbox_1080);

				if (students_all[student_valid[j]][0].matching.size() == save_num && students_all[student_valid[j]][0].have_matched){
					vector<FaceInfoInternal>facem;
					vector<FaceInfo> faces = detector.Detect(faceimg, facem);
					if (faces.size() != 0){
						FaceInfo faceinfo = faces[0];
						Extract(net4, faceimg, faceinfo);
						std::multimap<float, int, greater<float>>feat_map;
						for (int i = 0; i < standard_faces.size(); i++){
							float distance = 0;
							featureCompare(standard_faces[i].feature, faceinfo.feature, distance);
							feat_map.insert(make_pair(distance, i));
						}
						if (feat_map.begin()->first > 0.4){					
							students_all[student_valid[j]][0].matching.erase(students_all[student_valid[j]][0].matching.begin());
							students_all[student_valid[j]][0].matching.push_back(feat_map.begin()->second);
		
						}
					}

					int max_similar = 0;
					for (int i = 0; i < save_num; i++){
						int num = count(students_all[student_valid[j]][0].matching.begin(), students_all[student_valid[j]][0].matching.end(), students_all[student_valid[j]][0].matching[i]);
						if (num > max_similar && num>1){
							max_similar = num;
							students_all[student_valid[j]][0].matching_at_end = students_all[student_valid[j]][0].matching[i];
						}
					}
				}
			
			}
			//}
		}
	}
	/*timer.Toc();
	cout << "1 cost " << timer.Elasped() / 1000.0 << " s" << endl;*/

}
