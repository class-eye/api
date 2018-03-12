#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cv.h"  

#include "student/functions.hpp"
#include<cmath>
#ifdef __unix__
#include <json/json.h>
//#include <python2.7/Python.h>
#endif
using namespace cv;
using namespace std;

void refine(Rect& bbox, cv::Mat& img)
{
	if (bbox.x < 0 && bbox.y < 0 && 0 < bbox.x + bbox.width < img.size[1] && 0 < bbox.y + bbox.height < img.size[0]){
		float a = bbox.x;
		float  b = bbox.y;
		bbox.x = 0;
		bbox.y = 0;
		bbox.width = bbox.width + a;
		bbox.height = bbox.height + b;
	}
	if (bbox.x < 0 && 0 < bbox.y < img.size[0] && 0 < bbox.x + bbox.width<img.size[1] && bbox.y + bbox.height>img.size[0]){
		float  a = bbox.x;
		bbox.x = 0;
		bbox.width = bbox.width + a;
		bbox.height = img.size[0] - bbox.y;
	}
	if (0 < bbox.x < img.size[1] && bbox.y<0 && bbox.x + bbox.width>img.size[1] && 0 < bbox.y + bbox.height < img.size[0]){
		float  a = bbox.y;
		bbox.y = 0;
		bbox.width = img.size[1] - bbox.x;
		bbox.height = bbox.height + a;
	}
	if (0 < bbox.x < img.size[1] && 0 < bbox.y<img.size[0] && bbox.x + bbox.width>img.size[1] && bbox.y + bbox.height > img.size[0]){
		bbox.width = img.size[1] - bbox.x;
		bbox.height = img.size[0] - bbox.y;
	}
	if (bbox.x < 0 && 0 < bbox.y < img.size[0] && 0 < bbox.x + bbox.width < img.size[1] && 0 < bbox.y + bbox.height < img.size[0]){
		float  a = bbox.x;
		bbox.x = 0;
		bbox.width = bbox.width + a;
	}
	if (0 < bbox.x < img.size[1] && 0 < bbox.y<img.size[0] && bbox.x + bbox.width>img.size[1] && 0 < bbox.y + bbox.height < img.size[0]){
		bbox.width = img.size[1] - bbox.x;
	}
	if (0 < bbox.x < img.size[1] && bbox.y < 0 && 0 < bbox.x + bbox.width < img.size[1] && 0 < bbox.y + bbox.height < img.size[0]){
		float  a = bbox.y;
		bbox.y = 0;
		bbox.height = bbox.height + a;
	}
	if (0 < bbox.x < img.size[1] && 0 < bbox.y < img.size[0] && 0 < bbox.x + bbox.width<img.size[1] && bbox.y + bbox.height>img.size[0]){
		bbox.height = img.size[0] - bbox.y;
	}

}
cv::Mat CropPatch(const cv::Mat& img, cv::Rect& bbox) {
	int height = img.rows;
	int width = img.cols;
	int x1 = bbox.x;
	int y1 = bbox.y;
	int x2 = bbox.x + bbox.width;
	int y2 = bbox.y + bbox.height;
	cv::Mat patch = cv::Mat::zeros(bbox.height, bbox.width, img.type());
	// something stupid, totally out of boundary
	if (x1 >= width || y1 >= height || x2 <= 0 || y2 <= 0) {
		return patch;
	}
	// partially out of boundary
	if (x1 < 0 || y1 < 0 || x2 > width || y2 > height) {
		int vx1 = (x1 < 0 ? 0 : x1);
		int vy1 = (y1 < 0 ? 0 : y1);
		int vx2 = (x2 > width ? width : x2);
		int vy2 = (y2 > height ? height : y2);
		int sx = (x1 < 0 ? -x1 : 0);
		int sy = (y1 < 0 ? -y1 : 0);
		int vw = vx2 - vx1;
		int vh = vy2 - vy1;
		cv::Rect roi_src(vx1, vy1, vw, vh);
		cv::Rect roi_dst(sx, sy, vw, vh);
		img(roi_src).copyTo(patch(roi_dst));
	}
	else {
		img(bbox).copyTo(patch);
	}
	return patch;
}
cv::Matx22d AlignShapesKabsch2D(const cv::Mat_<double>& align_from, const cv::Mat_<double>& align_to)
{
	cv::SVD svd(align_from.t() * align_to);
	// make sure no reflection is there
	// corr ensures that we do only rotaitons and not reflections
	double d = cv::determinant(svd.vt.t() * svd.u.t());
	cv::Matx22d corr = cv::Matx22d::eye();
	if (d > 0){
		corr(1, 1) = 1;
	}
	else{
		corr(1, 1) = -1;
	}
	cv::Matx22d R;
	cv::Mat(svd.vt.t()*cv::Mat(corr)*svd.u.t()).copyTo(R);
	return R;
}
cv::Matx23d AlignShapeWithScale(cv::Mat_<double>& src, cv::Mat_<double>& dst)
{
	int n = src.rows;

	// First we mean normalise both src and dst
	double mean_src_x = cv::mean(src.col(0))[0];
	double mean_src_y = cv::mean(src.col(1))[0];

	double mean_dst_x = cv::mean(dst.col(0))[0];
	double mean_dst_y = cv::mean(dst.col(1))[0];

	cv::Mat_<double> src_mean_normed = src.clone();
	src_mean_normed.col(0) = src_mean_normed.col(0) - mean_src_x;
	src_mean_normed.col(1) = src_mean_normed.col(1) - mean_src_y;

	cv::Mat_<double> dst_mean_normed = dst.clone();
	dst_mean_normed.col(0) = dst_mean_normed.col(0) - mean_dst_x;
	dst_mean_normed.col(1) = dst_mean_normed.col(1) - mean_dst_y;

	// Find the scaling factor of each
	cv::Mat src_sq;
	cv::pow(src_mean_normed, 2, src_sq);

	cv::Mat dst_sq;
	cv::pow(dst_mean_normed, 2, dst_sq);

	double s_src = sqrt(cv::sum(src_sq)[0] / n);
	double s_dst = sqrt(cv::sum(dst_sq)[0] / n);

	src_mean_normed = src_mean_normed / s_src;
	dst_mean_normed = dst_mean_normed / s_dst;

	double s = s_dst / s_src;

	// Get the rotation
	cv::Matx22d R = AlignShapesKabsch2D(src_mean_normed, dst_mean_normed);
	// std::cout << R(0, 0) << " " << R(0, 1) << std::endl;
	// std::cout << R(1, 0) << " " << R(1, 1) << std::endl; 

	cv::Matx22d	A;
	cv::Mat(s * R).copyTo(A);

	cv::Mat_<double> aligned = (cv::Mat(cv::Mat(A) * src.t())).t();
	cv::Mat_<double> offset = dst - aligned;

	double t_x = cv::mean(offset.col(0))[0];
	double t_y = cv::mean(offset.col(1))[0];

	cv::Matx23d warp_matrix;
	warp_matrix(0, 0) = A(0, 0); warp_matrix(0, 1) = A(0, 1); warp_matrix(0, 2) = t_x;
	warp_matrix(1, 0) = A(1, 0); warp_matrix(1, 1) = A(1, 1); warp_matrix(1, 2) = t_y;

	return warp_matrix;
}
float PointToLineDis(Point2f cur, Point2f start, Point2f end){
	double a, b, c, dis;
	a = end.y - start.y;
	b = start.x - end.x;
	c = end.x * start.y - start.x * end.y;
	dis = abs(a * cur.x + b * cur.y + c) / sqrt(a * a + b * b);
	return dis;
}
bool PtInAnyRect1(Point2f pCur, Rect search)
{
	Point2f pLT, pRT, pLB, pRB;
	pLT.x = search.x;
	pLT.y = search.y;
	pRT.x = search.x + search.width;
	pRT.y = search.y;
	pLB.x = search.x;
	pLB.y = search.y + search.height;
	pRB.x = search.x + search.width;
	pRB.y = search.y + search.height;
	//任意四边形有4个顶点
	std::vector<double> jointPoint2fx;
	std::vector<double> jointPoint2fy;
	int nCount = 4;
	Point2f RectPoint2fs[4] = { pLT, pLB, pRB, pRT };
	int nCross = 0;
	for (int i = 0; i < nCount; i++)
	{
		Point2f pStart = RectPoint2fs[i];
		Point2f pEnd = RectPoint2fs[(i + 1) % nCount];

		if (pCur.y < min(pStart.y, pEnd.y) || pCur.y > max(pStart.y, pEnd.y))
			continue;

		double x = (double)(pCur.y - pStart.y) * (double)(pEnd.x - pStart.x) / (double)(pEnd.y - pStart.y) + pStart.x;
		if (x > pCur.x)nCross++;
	}
	return (nCross % 2 == 1);
}
bool PtInAnyRect2(Point2f pCur, Point2f pLT, Point2f pRT, Point2f pRB, Point2f pLB)
{
	/*Point2f pLT, pRT, pLB, pRB;
	pLT.x = search.x;
	pLT.y = search.y;
	pRT.x = search.x + search.width;
	pRT.y = search.y;
	pLB.x = search.x;
	pLB.y = search.y + search.height;
	pRB.x = search.x + search.width;
	pRB.y = search.y + search.height;*/

	//任意四边形有4个顶点
	std::vector<double> jointPoint2fx;
	std::vector<double> jointPoint2fy;
	int nCount = 4;
	Point2f RectPoint2fs[4] = { pLT, pLB, pRB, pRT };
	int nCross = 0;
	for (int i = 0; i < nCount; i++)
	{
		Point2f pStart = RectPoint2fs[i];
		Point2f pEnd = RectPoint2fs[(i + 1) % nCount];

		if (pCur.y < min(pStart.y, pEnd.y) || pCur.y > max(pStart.y, pEnd.y))
			continue;

		double x = (double)(pCur.y - pStart.y) * (double)(pEnd.x - pStart.x) / (double)(pEnd.y - pStart.y) + pStart.x;
		if (x > pCur.x)nCross++;
	}
	return (nCross % 2 == 1);
}
float CalculateVectorAngle(float x1, float y1, float x2, float y2, float x3, float y3)
{
	float x_1 = x2 - x1;
	float x_2 = x3 - x2;
	float y_1 = y2 - y1;
	float y_2 = y3 - y2;
	float lx = sqrt(x_1*x_1 + y_1*y_1);
	float ly = sqrt(x_2*x_2 + y_2*y_2);
	return 180.0 - acos((x_1*x_2 + y_1*y_2) / lx / ly) * 180 / 3.1415926;
}
int cosDistance(const cv::Mat q, const cv::Mat r, float& distance)
{
	assert((q.rows == r.rows) && (q.cols == r.cols));
	float fenzi = q.dot(r);
	float fenmu = sqrt(q.dot(q)) * sqrt(r.dot(r));
	distance = fenzi / fenmu;
	return 0;
}
float euDistance(Point2f q, Point2f r){
	float distance = sqrt(pow(q.x - r.x, 2) + pow(q.y - r.y, 2));
	return distance;
}
int featureCompare(const std::vector<float> query_feature, const std::vector<float> ref_feature, float& distance)
{
	cv::Mat q(query_feature);
	cv::Mat r(ref_feature);
	cosDistance(q, r, distance);   //cos distance
	return 0;
}
float Compute_IOU(const cv::Rect& rectA, const cv::Rect& rectB){
	if (rectA.x > rectB.x + rectB.width) { return 0.; }
	if (rectA.y > rectB.y + rectB.height) { return 0.; }
	if ((rectA.x + rectA.width) < rectB.x) { return 0.; }
	if ((rectA.y + rectA.height) < rectB.y) { return 0.; }
	float colInt = min(rectA.x + rectA.width, rectB.x + rectB.width) - max(rectA.x, rectB.x);
	float rowInt = min(rectA.y + rectA.height, rectB.y + rectB.height) - max(rectA.y, rectB.y);
	float intersection = colInt * rowInt;
	float areaA = rectA.width * rectA.height;
	float areaB = rectB.width * rectB.height;
	float intersectionPercent = intersection / (areaA + areaB - intersection);
	/*intersectRect.x = max(rectA.x, rectB.x);
	intersectRect.y = max(rectA.y, rectB.y);
	intersectRect.width = min(rectA.x + rectA.width, rectB.x + rectB.width) - intersectRect.x;
	intersectRect.height = min(rectA.y + rectA.height, rectB.y + rectB.height) - intersectRect.y;*/
	return intersectionPercent;
}
bool greate2(vector<float>a, vector<float>b){
	return a[1] > b[1];
}
bool greate3(Student_Info a, Student_Info b){
	return a.energy > b.energy;
}

Json::Value root_all;

void class_Json(vector<vector<Student_Info>>&students_all, vector<int>&student_valid, vector<Class_Info> &class_info_all, int &i, string &start_time, int &start_frame, int &end_frame, int &activity_order, string &end_time, int &negtive_num, Json::Value &class_infomation, string &dongzuo){
	char buff[200];
	//sprintf(buff, "%d/%d/%d-%02d:%02d:%02d", class_info_all[i].pstSystemTime.dwYear, class_info_all[i].pstSystemTime.dwMon, class_info_all[i].pstSystemTime.dwDay, class_info_all[i].pstSystemTime.dwHour, class_info_all[i].pstSystemTime.dwMin, class_info_all[i].pstSystemTime.dwSec);
	
	if (negtive_num == 10){
		//start_time = buff;
		start_frame = class_info_all[i].cur_frame;
		end_frame = class_info_all[i].cur_frame;
		activity_order+=2;
	}
	if (class_info_all[i].cur_frame - end_frame < 10){
		end_frame = class_info_all[i].cur_frame;
		//end_time = buff;
		
		for (int k = 0; k < student_valid.size(); k++){
			for (int j = 1; j < students_all[student_valid[k]].size(); j++){
				if (students_all[student_valid[k]][j].cur_frame1 >= start_frame && students_all[student_valid[k]][j].cur_frame1 <= end_frame){
					students_all[student_valid[k]][j].bow_head_each = false;
					students_all[student_valid[k]][j].bow_head = false;
					students_all[student_valid[k]][j].disscussion = false;
					students_all[student_valid[k]][j].raising_hand = false;
					students_all[student_valid[k]][j].standing = false;
				}
			}
		}
	}
	//if (end_frame - start_frame >= 4){
		//string append_string = "(" + start_time + "," + end_time + ")";
		string append_frame = "(" + to_string(start_frame) + "," + to_string(end_frame) + ")";
		int a;
		a = activity_order >= 2 ? activity_order : 2;
		//class_infomation[dongzuo][a - 2] = append_string;
		class_infomation[dongzuo][a - 1] = append_frame;
	//}
	
}
void student_Json(vector<vector<Student_Info>>&students_all, vector<int>&student_valid, int &i, int &j, string &start_time, int &start_frame, int &end_frame, int &activity_order, string &end_time, int &negtive_num, Point &ss, Json::Value &behavior_infomation, Json::Value &all_rect, string &dongzuo){

	char buff[200];
	//sprintf(buff, "%d/%d/%d-%02d:%02d:%02d", students_all[student_valid[i]][j].pstSystemTime.dwYear, students_all[student_valid[i]][j].pstSystemTime.dwMon, students_all[student_valid[i]][j].pstSystemTime.dwDay, students_all[student_valid[i]][j].pstSystemTime.dwHour, students_all[student_valid[i]][j].pstSystemTime.dwMin, students_all[student_valid[i]][j].pstSystemTime.dwSec);

	if (negtive_num == 3){
		//start_time = buff;
		start_frame = students_all[student_valid[i]][j].cur_frame1;
		end_frame = students_all[student_valid[i]][j].cur_frame1;
		ss.x = j;
		ss.y = j;
		activity_order += 2;
	}

	if (students_all[student_valid[i]][j].cur_frame1 - end_frame <= 3){
		end_frame = students_all[student_valid[i]][j].cur_frame1;
		ss.y = j;
		//end_time = buff;
	}
	vector<int>msf;
	for (int l = 0; l < students_all[student_valid[i]][0].miss_frame.size(); l++){
		if (students_all[student_valid[i]][0].miss_frame[l]>start_frame && students_all[student_valid[i]][0].miss_frame[l] < end_frame)
			msf.push_back(students_all[student_valid[i]][0].miss_frame[l]);
	}
	/*string miss_f = " ";
	for (int l = 0; l < msf.size(); l++){
	if (l == 0){
	miss_f = "[";
	}
	miss_f += to_string(msf[l]);
	if (l == msf.size() - 1){
	miss_f += "]";
	}
	else miss_f += ",";
	}*/
	//string append_string = "(" + start_time + "," + end_time + ")";
	string append_frame = "(" + to_string(start_frame) + "," + to_string(end_frame) + ")";

	int a;
	a = activity_order >= 2 ? activity_order : 2;

	behavior_infomation[dongzuo][a - 2] = append_frame;
	//behavior_infomation[dongzuo][a - 1] = append_string;

	//----------------------------------------------------
	
	all_rect[dongzuo][a - 2] = /*miss_f != " " ? append_frame + "," + miss_f : */append_frame;
	Json::Value student_loc;
	int count = 0;
	//for (int k = start_frame; k <= end_frame; k++){
	for (int k = start_frame; k <= end_frame; k++){
		Json::Value student_rect;
		auto iter = find(msf.begin(), msf.end(), k);
		if (iter != msf.end()){
			//student_rect.append(0);
			//student_rect.append(0);
			//student_rect.append(0);
			//student_rect.append(0);
			student_rect.append(students_all[student_valid[i]][count + ss.x].body_for_save.x);
			student_rect.append(students_all[student_valid[i]][count + ss.x].body_for_save.y);
			student_rect.append(students_all[student_valid[i]][count + ss.x].body_for_save.width);
			student_rect.append(students_all[student_valid[i]][count + ss.x].body_for_save.height);
		}
		else{
			student_rect.append(students_all[student_valid[i]][count + ss.x].body_for_save.x);
			student_rect.append(students_all[student_valid[i]][count + ss.x].body_for_save.y);
			student_rect.append(students_all[student_valid[i]][count + ss.x].body_for_save.width);
			student_rect.append(students_all[student_valid[i]][count + ss.x].body_for_save.height);
			count++;
		}
		student_loc.append(student_rect);
	}
	all_rect[dongzuo][a - 1] = Json::Value(student_loc);
	
}


void writeJson(vector<int>&student_valid, vector<vector<Student_Info>>&students_all, vector<Class_Info>&class_info_all, string &output,int &n){

	int pos1 = output.find_last_of("/");
	int pos2 = output.find_last_of(".");
	string videoname = output.substr(pos1, pos2 - pos1);

	vector<string>start_time(7,"time");
	vector<string>end_time(7,"time");
	vector<int>start_frame(7,0);
	vector<int>end_frame(7, 0);
	vector<int>activity_order(7,0);

	string status1 = "Raising hand";
	string status2 = "Standing";
	string status3 = "2-Disscussion";
	string status3back = "4-Disscussion";
	string status4 = "Dazing";
	string status5 = "Bow Head";

	Json::Value root1;
	Json::Value class_infomation;
	
	for (int i = 0; i < class_info_all.size(); i++){
		
		if (class_info_all[i].all_bow_head == true){
			int negtive_num = 0;
			if (i - 10 >= 0){
				for (int j = i - 10; j < i; j++){
					if (class_info_all[j].all_bow_head == false)negtive_num++;
				}
			}
			else{
				for (int j = 0; j < i; j++){
					if (class_info_all[j].all_bow_head == false)negtive_num++;
				}
				if (negtive_num == i)negtive_num = 10;
			}
			string dongzuo = "all_bow_head";
			class_Json(students_all, student_valid,class_info_all, i, start_time[0], start_frame[0], end_frame[0], activity_order[0], end_time[0], negtive_num, class_infomation, dongzuo);

		}
		if (class_info_all[i].all_disscussion_2 == true){
			int negtive_num = 0;
			if (i - 10 >= 0){
				for (int j = i - 10; j < i; j++){
					if (class_info_all[j].all_disscussion_2 == false)negtive_num++;
				}
			}
			else{
				for (int j = 0; j < i; j++){
					if (class_info_all[j].all_disscussion_2 == false)negtive_num++;
				}
				if (negtive_num == i)negtive_num = 10;
			}
			string dongzuo = "all_disscussion_2";
			class_Json(students_all, student_valid,class_info_all, i, start_time[1], start_frame[1], end_frame[1], activity_order[1], end_time[1], negtive_num, class_infomation, dongzuo);

		}
		if (class_info_all[i].all_disscussion_4 == true){
			int negtive_num = 0;
			if (i - 10 >= 0){
				for (int j = i - 10; j < i; j++){
					if (class_info_all[j].all_disscussion_4 == false)negtive_num++;
				}
			}
			else{
				for (int j = 0; j < i; j++){
					if (class_info_all[j].all_disscussion_4 == false)negtive_num++;
				}
				if (negtive_num == i)negtive_num = 10;
			}
			string dongzuo = "all_disscussion_4";
			class_Json(students_all, student_valid,class_info_all, i, start_time[2], start_frame[2], end_frame[2], activity_order[2], end_time[2], negtive_num, class_infomation, dongzuo);
		}
	}
	
	root1["class_infomation"] = Json::Value(class_infomation);
	ofstream out1;
	string jsonfile1 = output.substr(0, pos1) + "/" + videoname+"-Class" + ".json";
	out1.open(jsonfile1);
	Json::StyledWriter sw1;
	out1 << sw1.write(root1);
	out1.close();

	//---------------------------------------------------------------------------------
	vector<Point>ss(4);
	
	Json::Value root2;
	Json::Value root3;
	Json::Value root4;
	for (int i = 0; i < student_valid.size(); i++){
		
		for (int j = 0; j < 7; j++){
			activity_order[j] = 0;
			start_frame[j] = 0;
			end_frame[j] = 0;
			start_time[j] = "time";
			end_time[j] = "time";
			if (j < 4){
				ss[j].x = 0;
				ss[j].y = 0;
			}
		}
		
		Json::Value behavior_infomation;
		Json::Value all_rect;

		behavior_infomation["max_energy"] = students_all[student_valid[i]][0].cur_frame1;
		behavior_infomation["ID"] = student_valid[i];	
		all_rect["ID"] = student_valid[i];

		

	
		for (int j = 1; j < students_all[student_valid[i]].size(); j++){
			
			if (students_all[student_valid[i]][j].bow_head_each == true){

				int negtive_num = 0;
				if (j - 3 > 0){
					for (int k = j - 3; k < j; k++){
						if (students_all[student_valid[i]][k].bow_head_each == false)negtive_num++;
					}
				}
				else{
					for (int k = 1; k < j; k++){
						if (students_all[student_valid[i]][k].bow_head_each == false)negtive_num++;
					}
					if (negtive_num == j-1)negtive_num = 3;
				}
				string dongzuo = "bow_head";
				student_Json(students_all, student_valid, i, j, start_time[3], start_frame[3], end_frame[3], activity_order[3], end_time[3], negtive_num, ss[0], behavior_infomation, all_rect,dongzuo);
			}
		
			if (students_all[student_valid[i]][j].daze == true){
				int negtive_num = 0;

				if (j - 3 > 0){
					for (int k = j - 3; k < j; k++){
						if (students_all[student_valid[i]][k].daze == false)negtive_num++;
					}
				}
				else{
					for (int k = 1; k < j; k++){
						if (students_all[student_valid[i]][k].daze == false)negtive_num++;
					}
					if (negtive_num == j-1)negtive_num = 3;
				}

				string dongzuo = "daze";
				student_Json(students_all, student_valid, i, j, start_time[4], start_frame[4], end_frame[4], activity_order[4], end_time[4], negtive_num, ss[1], behavior_infomation, all_rect,dongzuo);
			}
		
			if (students_all[student_valid[i]][j].raising_hand == true && students_all[student_valid[i]][j].real_raise==true){
				/*if (student_valid[i] == 35){
					if (students_all[student_valid[i]][j].cur_frame1 == 106){
						cout << students_all[student_valid[i]][j - 1] .cur_frame1<<" "<< students_all[student_valid[i]][j - 1].raising_hand << " " << students_all[student_valid[i]][j - 1].real_raise << endl;
						cout << students_all[student_valid[i]][j - 2].cur_frame1 << " " << students_all[student_valid[i]][j - 2].raising_hand << " " << students_all[student_valid[i]][j - 2].real_raise << endl;
						cout << students_all[student_valid[i]][j - 3].cur_frame1 << " " << students_all[student_valid[i]][j - 3].raising_hand << " " << students_all[student_valid[i]][j - 3].real_raise << endl;
					}
				}*/
				int negtive_num = 0;
				if (j - 3 > 0){
					for (int k = j - 3; k < j; k++){
						if (students_all[student_valid[i]][k].raising_hand == false || students_all[student_valid[i]][k].real_raise == false)negtive_num++;
					}
				}
				else{
					for (int k = 1; k < j; k++){
						if (students_all[student_valid[i]][k].raising_hand == false || students_all[student_valid[i]][k].real_raise == false)negtive_num++;
					}
					if (negtive_num == j-1)negtive_num = 3;
				}
				string dongzuo = "raising_hand";
				student_Json(students_all, student_valid, i, j, start_time[5], start_frame[5], end_frame[5], activity_order[5], end_time[5], negtive_num, ss[2], behavior_infomation, all_rect,dongzuo);
			}
		
			if (students_all[student_valid[i]][j].standing == true){
				int negtive_num = 0;
				if (j - 3 > 0){
					for (int k = j - 3; k < j; k++){
						if (students_all[student_valid[i]][k].standing == false)negtive_num++;
					}
				}
				else{
					for (int k = 1; k < j; k++){
						if (students_all[student_valid[i]][k].standing == false)negtive_num++;
					}
					if (negtive_num == j-1)negtive_num = 3;
				}
				string dongzuo = "standing";
				student_Json(students_all, student_valid, i, j, start_time[6], start_frame[6], end_frame[6], activity_order[6], end_time[6], negtive_num, ss[3], behavior_infomation, all_rect,  dongzuo);
			}
		}

		root2["student"].append(behavior_infomation);
		root3["student"].append(all_rect);
		
	}

	ofstream out;
	string jsonfile = output.substr(0,pos1)+"/" +videoname +"-Stu"+ ".json";
	out.open(jsonfile);
	Json::StyledWriter sw;
	out << sw.write(root2);
	out.close();

	ofstream out2;
	string jsonfile2 = output.substr(0, pos1) + "/" + videoname + "-Rect" + ".json";
	out2.open(jsonfile2);
	Json::StyledWriter sw2;
	out2 << sw2.write(root3);
	out2.close();

	
}

void student_Json1(vector<vector<Student_Info>>&students_all, vector<vector<Student_Info>>&ID, vector<int>&student_valid, int &i, int &j, string &start_time, int &start_frame, int &end_frame, int &activity_order, string &end_time, int &negtive_num, Point &ss, Json::Value &behavior_infomation, Json::Value &all_rect, string &dongzuo){
	int matching_at_end = students_all[student_valid[i]][0].matching_at_end;
	char buff[200];
	//sprintf(buff, "%d/%d/%d-%02d:%02d:%02d", students_all[student_valid[i]][j].pstSystemTime.dwYear, students_all[student_valid[i]][j].pstSystemTime.dwMon, students_all[student_valid[i]][j].pstSystemTime.dwDay, students_all[student_valid[i]][j].pstSystemTime.dwHour, students_all[student_valid[i]][j].pstSystemTime.dwMin, students_all[student_valid[i]][j].pstSystemTime.dwSec);

	if (negtive_num == 3){
		//start_time = buff;
		start_frame = ID[matching_at_end][j].cur_frame1;
		end_frame = ID[matching_at_end][j].cur_frame1;
		ss.x = j;
		ss.y = j;
		activity_order += 2;
	}

	if (ID[matching_at_end][j].cur_frame1 - end_frame <= 3){
		end_frame = ID[matching_at_end][j].cur_frame1;
		ss.y = j;
		//end_time = buff;
	}
	vector<int>msf;
	for (int l = 0; l < ID[matching_at_end][0].miss_frame.size(); l++){
		if (ID[matching_at_end][0].miss_frame[l]>start_frame && ID[matching_at_end][0].miss_frame[l] < end_frame)
			msf.push_back(ID[matching_at_end][0].miss_frame[l]);
	}
	/*string miss_f = " ";
	for (int l = 0; l < msf.size(); l++){
	if (l == 0){
	miss_f = "[";
	}
	miss_f += to_string(msf[l]);
	if (l == msf.size() - 1){
	miss_f += "]";
	}
	else miss_f += ",";
	}*/
	//string append_string = "(" + start_time + "," + end_time + ")";
	string append_frame = "(" + to_string(start_frame) + "," + to_string(end_frame) + ")";

	int a;
	a = activity_order >= 2 ? activity_order : 2;

	behavior_infomation[dongzuo][a - 2] = append_frame;
	//behavior_infomation[dongzuo][a - 1] = append_string;

	//----------------------------------------------------

	all_rect[dongzuo][a - 2] = /*miss_f != " " ? append_frame + "," + miss_f : */append_frame;
	Json::Value student_loc;
	int count = 0;
	//for (int k = start_frame; k <= end_frame; k++){
	for (int k = start_frame; k <= end_frame; k++){
		Json::Value student_rect;
		auto iter = find(msf.begin(), msf.end(), k);
		if (iter != msf.end()){
			//student_rect.append(0);
			//student_rect.append(0);
			//student_rect.append(0);
			//student_rect.append(0);
			student_rect.append(ID[matching_at_end][count + ss.x].body_for_save.x);
			student_rect.append(ID[matching_at_end][count + ss.x].body_for_save.y);
			student_rect.append(ID[matching_at_end][count + ss.x].body_for_save.width);
			student_rect.append(ID[matching_at_end][count + ss.x].body_for_save.height);
		}
		else{
			student_rect.append(ID[matching_at_end][count + ss.x].body_for_save.x);
			student_rect.append(ID[matching_at_end][count + ss.x].body_for_save.y);
			student_rect.append(ID[matching_at_end][count + ss.x].body_for_save.width);
			student_rect.append(ID[matching_at_end][count + ss.x].body_for_save.height);
			count++;
		}
		student_loc.append(student_rect);
	}
	all_rect[dongzuo][a - 1] = Json::Value(student_loc);

}

void writeJson1(vector<int>&student_valid, vector<vector<Student_Info>>&students_all, vector<vector<Student_Info>>&ID, vector<Class_Info>&class_info_all, string &output, int &n){

	int pos1 = output.find_last_of("/");
	int pos2 = output.find_last_of(".");
	string videoname = output.substr(pos1, pos2 - pos1);

	vector<string>start_time(7, "time");
	vector<string>end_time(7, "time");
	vector<int>start_frame(7, 0);
	vector<int>end_frame(7, 0);
	vector<int>activity_order(7, 0);

	string status1 = "Raising hand";
	string status2 = "Standing";
	string status3 = "2-Disscussion";
	string status3back = "4-Disscussion";
	string status4 = "Dazing";
	string status5 = "Bow Head";

	Json::Value root1;
	Json::Value class_infomation;

	for (int i = 0; i < class_info_all.size(); i++){

		if (class_info_all[i].all_bow_head == true){
			int negtive_num = 0;
			if (i - 10 >= 0){
				for (int j = i - 10; j < i; j++){
					if (class_info_all[j].all_bow_head == false)negtive_num++;
				}
			}
			else{
				for (int j = 0; j < i; j++){
					if (class_info_all[j].all_bow_head == false)negtive_num++;
				}
				if (negtive_num == i)negtive_num = 10;
			}
			string dongzuo = "all_bow_head";
			class_Json(students_all, student_valid, class_info_all, i, start_time[0], start_frame[0], end_frame[0], activity_order[0], end_time[0], negtive_num, class_infomation, dongzuo);

		}
		if (class_info_all[i].all_disscussion_2 == true){
			int negtive_num = 0;
			if (i - 10 >= 0){
				for (int j = i - 10; j < i; j++){
					if (class_info_all[j].all_disscussion_2 == false)negtive_num++;
				}
			}
			else{
				for (int j = 0; j < i; j++){
					if (class_info_all[j].all_disscussion_2 == false)negtive_num++;
				}
				if (negtive_num == i)negtive_num = 10;
			}
			string dongzuo = "all_disscussion_2";
			class_Json(students_all, student_valid, class_info_all, i, start_time[1], start_frame[1], end_frame[1], activity_order[1], end_time[1], negtive_num, class_infomation, dongzuo);

		}
		if (class_info_all[i].all_disscussion_4 == true){
			int negtive_num = 0;
			if (i - 10 >= 0){
				for (int j = i - 10; j < i; j++){
					if (class_info_all[j].all_disscussion_4 == false)negtive_num++;
				}
			}
			else{
				for (int j = 0; j < i; j++){
					if (class_info_all[j].all_disscussion_4 == false)negtive_num++;
				}
				if (negtive_num == i)negtive_num = 10;
			}
			string dongzuo = "all_disscussion_4";
			class_Json(students_all, student_valid, class_info_all, i, start_time[2], start_frame[2], end_frame[2], activity_order[2], end_time[2], negtive_num, class_infomation, dongzuo);
		}
	}

	root1["class_infomation"] = Json::Value(class_infomation);
	ofstream out1;
	string jsonfile1 = output.substr(0, pos1) + "/" + videoname + "-Class" + ".json";
	out1.open(jsonfile1);
	Json::StyledWriter sw1;
	out1 << sw1.write(root1);
	out1.close();

	//---------------------------------------------------------------------------------
	vector<Point>ss(4);

	Json::Value root2;
	Json::Value root3;

	for (int i = 0; i < student_valid.size(); i++){
		if (students_all[student_valid[i]][0].matching_at_end != 100){

			int matching_at_end = students_all[student_valid[i]][0].matching_at_end;

			for (int j = 0; j < 7; j++){
				activity_order[j] = 0;
				start_frame[j] = 0;
				end_frame[j] = 0;
				start_time[j] = "time";
				end_time[j] = "time";
				if (j < 4){
					ss[j].x = 0;
					ss[j].y = 0;
				}
			}

			Json::Value behavior_infomation;
			Json::Value all_rect;

			//behavior_infomation["max_energy"] = students_all[student_valid[i]][0].cur_frame1;
			behavior_infomation["ID"] = matching_at_end;
			all_rect["ID"] = matching_at_end;

		

			for (int j = 1; j < ID[matching_at_end].size(); j++){

				if (ID[matching_at_end][j].bow_head_each == true){

					int negtive_num = 0;
					if (j - 3 > 0){
						for (int k = j - 3; k < j; k++){
							if (ID[matching_at_end][k].bow_head_each == false)negtive_num++;
						}
					}
					else{
						for (int k = 1; k < j; k++){
							if (ID[matching_at_end][k].bow_head_each == false)negtive_num++;
						}
						if (negtive_num == j - 1)negtive_num = 3;
					}
					string dongzuo = "bow_head";
					student_Json1(students_all,ID, student_valid, i, j, start_time[3], start_frame[3], end_frame[3], activity_order[3], end_time[3], negtive_num, ss[0], behavior_infomation, all_rect, dongzuo);
				}

				if (ID[matching_at_end][j].daze == true){
					int negtive_num = 0;

					if (j - 3 > 0){
						for (int k = j - 3; k < j; k++){
							if (ID[matching_at_end][k].daze == false)negtive_num++;
						}
					}
					else{
						for (int k = 1; k < j; k++){
							if (ID[matching_at_end][k].daze == false)negtive_num++;
						}
						if (negtive_num == j - 1)negtive_num = 3;
					}

					string dongzuo = "daze";
					student_Json1(students_all,ID, student_valid, i, j, start_time[4], start_frame[4], end_frame[4], activity_order[4], end_time[4], negtive_num, ss[1], behavior_infomation, all_rect, dongzuo);
				}

				if (ID[matching_at_end][j].raising_hand == true && ID[matching_at_end][j].real_raise == true){
					
					int negtive_num = 0;
					if (j - 3 > 0){
						for (int k = j - 3; k < j; k++){
							if (ID[matching_at_end][k].raising_hand == false || ID[matching_at_end][k].real_raise == false)negtive_num++;
						}
					}
					else{
						for (int k = 1; k < j; k++){
							if (ID[matching_at_end][k].raising_hand == false || ID[matching_at_end][k].real_raise == false)negtive_num++;
						}
						if (negtive_num == j - 1)negtive_num = 3;
					}
					string dongzuo = "raising_hand";
					student_Json1(students_all,ID, student_valid, i, j, start_time[5], start_frame[5], end_frame[5], activity_order[5], end_time[5], negtive_num, ss[2], behavior_infomation, all_rect, dongzuo);
				}

				if (ID[matching_at_end][j].standing == true){
					int negtive_num = 0;
					if (j - 3 > 0){
						for (int k = j - 3; k < j; k++){
							if (ID[matching_at_end][k].standing == false)negtive_num++;
						}
					}
					else{
						for (int k = 1; k < j; k++){
							if (ID[matching_at_end][k].standing == false)negtive_num++;
						}
						if (negtive_num == j - 1)negtive_num = 3;
					}
					string dongzuo = "standing";
					student_Json1(students_all,ID, student_valid, i, j, start_time[6], start_frame[6], end_frame[6], activity_order[6], end_time[6], negtive_num, ss[3], behavior_infomation, all_rect, dongzuo);
				}
			}

			root2["student"].append(behavior_infomation);
			root3["student"].append(all_rect);
			
		}

		ofstream out;
		string jsonfile = output.substr(0, pos1) + "/" + videoname + "-Stu" + ".json";
		out.open(jsonfile);
		Json::StyledWriter sw;
		out << sw.write(root2);
		out.close();

		ofstream out2;
		string jsonfile2 = output.substr(0, pos1) + "/" + videoname + "-Rect" + ".json";
		out2.open(jsonfile2);
		Json::StyledWriter sw2;
		out2 << sw2.write(root3);
		out2.close();

		
	}
}

void drawGrid(Mat &image, vector<int>student_valid,vector<vector<Student_Info>>students_all){
	vector<vector<int>>orderr = { { 14, 3, 21, 19, 5, 17 }, { 15, 20, 2, 9, 12, 22, 24 }, { 7, 0, 1, 26, 50, 48 }, { 27, 4, 6, 32, 29, 49, 13 }, { 23, 41, 8, 34, 39, 35, 31 }, { 38, 43, 28, 10, 16, 42, 51 }, { 18, 37, 33, 30, 44, 40 }, {47,45,25,11,46,36} };
	for (int i = 0; i < orderr.size(); i++){
		for (int j = 0; j < orderr[i].size()-1; j++){
			auto iter1 = find(student_valid.begin(), student_valid.end(), orderr[i][j]);
			int index1 = distance(student_valid.begin(), iter1);
			auto iter2 = find(student_valid.begin(), student_valid.end(), orderr[i][j+1]);
			int index2 = distance(student_valid.begin(), iter2);
			/*Point2f start = students_all[orderr[i][j]][students_all[orderr[i][j]].size() - 1].loc;
			Point2f end = students_all[orderr[i][j+1]][students_all[orderr[i][j+1]].size() - 1].loc;*/
			Point2f start = students_all[student_valid[index1]][students_all[student_valid[index1]].size() - 1].loc;
			Point2f end = students_all[student_valid[index2]][students_all[student_valid[index2]].size() - 1].loc;
			line(image, start, end, Scalar(255, 0, 0), 2, 8, 0);
		}
	}
}
