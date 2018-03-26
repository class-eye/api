#ifndef _STUDENT_H_
#define _STUDENT_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
#include <map>
//#include "pose.hpp"
#include "hand.hpp"
#include "face.hpp"
#include "jfda.hpp"
#include "Timer.hpp"
#include "pose.hpp"
#include "fs.hpp"
#include "facefeature.hpp"
#ifdef __unix__
#include <json/json.h>
//#include <python2.7/Python.h>
#endif
using namespace cv;
using namespace caffe;

struct Class_Info{
	bool all_bow_head=false;
	bool all_disscussion_2 = false;
	bool all_disscussion_4 = false;
	int cur_frame=0;
	
};
struct Student_Info{
	bool raising_hand=false;
	bool standing=false;
	bool disscussion= false;
	bool daze = false;
	bool bow_head = false;
	bool bow_head_each = false;

	bool turn_head = false;
	bool arm_vertical = false;
	bool whisper = false;
	bool turn_body = false;
	bool bow_head_tmp = false;
	Point2f loc;
	Point2f neck_loc;
	Rect body_bbox;
	Rect body_for_save;
	Rect face_bbox;
	//string output_body_dir;
	int away_from_seat = 0;
	int cur_frame1=0;
	int cur_size = 0;
	int energy = 0;
	int max_energy = 0;
	bool front=false;
	bool back = false;
	vector<int>miss_frame;
	//vector<Point2f>all_points;
	
	bool real_raise = false;
	float scores = 0.0;
	
	vector<float>good_face_features;

	bool have_matched = false;
	vector<int>matching;
	int matching_at_end = 100;
	int prev_matching_at_end = 100;
	map<int, int>match_history;
};

class Student_analy
{
public:
	Student_analy(const string& pose_net, const string& pose_model,
		    const string& hands_net, const string& hands_model, 
		    const string& front_face_net, const string& front_face_model, 
			const string& face_feature_net, const string& face_feature_model, 
			const string& real_front_face_net, const string& real_front_face_model, int gpu_device = -1);
	~Student_analy();
	int GetStandaredFeats(Mat &frame_1080, string &output, int &max_student_num,int stop = 0);
	std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_detect(jfda::JfdaDetector &detector, Mat &image, string &output, int &max_student_num);
	int good_face(jfda::JfdaDetector &detector, Mat &image_1080, int &max_student_num,int stop = 0);
	void face_match(jfda::JfdaDetector &detector, Mat &image_1080, int &max_student_num,int stop = 0);
	void add_location(Rect &box);
	void add_feature(jfda::JfdaDetector &detector,Mat &image_1080, Rect &box);
	void add_match(int &location_num, int &face_num);
	void init();
	void clear();

private:
	vector<vector<Student_Info>>students_all;
	vector<vector<Student_Info>>ID;
	vector<int>student_valid;
	vector<Class_Info> class_info_all;
	vector<vector<FaceInfo>>standard_faces;
	
	int n=0;
	int n1=0;
	
	Net *posenet;
	Net *handsnet;
	Net *frontface_net;
	Net *facefeature_net;
	Net *real_frontface_net;

	//void writeJson(string &output);
	//void student_Json(int &i, int &j, string &start_time, int &start_frame, int &end_frame, int &activity_order, string &end_time, int &negtive_num, Point &ss, Json::Value &behavior_infomation, Json::Value &all_rect, string &dongzuo);
	//void class_Json(int &i, string &start_time, int &start_frame, int &end_frame, int &activity_order, string &end_time, int &negtive_num, Json::Value &class_infomation, string &dongzuo);

};



#endif

