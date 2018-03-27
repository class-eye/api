#ifndef _BEHAVIOR_H_
#define _BEHAVIOR_H_
#include <opencv2/core/core.hpp>
#include "student.hpp"
#include "functions.hpp"
using namespace cv;
void Analys_Behavior(vector<vector<Student_Info>>&students_all, vector<int>&student_valid, vector<Class_Info> &class_info, Mat &image_1080, Mat &image, int &n);
void detect_discontinuous_behavior(Net &net2, Mat &image, PoseInfo &pose, Student_Info &student_info, int &i, int &v, int x[], int y[]);
//void face_match(Net &net3, Net &net4, jfda::JfdaDetector &detector, vector<vector<Student_Info>>&students_all, vector<int>&student_valid, int &n, Mat &image_1080, vector<FaceInfo>&standard_faces);
//void renew_face_match(Net &net4, jfda::JfdaDetector &detector, vector<vector<Student_Info>>&students_all, vector<int>&student_valid, int &n, Mat &image_1080, vector<FaceInfo>&standard_faces);
//void good_face(Net &net3, Net &net4, jfda::JfdaDetector &detector, vector<vector<Student_Info>>&students_all, vector<int>&student_valid, int &n, Mat &image_1080, vector<FaceInfo>&standard_faces, int &max_student_num);

#endif