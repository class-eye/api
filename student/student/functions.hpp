#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "student/student.hpp"
using namespace cv;
void refine(Rect& bbox, cv::Mat& img);
cv::Mat CropPatch(const cv::Mat& img, cv::Rect& bbox);
cv::Matx23d AlignShapeWithScale(cv::Mat_<double>& src, cv::Mat_<double>& dst);
float PointToLineDis(Point2f cur, Point2f start, Point2f end);
bool PtInAnyRect1(Point2f pCur, Rect search);
bool PtInAnyRect2(Point2f pCur, Point2f pLT, Point2f pRT, Point2f pRB, Point2f pLB);
float CalculateVectorAngle(float x1, float y1, float x2, float y2, float x3, float y3);
int cosDistance(const cv::Mat q, const cv::Mat r, float& distance);
float euDistance(Point2f q, Point2f r);
int featureCompare(const std::vector<float> query_feature, const std::vector<float> ref_feature, float& distance);
float Compute_IOU(const cv::Rect& rectA, const cv::Rect& rectB);
bool greate2(vector<float>a, vector<float>b);
bool greate3(Student_Info a, Student_Info b);
void writeJson(vector<int>&student_valid, vector<vector<Student_Info>>&students_all, vector<Class_Info>&class_info_all, string &videoname,int &n);
//void writeJson1(vector<int>&student_valid, vector<vector<Student_Info>>&students_all, vector<vector<Student_Info>>&ID, vector<Class_Info>&class_info_all, string &output, int &n);
void drawGrid(Mat &image, vector<int>student_valid,vector<vector<Student_Info>>students_all);
#endif