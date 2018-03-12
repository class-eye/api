#include <fstream>
#include <iostream>
#include <thread>
#include <iostream>
#include <numeric>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/caffe.hpp"
#include "cv.h"  
#include "student/student.hpp"
#include "student/behavior.hpp"
#include "student/functions.hpp"
#include<cmath>
#include<tuple>

using namespace cv;
using namespace std;
using namespace caffe;
//vector<Class_Info>class_info_all;
//vector<int>student_valid;
//vector<vector<Student_Info>>students_all(70);
int standard_frame = 1;

Student_analy::Student_analy(const string& pose_net, const string& pose_model,
	const string& hands_net, const string& hands_model,
	const string& front_face_net, const string& front_face_model,
	const string& face_feature_net, const string& face_feature_model, int gpu_device){
	students_all.resize(70);
	ID.resize(70);
	if (gpu_device < 0) {
		caffe::SetMode(caffe::CPU, -1);
	}
	else {
		if(caffe::GPUAvailable())
			caffe::SetMode(caffe::GPU, gpu_device);
	}
	posenet = new caffe::Net(pose_net);
	posenet->CopyTrainedLayersFrom(pose_model);
	handsnet = new caffe::Net(hands_net);
	handsnet->CopyTrainedLayersFrom(hands_model);

	frontface_net = new caffe::Net(front_face_net);
	frontface_net->CopyTrainedLayersFrom(front_face_model);

	facefeature_net = new caffe::Net(face_feature_net);
	facefeature_net->CopyTrainedLayersFrom(face_feature_model);
}

int Student_analy::GetStandaredFeats(Mat &frame_1080, string &output, int &max_student_num){
	if (n%standard_frame == 0){
		PoseInfo pose;
		Timer timer;
		Mat frame;
		cv::resize(frame_1080, frame, Size(0, 0), 2 / 3., 2 / 3.);
		pose_detect(*posenet, frame, pose);
		int stu_n = 0;
		int stu_real = 0;
		float score_sum = 0;
		for (int i = 0; i < pose.subset.size(); i++){
			float score = float(pose.subset[i][18]) / pose.subset[i][19];			
			if (pose.subset[i][19] >= 4 && score >= 0.4){
				stu_real++;
				/*if (pose.subset[i][1] == -1){
					for (int j = 0; j < 18; j++){
						if (pose.subset[i][j] != -1)cv::circle(frame, Point2f(pose.candicate[pose.subset[i][j]][0], pose.candicate[pose.subset[i][j]][1]), 5, cv::Scalar(0,255,255), -1);
					}
				}*/
				if (pose.subset[i][1] != -1){
					score_sum += pose.subset[i][18];
					stu_n++;
					if (pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
						if (pose.candicate[pose.subset[i][2]][0] > pose.candicate[pose.subset[i][5]][0]){
							stu_n--;
						}
					}
				}
			}
		}
		if (stu_n >= max_student_num){
			//max_student_num = stu_n;
			cout << stu_n << " / " << stu_real << endl;
			n1++;
			//if (score_sum > score_all){
				//score_all = score_sum;

				student_valid.clear();
				for (int i = 0; i < 70; i++){
					students_all[i].clear();
				}
				//int xuhao = 0;
				for (int i = 0; i < pose.subset.size(); i++){
					float score = float(pose.subset[i][18]) / pose.subset[i][19];
					if (pose.subset[i][19] >= 4 && score >= 0.4){
						if (pose.subset[i][1] != -1){
							float wid1 = 0, wid2 = 0, wid = 0;
							if (pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
								wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
								wid2 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
								wid = MAX(wid1, wid2);
								if (wid == 0)continue;
							}
							if (pose.subset[i][2] != -1 && pose.subset[i][5] == -1){
								wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
								wid2 = wid1;
								wid = wid1;
								if (wid == 0)continue;
							}
							if (pose.subset[i][2] == -1 && pose.subset[i][5] != -1){
								wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
								wid2 = wid1;
								wid = wid1;
								if (wid == 0)continue;
							}


							Rect standard_rect;
							standard_rect.x = pose.candicate[pose.subset[i][1]][0] - wid - 5;
							standard_rect.y = pose.candicate[pose.subset[i][1]][1] - 5;
							standard_rect.width = wid1 + wid2 + 10;
							standard_rect.height = wid1 + wid2;
							if (standard_rect.height < 5)standard_rect.height = 15;
							refine(standard_rect, frame);
							cv::rectangle(frame, standard_rect, Scalar(0, 0, 255), 2, 8, 0);

							Student_Info student_ori;
							student_ori.body_bbox = standard_rect;
							student_ori.neck_loc = Point2f(pose.candicate[pose.subset[i][1]][0], pose.candicate[pose.subset[i][1]][1]);

							if (pose.subset[i][0] != -1){
								student_ori.loc = Point2f(pose.candicate[pose.subset[i][0]][0], pose.candicate[pose.subset[i][0]][1]);
								student_ori.front = true;
							}
							else{
								student_ori.loc = student_ori.neck_loc;
								student_ori.front = false;
							}
							student_valid.push_back(i);
							students_all[i].push_back(student_ori);
							string b = output + "/" + "00.jpg";
							//cv::circle(frame, student_ori.loc, 3, cv::Scalar(0, 0, 255), -1);
							//if (xuhao == max_student_num - 1){
							//	cv::putText(frame, to_string(xuhao), student_ori.loc, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
							//}
							//else{
							cv::putText(frame, to_string(i), student_ori.loc, FONT_HERSHEY_COMPLEX, 0.7, Scalar(0, 0, 255), 2);
							//}
							//xuhao++;
							cv::imwrite(b, frame);
						}
					}
				}
			//}
		}
		else{ cout << "student_num: " << stu_n << endl; }
		return n1;
	}
}


std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>Student_analy::student_detect(jfda::JfdaDetector &detector, Mat &image_1080, string &output,int &max_student_num){
	/*vector<Student_Info>student_detect(Net &net1, Mat &image, int &n, PoseInfo &pose,string &output)*/
	Timer timer;
	
	if (n % standard_frame == 0){
		Mat image;
		cv::resize(image_1080, image, Size(0, 0), 2 / 3., 2 / 3.);
		timer.Tic();
		PoseInfo pose;
		//timer.Tic();
		pose_detect(*posenet, image, pose);
		//timer.Toc();
		//cout << "pose detect cost " << timer.Elasped() / 1000.0 << " s" << endl;
		//timer.Tic();
		int color[18][3] = { { 255, 0, 0 }, { 255, 85, 0 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 170, 0, 255 }, { 170, 0, 255 }, { 255, 0, 170 }, { 255, 0, 170 } };
		int x[18];
		int y[18];

		for (int i = 0; i < pose.subset.size(); i++){
			Student_Info student_info;
			student_info.cur_frame1 = n;
			
			int v = 0;
			float score = float(pose.subset[i][18]) / pose.subset[i][19];
			if (pose.subset[i][19] >= 4 && score >= 0.4){
				for (int j = 0; j < 8; j++){
					if (pose.subset[i][j] == -1){
						x[j] = 0;
						y[j] = 0;
						v = 1;
						if (j == 0 || j == 4 || j == 7){
							v = 0;
						}
					}
					else{
						x[j] = pose.candicate[pose.subset[i][j]][0];
						y[j] = pose.candicate[pose.subset[i][j]][1];
					}
					//student_info.all_points.push_back(Point2f(x[j], y[j]));
				}
				//----------------------------------判断非连续的动作-----------------------------------------------

				detect_discontinuous_behavior(*handsnet, image, pose, student_info, i, v, x, y);

				//--------------------------给人头框-----------------------------------

				if (pose.subset[i][1] != -1 && pose.subset[i][0] != -1){
					float wid1 = 0, wid2 = 0, wid = 0;
					if (pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
						wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
						wid2 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
						wid = MAX(wid1, wid2);
						if (wid == 0)continue;
					}
					if (pose.subset[i][2] != -1 && pose.subset[i][5] == -1){
						wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
						wid2 = wid1;
						wid = wid1;
						if (wid == 0)continue;
					}
					if (pose.subset[i][2] == -1 && pose.subset[i][5] != -1){
						wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
						wid2 = wid1;
						wid = wid1;
						if (wid == 0)continue;
					}

					Rect face_box;
					face_box.x = pose.candicate[pose.subset[i][0]][0] - 2.1*wid / 2;
					face_box.y = pose.candicate[pose.subset[i][0]][1] - 2.5 * wid *1.2 / 2;
					face_box.width = wid*2.4;
					face_box.height = wid*2.4;
					refine(face_box, image);
					student_info.face_bbox = face_box;

				}

				//--------------------use IOU to classify--------------------------------------------------------------------------------

				//----------------obtain a rect range for i person in a new frame---------------------
				if (/*pose.subset[i][0] != -1&&*/pose.subset[i][1] != -1 && pose.subset[i][2] != -1 && pose.subset[i][5] != -1){

					float wid1 = abs(x[1] - x[2]);
					float wid2 = abs(x[1] - x[5]);
					float wid = MAX(wid1, wid2);
					if (wid == 0)continue;

					Rect rect_for_save;
					rect_for_save.x = x[1] - wid - 5;
					rect_for_save.y = y[1] - (wid1 + wid2 - 5);
					rect_for_save.width = wid1 + wid2 + 10;
					rect_for_save.height = 2 * (wid1 + wid2 - 5);
					if (rect_for_save.height < 5)rect_for_save.height = 15;
					//cv::rectangle(image, rect_for_save, Scalar(0, 255, 0), 2, 8, 0);
					refine(rect_for_save, image);
					student_info.body_for_save = rect_for_save;
					int thr = 0;
					if (y[1] < image.size().height / 3)thr = 30;
					else thr = 40;
					Rect cur_rect;
					if (student_info.arm_vertical){
						cur_rect.x = x[1] - wid - 5;
						cur_rect.y = y[1];
						cur_rect.width = wid1 + wid2 + 10;
						cur_rect.height = wid1 + wid2 - 15 + thr;
					}
					else{
						cur_rect.x = x[1] - wid;
						cur_rect.y = y[1];
						cur_rect.width = wid1 + wid2 + 10;
						cur_rect.height = wid1 + wid2;
						if (cur_rect.height < 5)cur_rect.height = 15;
					}
					refine(cur_rect, image);

					//cv::rectangle(image, cur_rect, Scalar(0, 255, 0), 2, 8, 0);
					student_info.body_bbox = cur_rect;
					if (cur_rect.y < image.size().height*0.3 && cur_rect.height>80){
						cur_rect = Rect(0, 0, 1, 1);
					}
					student_info.neck_loc = Point2f(x[1], y[1]);
					if (pose.subset[i][0] != -1){
						student_info.loc = Point2f(x[0], y[0]);
						student_info.front = true;
					}
					else{
						student_info.loc = student_info.neck_loc;
						student_info.front = false;
					}

					//---------------- IOU recognization in a rect range --------------------------------------
					
					std::multimap<float, int, greater<float>>IOU_map;
					for (int j = 0; j < student_valid.size(); j++){
						float cur_IOU = Compute_IOU(students_all[student_valid[j]][0].body_bbox, cur_rect);
						IOU_map.insert(make_pair(cur_IOU, student_valid[j]));
					}
					if (IOU_map.begin()->first > 0){
						int thre = (student_info.loc.y < image.size().height / 2) ? 100 : 150;
						int size1 = students_all[IOU_map.begin()->second].size();
						float dis = abs(students_all[IOU_map.begin()->second][size1 - 1].loc.y - student_info.loc.y);
						if (dis < thre){
							students_all[IOU_map.begin()->second].push_back(student_info);
						}
					}
					else{
						//cv::rectangle(image,student_info.body_bbox,Scalar(0,255,0),1,8,0);
						students_all[69].push_back(student_info);
					}
				}

				/*for (int j = 0; j < 8; j++){
					if (!(x[j] || y[j])){
					continue;
					}
					else{
					cv::circle(image, Point2f(x[j], y[j]), 3, cv::Scalar(color[j][0], color[j][1], color[j][2]), -1);
					}
					}*/

			} //if (pose.subset[i][19] >= 3 && score >= 0.4) end
		}//for (int i = 0; i < pose.subset.size(); i++) end

#if 0
		//-------------------------人脸匹配--------------------------------------------
		//timer.Tic();
		int matching_num = 0;
		//int good_face_num = 0;
		for (int i = 0; i < student_valid.size(); i++){
			if (students_all[student_valid[i]][0].matching_at_end < 100){
				matching_num++;
			}
			/*if (!students_all[student_valid[i]][0].good_face_features.empty()){
				good_face_num++;
			}*/
		}
		//cout << "good face numble: " << good_face_num << endl;
		
		if (standard_faces.size() < max_student_num){
			cout << "standard_faces.size(): " << standard_faces.size() << endl;
			good_face(net3, net4, detector, students_all, student_valid, n, image_1080, standard_faces,max_student_num);
		}
		else{
			//cout << "standard_faces.size(): " << standard_faces.size() << endl;
			cout << "good face done" << endl;
		}

		cout << "matching people numble: " << matching_num << endl;
		if (matching_num < student_valid.size() && standard_faces.size()==max_student_num){
			face_match(net3,net4, detector, students_all, student_valid, n, image_1080, standard_faces);
			//renew_face_match(net4, detector, students_all, student_valid, n, image_1080, standard_faces);
		}
		/*face_match(net4, detector, students_all, student_valid, n, image_1080, standard_faces);
		renew_face_match(net4, detector, students_all, student_valid, n, image_1080, standard_faces);*/

#endif	

		//timer.Toc();
		//cout << "face match cost " << timer.Elasped() / 1000.0 << " s" << endl;

		//----------------------分析行为----------------------------------
		Analys_Behavior(students_all, ID,student_valid, class_info_all, image_1080,image, n);
		
		for (int i = 0; i < student_valid.size(); i++){
			if (students_all[student_valid[i]][0].matching_at_end < 100){
				cv::putText(image, to_string(students_all[student_valid[i]][0].matching_at_end), ID[students_all[student_valid[i]][0].matching_at_end].back().loc, FONT_HERSHEY_COMPLEX, 0.7, Scalar(0, 0, 255), 2);
				//cout << "student  " << student_valid[i] << "matching " << students_all[student_valid[i]][0].matching_at_end << endl;
			}
		}

	
		if (n % (10) == 0){
			writeJson(student_valid, students_all, class_info_all, output, n);
		}
		/*if (n % (10) == 0){
			writeJson1(student_valid, students_all,ID, class_info_all, output, n);
		}*/
       


		/*for (int i = 0; i < student_valid.size(); i++){
			if (students_all[student_valid[i]][0].matching_at_end < 100){
				rectangle(image, students_all[student_valid[i]][0].body_bbox, Scalar(0, 255, 0),2);
			}
		}*/

		//drawGrid(image,student_valid,students_all);
		string output1;	
		output1 = output + "/" + to_string(n) + ".jpg";
		//cv::resize(image, image, Size(0, 0), 1 / 2., 1 / 2.);
		cv::imwrite(output1, image);
		timer.Toc();
		cout << "the " << n << " frame cost " << timer.Elasped() / 1000.0 << " s" << endl;
		n++;
		cout << n << endl;
	} //if (n % standard_frame == 0) end
	return std::make_tuple(students_all, class_info_all);

}
void Student_analy::good_face(jfda::JfdaDetector &detector, Mat &image_1080, int &max_student_num){
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
						std::tuple<bool, float>front_face_or_not = is_front_face(*frontface_net, faceimg, faceinfo.bbox);
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
							string b = output3 + "/" + "0_" + to_string(sco) + "_standard.jpg";
							cv::imwrite(b, faceimg);

							faceinfo.id = student_valid[j];
							Extract(*facefeature_net, faceimg, faceinfo);
							students_all[student_valid[j]][0].good_face_features.assign(faceinfo.feature.begin(), faceinfo.feature.end());
							standard_faces.push_back(faceinfo);
						}
					}
				}
			}
		}
	}
}
void Student_analy::face_match(jfda::JfdaDetector &detector, Mat &image_1080){
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

							std::tuple<bool, float>front_face_or_not = is_front_face(*frontface_net, faceimg, faceinfo.bbox);
							bool front_face = get<0>(front_face_or_not);
							float sco = get<1>(front_face_or_not);
							if (front_face == true){
								Extract(*facefeature_net, faceimg, faceinfo);
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
}

