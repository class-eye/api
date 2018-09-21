#include <fstream>
#include <thread>
#include <iostream>
#include <numeric>
#include <string>
#include <map>
#include <algorithm>
#include "student/ssh_face.hpp"
#include "caffe/caffe.hpp"
#include <student/fs.hpp>
#include "student/student.hpp"
#include "student/functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using namespace cv;
using namespace std;

vector<int> NonMaximumSuppression(const vector<float>& scores,
	const vector<BBox>& bboxes,
	const float nms_th) {
	typedef std::multimap<float, int> ScoreMapper;
	ScoreMapper sm;
	const int n = scores.size();
	vector<float> areas(n);
	for (int i = 0; i < n; i++) {
		areas[i] = (bboxes[i].x2 - bboxes[i].x1 + 1)*(bboxes[i].y2 - bboxes[i].y1 + 1);
		sm.insert(ScoreMapper::value_type(scores[i], i));
	}
	vector<int> picked;
	while (!sm.empty()) {
		int last_idx = sm.rbegin()->second;
		picked.push_back(last_idx);
		const BBox& last = bboxes[last_idx];
		for (ScoreMapper::iterator it = sm.begin(); it != sm.end();) {
			int idx = it->second;
			const BBox& curr = bboxes[idx];
			float x1 = max(curr.x1, last.x1);
			float y1 = max(curr.y1, last.y1);
			float x2 = min(curr.x2, last.x2);
			float y2 = min(curr.y2, last.y2);
			float w = max(0.f, x2 - x1 + 1);
			float h = max(0.f, y2 - y1 + 1);
			float ov = (w*h) / (areas[idx] + areas[last_idx] - w*h);
			if (ov > nms_th) {
				ScoreMapper::iterator it_ = it;
				it_++;
				sm.erase(it);
				it = it_;
			}
			else {
				it++;
			}
		}
	}
	return picked;
}
void SqaureFaceBox(vector<BBox>& faces) {
	for (auto& face : faces) {
		float x_center = (face.x1 + face.x2) / 2;
		float y_center = (face.y1 + face.y2) / 2;
		float w = face.x2 - face.x1;
		float h = face.y2 - face.y1;
		float l = max(w, h);
		face.x1 = x_center - l / 2;
		face.y1 = y_center - l / 2;
		face.x2 = x_center + l / 2;
		face.y2 = y_center + l / 2;
	}
}
//void SshFaceDetWorker::Config::Check() const {
//	CHECK_GE(min_size, 100);
//	CHECK_GE(max_size, 200);
//	CHECK_GT(nms_th, 0);
//	CHECK_LT(nms_th, 1);
//	CHECK_GT(score_th, 0);
//	CHECK_LT(score_th, 1);
//	/*CHECK(fs::IsExists(fs::JoinPath({ model_dir, net_prefix + ".prototxt" })));
//	CHECK(fs::IsExists(fs::JoinPath({ model_dir, net_prefix + ".caffemodel" })));*/
//}

void SshFaceDetWorker::SetConfig(Config config) {
	//config.Check();
	config_ = config;
}

//void SshFaceDetWorker::Initialize() {
//	//SetCaffeMode(config_.gpu_id);
//	if (config_.gpu_id < 0) {
//		caffe::SetMode(caffe::CPU, -1);
//	}
//	else {
//		if (caffe::GPUAvailable()){
//			cout << "GPU Mode" << endl;
//			caffe::SetMode(caffe::GPU, config_.gpu_id);
//		}
//	}
//	net_ = new caffe::Net(fs::JoinPath({ config_.model_dir, config_.net_prefix + ".prototxt" }));
//	net_->CopyTrainedLayersFrom(fs::JoinPath({ config_.model_dir, config_.net_prefix + ".caffemodel" }));
//
//	//net_.reset(new caffe::Net(fs::JoinPath({ config_.model_dir, config_.net_prefix + ".prototxt" })));
//	//net_->CopyTrainedLayersFrom(fs::JoinPath({ config_.model_dir, config_.net_prefix + ".caffemodel" }));
//
//}
////
////void SshFaceDetWorker::Deinitialize() {
////	net_.reset();
////	caffe::MemPoolClear();
////}
//void SshFaceDetWorker::Deinitialize(){
//	delete net_;
//	caffe::MemPoolClear();
//}

SshFaceDetWorker::SshFaceDetWorker(const string& ssh_net, const string &ssh_model,
	const string& real_front_face_net, const string& real_front_face_model,
	const string& face_feature_net, const string& face_feature_model,
	const string& pose_feat_net, const string& pose_feat_model)
{
	if (config_.gpu_id < 0) {
		caffe::SetMode(caffe::CPU, -1);
	}
	else {
		if (caffe::GPUAvailable()){
			cout << "GPU Mode" << endl;
			caffe::SetMode(caffe::GPU, config_.gpu_id);
		}
	}

	matching.resize(70);

	net_ = new caffe::Net(ssh_net);
	net_->CopyTrainedLayersFrom(ssh_model);

	real_frontface_net = new caffe::Net(real_front_face_net);
	real_frontface_net->CopyTrainedLayersFrom(real_front_face_model);

	facefeature_net = new caffe::Net(face_feature_net);
	facefeature_net->CopyTrainedLayersFrom(face_feature_model);

	pose_net = new caffe::Net(pose_feat_net);
	pose_net->CopyTrainedLayersFrom(pose_feat_model);

}
SshFaceDetWorker::~SshFaceDetWorker(){
	delete net_; 
	delete real_frontface_net;
	delete facefeature_net;
	delete pose_net;
	caffe::MemPoolClear();
}
static float ComputeScaleFactor(int width, int height, int target_size, int max_size) {
	int mmin = min(width, height);
	int mmax = max(width, height);
	float scale_factor = static_cast<float>(target_size) / mmin;
	if (scale_factor * mmax > max_size) {
		scale_factor = static_cast<float>(max_size) / mmax;
	}
	return scale_factor;
}

vector<BBox>SshFaceDetWorker::detect(cv::Mat img)
{

	/*caffe::Profiler* profiler = caffe::Profiler::Get();
	profiler->ScopeStart("SshFaceDetWorker");*/
	float scale_factor = ComputeScaleFactor(img.cols, img.rows, config_.min_size, config_.max_size);

	cv::Mat img_resized;
	cv::resize(img, img_resized, cv::Size(0, 0), scale_factor, scale_factor);

	// prepare input data
	std::vector<cv::Mat> bgr;
	cv::split(img_resized, bgr);
	bgr[0].convertTo(bgr[0], CV_32F, 1.f, -102.9801f);
	bgr[1].convertTo(bgr[1], CV_32F, 1.f, -115.9465f);
	bgr[2].convertTo(bgr[2], CV_32F, 1.f, -122.7717f);
	std::shared_ptr<Blob> input_data = net_->blob_by_name("data");
	input_data->Reshape(1, 3, img_resized.rows, img_resized.cols);
	const int bias = input_data->offset(0, 1, 0, 0);
	const int bytes = bias*sizeof(float);
	memcpy(input_data->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
	memcpy(input_data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
	memcpy(input_data->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);
	std::shared_ptr<Blob> im_info = net_->blob_by_name("im_info");
	im_info->mutable_cpu_data()[0] = img_resized.rows;
	im_info->mutable_cpu_data()[1] = img_resized.cols;
	im_info->mutable_cpu_data()[2] = scale_factor;
	/*string nam = "../inputimg1/" + to_string(n) + ".jpg";
	cv::imwrite(nam, img_resized);*/
	// forward
	net_->Forward();
	// post process
	std::shared_ptr<Blob> prob = net_->blob_by_name("ssh_cls_prob");
	std::shared_ptr<Blob> bbox = net_->blob_by_name("ssh_boxes");

	std::vector<float> scores;
	std::vector<BBox> bboxes;

	int num_rois = prob->num();
	for (int i = 0; i < num_rois; i++) {
		float score = prob->data_at(i, 0, 0, 0);
		if (score > config_.score_th) {
			scores.push_back(score);
			BBox face;
			face.x1 = bbox->data_at(i, 1, 0, 0) / scale_factor;
			face.y1 = bbox->data_at(i, 2, 0, 0) / scale_factor;
			face.x2 = bbox->data_at(i, 3, 0, 0) / scale_factor;
			face.y2 = bbox->data_at(i, 4, 0, 0) / scale_factor;
			bboxes.push_back(face);
		}
	}
	// nms
	auto keep = NonMaximumSuppression(scores, bboxes, config_.nms_th);
	const int num_picked = keep.size();
	vector<BBox>all_bbox;
	if (num_picked > 0){
		for (int i = 0; i < num_picked; i++){
			BBox& bbox = bboxes[keep[i]];
			/*cv::Rect rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1 + 1, bbox.y2 - bbox.y1 + 1);
			cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
			char buff[300];
			sprintf(buff, "%s: %.2f", kClassNames[c], scores[i]);
			cv::putText(img, buff, cv::Point(bbox.x1, bbox.y1), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));*/
			all_bbox.push_back(bbox);
		}
	}

	// sqaure
	if (config_.square_box) {
		SqaureFaceBox(all_bbox);
	}
	//profiler->ScopeEnd();
	return all_bbox;

}

void SshFaceDetWorker::GetStandaredFeats_ssh(vector<BBox>faces, jfda::JfdaDetector &detector, Mat &image_1080){
	Timer timer;
	timer.Tic();

	//��̬����

	/*PoseInfo pose;
	Mat frame;
	cv::resize(image_1080, frame, Size(0, 0), 2 / 3., 2 / 3.);
	timer.Tic();
	pose_detect(*pose_net, frame, pose);
	timer.Toc();
	cout << "pose cost " << timer.Elasped() / 1000.0 << " s" << endl;
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

				string b = "../pose_stand/" +to_string(n)+".jpg";
				cv::imwrite(b, frame);
			}
		}
	}*/

	for (int i = 0; i < faces.size(); i++){
		//����ȷ��λ��
		
		Rect bigger_face;
		int width = faces[i].x2 - faces[i].x1;
		int height = faces[i].y2 - faces[i].y1;
		bigger_face.x = faces[i].x1 - width / 2;
		bigger_face.y = faces[i].y1 - height / 2;
		bigger_face.width = width * 2;
		bigger_face.height = height * 2;
		Rect ssh_face(faces[i].x1, faces[i].y1, width, height);
		refine(bigger_face, image_1080);
		rectangle(image_1080, bigger_face, Scalar(0, 0, 255), 2);		
		cv::putText(image_1080, to_string(i), Point(faces[i].x1, faces[i].y1), FONT_HERSHEY_COMPLEX, 0.7, Scalar(0, 255, 0), 2);
		
		//�õ���������
		Mat faceimg = image_1080(bigger_face);
		
		vector<FaceInfoInternal>facem;
		vector<FaceInfo> faces_jfda = detector.Detect(faceimg, facem);
		if (faces_jfda.size() == 1){
			FaceInfo faceinfo = faces_jfda[0];
			faceinfo.sdbbox = bigger_face;
			std::tuple<bool, float>real_front_face_or_not = real_front_face(*real_frontface_net, faceimg, faces_jfda[0].bbox);
			faceinfo.sco[1] = get<1>(real_front_face_or_not);
			Mat ssh_img = CropPatch(faceimg, faces_jfda[0].bbox);
			Extract(*facefeature_net, faceimg, faceinfo);
			standard_faces.push_back(faceinfo);
			string output3 = "../standard_face/" + to_string(i);
			if (!fs::IsExists(output3)){
				fs::MakeDir(output3);
			}
			string output4 = output3 + "/" + to_string(i) + ".jpg";
			cv::imwrite(output4, ssh_img);
		}
		
		//��ȡssh��������
		
		/*refine(ssh_face, image_1080);
		Mat ssh_img = image_1080(ssh_face);
		Extract(*facefeature_net, ssh_img, face);
		standard_faces.push_back(face);
		string output3 = "../standard_face/" + to_string(i);
		if (!fs::IsExists(output3)){
			fs::MakeDir(output3);
		}
		string output4 = output3 + "/" + to_string(i) + ".jpg";
		cv::imwrite(output4, ssh_img);*/

	}
	timer.Toc();
	cout << "standard cost " << timer.Elasped() / 1000.0 << " s" << endl;
	n++;
	string output = "../standard"+to_string(n)+".jpg";
	imwrite(output, image_1080);

}

int SshFaceDetWorker::good_face_ssh(vector<BBox>faces, jfda::JfdaDetector &detector, Mat &image_1080){
	Timer timer;
	timer.Tic();
	for (int i = 0; i < faces.size(); i++){
		Rect bigger_face;
		int width = faces[i].x2 - faces[i].x1;
		int height = faces[i].y2 - faces[i].y1;
		bigger_face.x = faces[i].x1 - width / 2;
		bigger_face.y = faces[i].y1 - height / 2;
		bigger_face.width = width * 2;
		bigger_face.height = height * 2;
		Rect ssh_face(faces[i].x1, faces[i].y1, width, height);
		refine(bigger_face, image_1080);
		std::multimap<float, int, greater<float>>IOU_map;

		//IOU������

		for (int j = 0; j < standard_faces.size(); j++){
			float cur_IOU = Compute_IOU(standard_faces[j].sdbbox, bigger_face);
			//cout << standard_faces[j].bbox << "     " << bigger_face << endl;
			IOU_map.insert(make_pair(cur_IOU, j));
		}
		if (IOU_map.begin()->first > 0){
			Mat faceimg = image_1080(bigger_face);
			vector<FaceInfoInternal>facem;
			vector<FaceInfo> faces_jfda = detector.Detect(faceimg, facem);
			if (faces_jfda.size() == 1){
				FaceInfo faceinfo = faces_jfda[0];
				std::tuple<bool, float>real_front_face_or_not = real_front_face(*real_frontface_net, faceimg, faces_jfda[0].bbox);
				float real_sco = get<1>(real_front_face_or_not);
				if (real_sco > standard_faces[IOU_map.begin()->second].sco[1]){
					faceinfo.sco[1] = real_sco;
					faceinfo.sdbbox = bigger_face;

					cout << "replace " << IOU_map.begin()->second << endl;

					/*Rect ssh_face(faces[i].x1, faces[i].y1, width, height);
					refine(ssh_face, image_1080);*/
					Mat ssh_img = CropPatch(faceimg, faces_jfda[0].bbox);
					Extract(*facefeature_net, faceimg, faceinfo);
					standard_faces[IOU_map.begin()->second] = faceinfo;
					string output3 = "../standard_face/" + to_string(IOU_map.begin()->second);
					string output4 = output3 + "/" + to_string(IOU_map.begin()->second) + "_" + to_string(n) + ".jpg";
					cv::imwrite(output4, ssh_img);
				}
			}
		}
	
	}
	timer.Toc();
	cout << n<<" good face cost " << timer.Elasped() / 1000.0 << " s" << endl;
	n++;
	if (n > 120)return 1;
	else return 0;
}

int SshFaceDetWorker::face_match(vector<BBox>faces, jfda::JfdaDetector &detector, Mat &image_1080){
	int count = 0;
	for (int i = 0; i < faces.size(); i++){
		cout << 1 << endl;
		Rect bigger_face;
		int width = faces[i].x2 - faces[i].x1;
		int height = faces[i].y2 - faces[i].y1;
		bigger_face.x = faces[i].x1 - width / 2;
		bigger_face.y = faces[i].y1 - height / 2;
		bigger_face.width = width * 2;
		bigger_face.height = height * 2;
		Rect ssh_face(faces[i].x1, faces[i].y1, width, height);
		refine(bigger_face, image_1080);
		Mat faceimg = image_1080(bigger_face);
		vector<FaceInfoInternal>facem;
		vector<FaceInfo> faces_jfda = detector.Detect(faceimg, facem);
		/*Mat patch = CropPatch(image_1080, ssh_face);
		FaceInfo face;
		face.bbox = ssh_face;
		Extract(*facefeature_net, patch, face);*/
		cout << 2 << endl;
		if (faces_jfda.size() == 1){
			FaceInfo faceinfo = faces_jfda[0];
			//Mat ssh_img = CropPatch(faceimg, faces_jfda[0].bbox);
			Extract(*facefeature_net, faceimg, faceinfo);
			std::multimap<float, int, greater<float>>feat_map;
			float distance = 0.;
			for (int j = 0; j < standard_faces.size(); j++){
				if (standard_faces[j].is_matched == false){
					featureCompare(standard_faces[j].feature, faceinfo.feature, distance);
					feat_map.insert(make_pair(distance, j));
				}
			}
			auto iter = feat_map.begin();
			auto iter1 = std::next(iter, 1);
			//cout << "score: " << iter->first << endl;
			if (iter->first>0.45 && (iter->first - iter1->first) > 0.05){
				matching[iter->second].push_back(faceinfo);
				standard_faces[iter->second].is_matched = true;
				string output3 = "../standard_face/" + to_string(iter->second);
				string output4 = output3 + "/" + to_string(iter->second) + "_" + to_string(n) + ".jpg";
				Mat ssh_img = CropPatch(faceimg, faces_jfda[0].bbox);
				imwrite(output4, ssh_img);
			}
		}
		cout << 3 << endl;
	}
	for (auto sta : standard_faces){
		if (sta.is_matched == true)count++;
	}
	cout << count << " is Matched" << endl;
	n++;
	return count;
}