#include <numeric>
#include <map>
#include <iostream>
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
			float x1 = std::max(curr.x1, last.x1);
			float y1 = std::max(curr.y1, last.y1);
			float x2 = std::min(curr.x2, last.x2);
			float y2 = std::min(curr.y2, last.y2);
			float w = std::max(0.f, x2 - x1 + 1);
			float h = std::max(0.f, y2 - y1 + 1);
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
		float l = std::max(w, h);
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
	const string& front_face_net, const string& front_face_model,
	const string& real_front_face_net, const string& real_front_face_model)
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
	net_ = new caffe::Net(ssh_net);
	net_->CopyTrainedLayersFrom(ssh_model);

	real_frontface_net = new caffe::Net(real_front_face_net);
	real_frontface_net->CopyTrainedLayersFrom(real_front_face_model);

	facefeature_net = new caffe::Net(real_front_face_net);
	facefeature_net->CopyTrainedLayersFrom(real_front_face_model);
}
SshFaceDetWorker::~SshFaceDetWorker(){
	delete net_; 
	delete real_frontface_net;
	delete facefeature_net;
}
static float ComputeScaleFactor(int width, int height, int target_size, int max_size) {
	int mmin = std::min(width, height);
	int mmax = std::max(width, height);
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

int SshFaceDetWorker::GetStandaredFeats_ssh(SshFaceDetWorker &ssh, jfda::JfdaDetector &detector, Mat &image_1080, int &max_student_num){
	vector<BBox>faces = ssh.detect(image_1080);
	if (faces.size() >= max_student_num){
		cout << faces.size() << endl;
		for (int i = 0; i < faces.size(); i++){
			//用脸确定位置
			FaceInfo face;
			Rect bigger_face;
			int width = faces[i].x2 - faces[i].x1;
			int height = faces[i].y2 - faces[i].y1;
			bigger_face.x = faces[i].x1 - width / 2;
			bigger_face.y = faces[i].y1 - height / 2;
			bigger_face.width = width * 2;
			bigger_face.height = height * 2;
			refine(bigger_face, image_1080);
			face.bbox = bigger_face;
			//得到正脸分数
			Mat faceimg = image_1080(bigger_face);
			vector<FaceInfoInternal>facem;
			vector<FaceInfo> faces_jfda = detector.Detect(faceimg, facem);
			if (faces.size() == 1){	
				std::tuple<bool, float>real_front_face_or_not = real_front_face(*real_frontface_net, faceimg, faces_jfda[0].bbox);
				face.sco[1] = get<1>(real_front_face_or_not);
			}
			//提取ssh人脸特征
			Rect ssh_face(faces[i].x1, faces[i].y1, width, height);
			Mat ssh_img = image_1080(ssh_face);
			Extract(*facefeature_net, ssh_img, face);

			standard_faces.push_back(face);
		}
		return 1;
	}
	return 0;
}

int SshFaceDetWorker::good_face_ssh(SshFaceDetWorker &ssh, jfda::JfdaDetector &detector,Mat &image_1080, int &max_student_num){
	vector<BBox>faces=ssh.detect(image_1080);
	for (int i = 0; i < faces.size(); i++){
		Rect bigger_face;
		int width = faces[i].x2 - faces[i].x1;
		int height = faces[i].y2 - faces[i].y1;
		bigger_face.x = faces[i].x1-width/2;
		bigger_face.y = faces[i].y1 - height / 2;
		bigger_face.width = width * 2;
		bigger_face.height = height * 2;

		std::multimap<float, int,greater<float>>IOU_map;


		Mat faceimg = image_1080(bigger_face);
		vector<FaceInfoInternal>facem;
		vector<FaceInfo> faces = detector.Detect(faceimg, facem);
		if (faces.size() == 1){
			FaceInfo faceinfo = faces[0];
			std::tuple<bool, float>real_front_face_or_not = real_front_face(*real_frontface_net, faceimg, faceinfo.bbox);
			float real_sco = get<1>(real_front_face_or_not);
			string output1 = "../standard_face/";
		}
}