#include <iostream>
#include <algorithm>
#include <caffe/net.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "student/jfda.hpp"

using std::vector;
using std::string;
using caffe::Blob;
using std::shared_ptr;
using namespace cv;

/*! \brief structure for internal faces */

namespace jfda {

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

class JfdaDetector::Impl {
public:
  Impl()
    : min_size_(24), max_size_(-1), factor_(0.7), th1_(0.1), th2_(0.2), th3_(0.3), max_img_size_(640),
      pnet(NULL), rnet(NULL), onet(NULL), lnet(NULL) {
  }
  vector<FaceInfo> Detect(const cv::Mat& img, vector<FaceInfoInternal>& facem);
  vector<FaceInfo> Detbox(const cv::Mat& img,vector<FaceInfoInternal>& facem);
public:
  int min_size_;
  int max_size_;
  float factor_;
  float th1_, th2_, th3_;
  /*! \brief max size of image's width or height */
  int max_img_size_;
  /*! \brief caffe cnn networks */
  caffe::Net *pnet, *rnet, *onet, *lnet;
};

JfdaDetector::JfdaDetector(const string& pnet, const string& pmodel,
                           const string& rnet, const string& rmodel,
                           const string& onet, const string& omodel,
                           const string& lnet, const string& lmodel,
                           int gpu_device) {
  
  impl_ = new Impl;
  impl_->pnet = new caffe::Net(pnet);
  impl_->pnet->CopyTrainedLayersFrom(pmodel);
  impl_->rnet = new caffe::Net(rnet);
  impl_->rnet->CopyTrainedLayersFrom(rmodel);
  impl_->onet = new caffe::Net(onet);
  impl_->onet->CopyTrainedLayersFrom(omodel);
  impl_->lnet = new caffe::Net(lnet);
  impl_->lnet->CopyTrainedLayersFrom(lmodel);
}

JfdaDetector::~JfdaDetector() {
  delete impl_->pnet;
  delete impl_->rnet;
  delete impl_->onet;
  delete impl_->lnet;
  delete impl_;
}

void JfdaDetector::SetMinSize(int min_size) {
  if (min_size > 8) {
    impl_->min_size_ = min_size;
  }
}

void JfdaDetector::SetMaxSize(int max_size) {
  if (max_size > 0) {
    impl_->max_size_ = max_size;
  }
}

void JfdaDetector::SetImageScaleFactor(float factor) {
  if (factor > 0. && factor < 1.) {
    impl_->factor_ = factor;
  }
}

void JfdaDetector::SetStageThresholds(float th1, float th2, float th3) {
  if (th1 > 0.1 && th1 < 1.) {
    impl_->th1_ = th1;
  }
  if (th2 > 0.1 && th2 < 1.) {
    impl_->th2_ = th2;
  }
  if (th3 > 0.2 && th3 < 1.) {
    impl_->th3_ = th3;
  }
}

void JfdaDetector::SetMaxImageSize(int max_image_size) {
	
  if (max_image_size > 128) {
    impl_->max_img_size_ = max_image_size;
  }
}
std::vector<FaceInfo> JfdaDetector::Detbox(const cv::Mat& img,std::vector<FaceInfoInternal> &faces){
	return impl_->Detbox(img,faces);
}
vector<FaceInfo> JfdaDetector::Detect(const cv::Mat& img, vector<FaceInfoInternal> &facem) {
  // we need color image
  CV_Assert(img.type() == CV_8UC3);
  int w = img.cols;
  int h = img.rows;
  int max_size = impl_->max_img_size_;
  if (std::max(w, h) <= max_size) {
    return impl_->Detect(img,facem);
  }
  // resize origin image
  cv::Mat img_small;
  float scale = 1.;
  if (w > h) {
    scale = w / static_cast<float>(max_size);
    h = static_cast<int>(h / scale);
    w = max_size;
  }
  else {
    scale = h / static_cast<float>(max_size);
    w = static_cast<int>(w / scale);
    h = max_size;
  }
  cv::resize(img, img_small, cv::Size(w, h));
  // detect
  vector<FaceInfo> faces = impl_->Detect(img_small,facem);
  // scale back to origin image
  const int n = faces.size();
  for (int i = 0; i < n; i++) {
    FaceInfo& face = faces[i];
    face.bbox.x = static_cast<int>(face.bbox.x*scale);
    face.bbox.y = static_cast<int>(face.bbox.y*scale);
    face.bbox.width = static_cast<int>(face.bbox.width*scale);
    face.bbox.height = static_cast<int>(face.bbox.height*scale);
    for (int j = 0; j < 5; j++) {
      face.landmark[j].x *= scale;
      face.landmark[j].y *= scale;
    }
  }
  return faces;
}



/*!
 * \brief non-maximum suppression for face regions
 * \param faces     faces
 * \param th        threshold for overlap
 * \param is_union  whether to use union strategy or min strategy
 * \return          result faces after nms
 */
static vector<FaceInfoInternal> Nms(vector<FaceInfoInternal>& faces, float th, bool is_union=true);
/*!
 * \brief do bbox regression, x_new = x_old + dx * l
 * \param faces   faces, write result in place
 */
static void BBoxRegression(vector<FaceInfoInternal>& faces);
/*!
 * \brief make face bbox a square
 * \param faces   faces, write result in place
 */
static void BBoxSquare(vector<FaceInfoInternal>& faces);
/*!
 * \brief locate landmark in original image, points are aligned to top left of the image
 * \param faces   faces, write result in place
 */
static void LocateLandmark(vector<FaceInfoInternal>& faces);

/*!
 * \brief just for debug, rely on `opencv/highgui` module
 */
static void debug_faces(const cv::Mat& img, vector<FaceInfoInternal>& faces) {
  cv::Mat tmp;
  img.copyTo(tmp);
  for (int i = 0; i < faces.size(); i++) {
    FaceInfoInternal& face = faces[i];
    cv::Rect bbox(face.bbox[0], face.bbox[1], face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]);
    cv::rectangle(tmp, bbox, cv::Scalar(0, 0, 255), 2);
  }
  cv::imshow("tmp", tmp);
  cv::waitKey();
}

vector<FaceInfo> JfdaDetector::Impl::Detbox(const cv::Mat& img, vector<FaceInfoInternal>& facem){
	int n = facem.size();
	if (n == 0) {
		return vector<FaceInfo>();
	}
	shared_ptr<Blob> input = onet->blob_by_name("data");
	input->Reshape(n, 3, 48, 48);
	float*  input_data = input->mutable_cpu_data();
	for (int i = 0; i < n; i++) {
		FaceInfoInternal& face = facem[i];
		cv::Rect bbox(face.bbox[0], face.bbox[1], face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]);
		cv::Mat patch = CropPatch(img, bbox);
		cv::resize(patch, patch, cv::Size(48, 48));
		vector<cv::Mat> bgr;
		cv::split(patch, bgr);
		bgr[0].convertTo(bgr[0], CV_32F, 1.f / 128.f, -1.f);
		bgr[1].convertTo(bgr[1], CV_32F, 1.f / 128.f, -1.f);
		bgr[2].convertTo(bgr[2], CV_32F, 1.f / 128.f, -1.f);
		const int bytes = input->offset(0, 1)*sizeof(float);
		memcpy(input_data + input->offset(i, 0), bgr[0].data, bytes);
		memcpy(input_data + input->offset(i, 1), bgr[1].data, bytes);
		memcpy(input_data + input->offset(i, 2), bgr[2].data, bytes);
	}
	onet->Forward();
	shared_ptr<Blob> prob = onet->blob_by_name("prob");
	shared_ptr<Blob> bbox_offset = onet->blob_by_name("bbox_pred");
	shared_ptr<Blob> landmark = onet->blob_by_name("landmark_pred");
	vector<FaceInfoInternal> faces3;
	for (int i = 0; i < n; i++) {
		if (prob->data_at(i, 1, 0, 0) > th3_) {
			FaceInfoInternal face = facem[i];
			face.score = prob->data_at(i, 1, 0, 0);
			for (int j = 0; j < 4; j++) {
				face.offset[j] = bbox_offset->data_at(i, j, 0, 0);
			}
			for (int j = 0; j < 10; j++) {
				face.landmark[j] = landmark->data_at(i, j, 0, 0);
			}
			faces3.push_back(face);
		}
	}
	LocateLandmark(faces3);
	BBoxRegression(faces3);
	faces3 = Nms(faces3, 0.7, false);

	n = faces3.size();
	vector<FaceInfo> result(n);
	for (int i = 0; i < n; i++) {
		FaceInfoInternal& face = faces3[i];
		FaceInfo& res = result[i];
		res.bbox.x = face.bbox[0];
		res.bbox.y = face.bbox[1];
		res.bbox.width = face.bbox[2] - face.bbox[0];
		res.bbox.height = face.bbox[3] - face.bbox[1];
		res.score = face.score;
		res.landmark.clear();
		for (int j = 0; j < 10; j += 2) {
			res.landmark.push_back(cv::Point2f(face.landmark[j], face.landmark[j + 1]));
		}
		// refine the bbox
		float x_min = img.cols;
		float x_max = -1;
		float y_min = img.rows;
		float y_max = -1;
		for (int j = 0; j < 5; j++) {
			x_min = std::min(x_min, res.landmark[j].x);
			x_max = std::max(x_max, res.landmark[j].x);
			y_min = std::min(y_min, res.landmark[j].y);
			y_max = std::max(y_max, res.landmark[j].y);
		}
		float w = x_max - x_min;
		float h = y_max - y_min;
		float r = 0.5;
		float s = std::max(w, h);
		int x = static_cast<int>(x_min - r*s);
		int y = static_cast<int>(y_min - r*s);
		s *= 1.f + 2.f * r;
		res.bbox.x = x;
		res.bbox.y = y;
		res.bbox.width = res.bbox.height = static_cast<int>(s);
	}
	return result;
}

vector<FaceInfo> JfdaDetector::Impl::Detect(const cv::Mat& img,vector<FaceInfoInternal>& facem) {


	float base = 12.f / min_size_;
	int height = img.rows;
	int width = img.cols;
	// get image pyramid scales
	float l = std::min(height, width);
	if (max_size_ > 0) {
		l = std::min(l, static_cast<float>(max_size_));
	}
	l *= base;
	vector<float> scales;
	while (l > 12.f) {
		scales.push_back(base);
		base *= factor_;
		l *= factor_;
	}
	vector<FaceInfoInternal> faces;
	// stage-1
	//Timer timer;
	//timer.Tic();
	for (int i = 0; i < scales.size(); i++) {
		float scale = scales[i];
		int w = static_cast<int>(ceil(scale*width));
		int h = static_cast<int>(ceil(scale*height));
		cv::Mat data;
		vector<cv::Mat> bgr;
		cv::resize(img, data, cv::Size(w, h));
		cv::split(data, bgr);
		bgr[0].convertTo(bgr[0], CV_32F, 1.f / 128.f, -1.f);
		bgr[1].convertTo(bgr[1], CV_32F, 1.f / 128.f, -1.f);
		bgr[2].convertTo(bgr[2], CV_32F, 1.f / 128.f, -1.f);
		shared_ptr<Blob> input = pnet->blob_by_name("data");
		input->Reshape(1, 3, h, w);
		const int bias = input->offset(0, 1, 0, 0);
		const int bytes = bias*sizeof(float);
		memcpy(input->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
		memcpy(input->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
		memcpy(input->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);

		pnet->Forward();
		shared_ptr<Blob> prob = pnet->blob_by_name("prob");
		shared_ptr<Blob> bbox_offset = pnet->blob_by_name("bbox_pred");
		const int hm_h = prob->shape(2);
		const int hm_w = prob->shape(3);
		vector<FaceInfoInternal> faces_this_scale;
		for (int y = 0; y < hm_h; y++) {
			for (int x = 0; x < hm_w; x++) {
				if (prob->data_at(0, 1, y, x) > th1_) {
					FaceInfoInternal face;
					face.bbox[0] = 2 * x;
					face.bbox[1] = 2 * y;
					face.bbox[2] = 2 * x + 12;
					face.bbox[3] = 2 * y + 12;
					face.bbox[0] /= scale;
					face.bbox[1] /= scale;
					face.bbox[2] /= scale;
					face.bbox[3] /= scale;
					face.score = prob->data_at(0, 1, y, x);
					for (int j = 0; j < 4; j++) {
						face.offset[j] = bbox_offset->data_at(0, j, y, x);
					}
					faces_this_scale.push_back(face);
				}
			}
		}
		faces_this_scale = Nms(faces_this_scale, 0.5, true);
		faces.insert(faces.end(), faces_this_scale.begin(), faces_this_scale.end());
	}
	faces = Nms(faces, 0.7, true);
	BBoxRegression(faces);
	BBoxSquare(faces);


  // stage-2
 // Timer timer2;
  //timer2.Tic();
  int n = faces.size();
  if (n == 0) {
    return vector<FaceInfo>();
  }
  shared_ptr<Blob> input = rnet->blob_by_name("data");
  input->Reshape(n, 3, 24, 24);
  float* input_data = input->mutable_cpu_data();
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    cv::Rect bbox(face.bbox[0], face.bbox[1], face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]);
    cv::Mat patch = CropPatch(img, bbox);
    cv::resize(patch, patch, cv::Size(24, 24));
    vector<cv::Mat> bgr;
    cv::split(patch, bgr);
    bgr[0].convertTo(bgr[0], CV_32F, 1.f / 128.f, -1.f);
    bgr[1].convertTo(bgr[1], CV_32F, 1.f / 128.f, -1.f);
    bgr[2].convertTo(bgr[2], CV_32F, 1.f / 128.f, -1.f);
    const int bytes = input->offset(0, 1)*sizeof(float);
    memcpy(input_data + input->offset(i, 0), bgr[0].data, bytes);
    memcpy(input_data + input->offset(i, 1), bgr[1].data, bytes);
    memcpy(input_data + input->offset(i, 2), bgr[2].data, bytes);
  }
  rnet->Forward();
  shared_ptr<Blob> prob = rnet->blob_by_name("prob");
  shared_ptr<Blob> bbox_offset = rnet->blob_by_name("bbox_pred");
  vector<FaceInfoInternal> faces_stage2;
  for (int i = 0; i < n; i++) {
    if (prob->data_at(i, 1, 0, 0) > th2_) {
      FaceInfoInternal face = faces[i];
      face.score = prob->data_at(i, 1, 0, 0);
      for (int j = 0; j < 4; j++) {
        face.offset[j] = bbox_offset->data_at(i, j, 0, 0);
      }
      faces_stage2.push_back(face);
    }
  }
  faces_stage2 = Nms(faces_stage2, 0.7);
  BBoxRegression(faces_stage2);
  BBoxSquare(faces_stage2);
  faces = faces_stage2;
  facem = faces_stage2;
  //timer2.Toc();
 // std::cout << "利大2" << ":" << timer2.Elasped() << "ms" << std::endl;
  // stage-3
  //Timer timer3;
  //timer3.Tic();
  n = faces.size();
  if (n == 0) {
    return vector<FaceInfo>();
  }
  input = onet->blob_by_name("data");
  input->Reshape(n, 3, 48, 48);
  input_data = input->mutable_cpu_data();
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    cv::Rect bbox(face.bbox[0], face.bbox[1], face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]);
    cv::Mat patch = CropPatch(img, bbox);
    cv::resize(patch, patch, cv::Size(48, 48));
    vector<cv::Mat> bgr;
    cv::split(patch, bgr);
    bgr[0].convertTo(bgr[0], CV_32F, 1.f / 128.f, -1.f);
    bgr[1].convertTo(bgr[1], CV_32F, 1.f / 128.f, -1.f);
    bgr[2].convertTo(bgr[2], CV_32F, 1.f / 128.f, -1.f);
    const int bytes = input->offset(0, 1)*sizeof(float);
    memcpy(input_data + input->offset(i, 0), bgr[0].data, bytes);
    memcpy(input_data + input->offset(i, 1), bgr[1].data, bytes);
    memcpy(input_data + input->offset(i, 2), bgr[2].data, bytes);
  }
  onet->Forward();
  prob = onet->blob_by_name("prob");
  bbox_offset = onet->blob_by_name("bbox_pred");
  shared_ptr<Blob> landmark = onet->blob_by_name("landmark_pred");
  vector<FaceInfoInternal> faces_stage3;
  for (int i = 0; i < n; i++) {
    if (prob->data_at(i, 1, 0, 0) > th3_) {
      FaceInfoInternal face = faces[i];
      face.score = prob->data_at(i, 1, 0, 0);
      for (int j = 0; j < 4; j++) {
        face.offset[j] = bbox_offset->data_at(i, j, 0, 0);
      }
      for (int j = 0; j < 10; j++) {
        face.landmark[j] = landmark->data_at(i, j, 0, 0);
      }
	  
      faces_stage3.push_back(face);
    }
  }
  LocateLandmark(faces_stage3);
  BBoxRegression(faces_stage3);
  faces_stage3 = Nms(faces_stage3, 0.7, false);
  faces = faces_stage3;
  //timer3.Toc();
  //std::cout << "利大3" << ":" << timer3.Elasped() << "ms" << std::endl;
  // stage-4
  //Timer timer4;
  //timer4.Tic();
  /*
  n = faces.size();
  if (n == 0) {
    return vector<FaceInfo>();
  }
  input = lnet->blob_by_name("data");
  input->Reshape(n, 15, 24, 24);
  input_data = input->mutable_cpu_data();
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    float* landmark = face.landmark;
    int l = static_cast<int>(std::max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) * 0.25f);
    // every landmark patch
    for (int j = 0; j < 5; j++) {
      float x = landmark[2 * j];
      float y = landmark[2 * j + 1];
      cv::Rect patch_bbox(x - l / 2, y - l / 2, l, l);
      cv::Mat patch = CropPatch(img, patch_bbox);
      cv::resize(patch, patch, cv::Size(24, 24));
      vector<cv::Mat> bgr;
      cv::split(patch, bgr);
      bgr[0].convertTo(bgr[0], CV_32F, 1.f / 128.f, -1.f);
      bgr[1].convertTo(bgr[1], CV_32F, 1.f / 128.f, -1.f);
      bgr[2].convertTo(bgr[2], CV_32F, 1.f / 128.f, -1.f);
      const int bytes = input->offset(0, 1)*sizeof(float);
      memcpy(input_data + input->offset(i, 3 * j + 0), bgr[0].data, bytes);
      memcpy(input_data + input->offset(i, 3 * j + 1), bgr[1].data, bytes);
      memcpy(input_data + input->offset(i, 3 * j + 2), bgr[2].data, bytes);
    }
  }
  lnet->Forward();
  shared_ptr<Blob<float> > landmark_offset = lnet->blob_by_name("landmark_offset");
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    int l = static_cast<int>(std::max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) * 0.25f);
    for (int j = 0; j < 10; j++) {
      face.landmark[j] += landmark_offset->data_at(i, j, 0, 0) * l;
    }
  }
  //timer4.Toc();
 // std::cout << "利大4" << ":" << timer4.Elasped() << "ms" << std::endl;
  // output
  */
  n = faces.size();
  vector<FaceInfo> result(n);
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    FaceInfo& res = result[i];
    res.bbox.x = face.bbox[0];
    res.bbox.y = face.bbox[1];
    res.bbox.width = face.bbox[2] - face.bbox[0];
    res.bbox.height = face.bbox[3] - face.bbox[1];
	res.score = face.score;
    res.landmark.clear();
    for (int j = 0; j < 10; j += 2) {
      res.landmark.push_back(cv::Point2f(face.landmark[j], face.landmark[j + 1]));
    }
    // refine the bbox
    float x_min = img.cols;
    float x_max = -1;
    float y_min = img.rows;
    float y_max = -1;
    for (int j = 0; j < 5; j++) {
      x_min = std::min(x_min, res.landmark[j].x);
      x_max = std::max(x_max, res.landmark[j].x);
      y_min = std::min(y_min, res.landmark[j].y);
      y_max = std::max(y_max, res.landmark[j].y);
    }
    float w = x_max - x_min;
    float h = y_max - y_min;
    float r = 0.5;
    float s = std::max(w, h);
    int x = static_cast<int>(x_min - r*s);
    int y = static_cast<int>(y_min - r*s);
    s *= 1.f + 2.f * r;
    res.bbox.x = x;
    res.bbox.y = y;
    res.bbox.width = res.bbox.height = static_cast<int>(s);
  }
  
  return result;
}

vector<FaceInfoInternal> Nms(vector<FaceInfoInternal>& faces, float th, bool is_union) {
  typedef std::multimap<float, int> ScoreMapper;
  ScoreMapper sm;
  const int n = faces.size();
  vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    sm.insert(ScoreMapper::value_type(face.score, i));
    areas[i] = (face.bbox[3] - face.bbox[1])*(face.bbox[2] - face.bbox[0]);
  }
  vector<int> picked;
  while (!sm.empty()) {
    int last = sm.rbegin()->second;
    picked.push_back(last);
    FaceInfoInternal& last_face = faces[last];
    for (ScoreMapper::iterator it = sm.begin(); it != sm.end();) {
      int idx = it->second;
      FaceInfoInternal& face = faces[idx];
      float x1 = std::max(face.bbox[0], last_face.bbox[0]);
      float y1 = std::max(face.bbox[1], last_face.bbox[1]);
      float x2 = std::min(face.bbox[2], last_face.bbox[2]);
      float y2 = std::min(face.bbox[3], last_face.bbox[3]);
      float w = std::max(0.f, x2 - x1);
      float h = std::max(0.f, y2 - y1);
      float ov = (is_union ? (w*h) / (areas[idx] + areas[last] - w*h)
                           : (w*h) / std::min(areas[idx], areas[last]));
      if (ov > th) {
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
  const int m = picked.size();
  vector<FaceInfoInternal> result(m);
  for (int i = 0; i < m; i++) {
    result[i] = faces[picked[i]];
  }
  return result;
}

void BBoxRegression(vector<FaceInfoInternal>& faces) {
  const int n = faces.size();
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    float w = face.bbox[2] - face.bbox[0];
    float h = face.bbox[3] - face.bbox[1];
    face.bbox[0] += face.bbox[5] * w;
    face.bbox[1] += face.bbox[6] * h;
    face.bbox[2] += face.bbox[7] * w;
    face.bbox[3] += face.bbox[8] * h;
  }
}

void BBoxSquare(vector<FaceInfoInternal>& faces) {
  const int n = faces.size();
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    float x_center = (face.bbox[0] + face.bbox[2]) / 2.f;
    float y_center = (face.bbox[1] + face.bbox[3]) / 2.f;
    float w = face.bbox[2] - face.bbox[0];
    float h = face.bbox[3] - face.bbox[1];
    float l = std::max(w, h);
    face.bbox[0] = x_center - l / 2.f;
    face.bbox[1] = y_center - l / 2.f;
    face.bbox[2] = x_center + l / 2.f;
    face.bbox[3] = y_center + l / 2.f;
  }
}

void LocateLandmark(vector<FaceInfoInternal>& faces) {
  const int n = faces.size();
  for (int i = 0; i < n; i++) {
    FaceInfoInternal& face = faces[i];
    float w = face.bbox[2] - face.bbox[0];
    float h = face.bbox[3] - face.bbox[1];
    for (int j = 0; j < 10; j += 2) {
      face.landmark[j] = face.bbox[0] + face.landmark[j] * w;
      face.landmark[j + 1] = face.bbox[1] + face.landmark[j + 1] * h;
    }
  }
}

}  // namespace jfda
