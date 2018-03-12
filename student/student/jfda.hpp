#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
using namespace std;
struct  FaceInfo{
	/*! \brief top left, right bottom bbox, [x1, y1, x2, y2] */
	cv::Rect bbox;
	/*! \brief face score */
	float score;
	/*! \brief bbox offset [dx1, dy1, dx2, dy2] */
	/*! \brief landmark, left eye,, right eye, nose, left mouth, right mouth [x1, y1, x2, y2, ...] */
	std::vector<CvPoint2D32f> landmark;
	string path;
	vector<float>feature;
	int id = -1;
};
struct FaceInfoInternal {
	/*! \brief top left, right bottom bbox, [x1, y1, x2, y2] */
	float bbox[4];
	/*! \brief face score */
	float score;
	/*! \brief bbox offset [dx1, dy1, dx2, dy2] */
	float offset[4];
	/*! \brief landmark, left eye,, right eye, nose, left mouth, right mouth [x1, y1, x2, y2, ...] */
	float landmark[10];
};

namespace jfda {
	
/*!
 * \brief multi-task cascade CNN detector for face detection and alignment
 * \note  this object is not thread safe. every thread can only be binded to one gpu.
 *        we can create multi objects on the same gpu in one thread, but can not
 *        create multi objects on different gpus in one thread.
 */
class JfdaDetector {
public:
  /*!
   * \brief constructor
   * \param pnet, pmodel  caffe prototxt and caffemodel for pnet
   * \param rnet, rmodel  caffe prototxt and caffemodel for rnet
   * \param onet, omodel  caffe prototxt and caffemodel for onet
   * \param lnet, lmodel  caffe prototxt and caffemodel for lnet
   * \param gpu_device    gpu id for this model to run, -1 for cpu
   */
  JfdaDetector(const std::string& pnet, const std::string& pmodel,
               const std::string& rnet, const std::string& rmodel,
               const std::string& onet, const std::string& omodel,
               const std::string& lnet, const std::string& lmodel,
               int gpu_device=-1);
  ~JfdaDetector();
  /*!
   * \brief set minimum size of face, default 24
   * \param min_size  minimum size of face
   */
  void SetMinSize(int min_size);
  /*!
   * \brief set maximum size of face, default -1
   * \param max_size  maximum size of face, -1 for unlimited size which be bounded by image size
   */
  void SetMaxSize(int max_size);
  /*!
   * \brief set 3 threshold for pnet, rnet and onet, default 0.6, 0.7, 0.8
   * \param th1, th2, th3   network output threshold
   */
  void SetStageThresholds(float th1, float th2, float th3);
  /*!
   * \brief set image pyraimd scale factor, default 0.7
   * \param factor  image pyraimd scale factor
   */
  void SetImageScaleFactor(float factor);
  /*!
   * \brief set max size of image's width or height, resize will be performed
   * \param max_image_size  max size of image's width or height
   */
  void SetMaxImageSize(int max_image_size);
  /*!
   * \brief detect face
   * \param img   color image to detect face
   * \return      vector of face result
   */
  std::vector<FaceInfo> Detect(const cv::Mat& img, std::vector<FaceInfoInternal> &facem);
  std::vector<FaceInfo> Detbox(const cv::Mat& img,std::vector<FaceInfoInternal> &facem);
private:
  JfdaDetector(const JfdaDetector& other) = delete;
  JfdaDetector& operator=(const JfdaDetector& other) = delete;
  class Impl;
  Impl* impl_;
};

}  // namespace jfda
