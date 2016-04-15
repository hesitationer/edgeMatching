/****************************************************************************
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * See the file LICENSE.txt at the root directory of this source
 * distribution for additional information about the GNU GPL.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *
 *****************************************************************************/
#ifndef __HOGDetector_h__
#define __HOGDetector_h__

#include <map>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>


namespace hog {

struct Detection_t {
  //! Detection bounding box.
  cv::Rect m_boundingBox;
  //! Distance.
  double m_dist;
  //! Detection scale.
  int m_scale;
  //! Template index.
  int m_templateIndex;

  Detection_t()
      : m_boundingBox(), m_dist(-1), m_scale(100), m_templateIndex(-1) {
  }

  Detection_t(const cv::Rect &r, const float dist)
      : m_boundingBox(r), m_dist(dist), m_scale(100), m_templateIndex(-1) {
  }

  /*
   * Used for std::sort()
   */
  bool operator<(const Detection_t& d) const {
    return m_dist < d.m_dist;
  }
};

struct Template_info_t {
	//! HOG values.
	cv::Mat m_hog;
	//! Integral HOG.
	std::vector<cv::Mat> m_integralHOG;
	//! Size of the template.
	cv::Size m_size;

	Template_info_t() : m_hog(), m_integralHOG(), m_size() {
	}

	Template_info_t(const cv::Mat &hog, const std::vector<cv::Mat> &integralHOG, const cv::Size &size)
			: m_hog(hog), m_integralHOG(integralHOG), m_size(size) {
	}
};

struct Query_info_t {
	//! Integral HOG.
	std::vector<cv::Mat> m_integralHOG;
	//! Size of the query.
	cv::Size m_size;

	Query_info_t(const std::vector<cv::Mat> &integralHOG, const cv::Size &size)
			: m_integralHOG(integralHOG), m_size(size) {
	}
};

class HOGDetector {
public:

  HOGDetector();

  void detect(const cv::Mat &query_image, std::vector<Detection_t> &detections, const double distThresh,
  		const int offsetX=5, const int offsetY=5);

  void detectMultiScale(const cv::Mat &query_img, std::vector<Detection_t> &detections, const double distThresh,
  		const int offsetX=5, const int offsetY=5, const int minScale=50, const int maxScale=200, const int scaleStep=10);

  cv::Mat get_hogdescriptor_visual_image(cv::Mat& origImg,
      std::vector<float>& descriptorValues, cv::Size winSize, cv::Size cellSize,
      int scaleFactor, double viz_factor);

  inline bool getUseSpatialRejection() const {
  	return m_useSpatialRejection;
  }

  void setTemplateImages(const std::map<int, cv::Mat> &mapOfTemplateImages);

  inline void setUseSpatialRejection(const bool use) {
  	m_useSpatialRejection = use;
  }


private:

  void calculateHOG_rect(cv::Mat &hogCell, std::vector<cv::Mat> integrals,
      cv::Rect global_roi, int nbins, int nbCellX=3, int nbCellY=3/*, int _normalization*/);

  std::vector<cv::Mat> calculateIntegralHOG(const cv::Mat &_in, const int _nbins);

  void detect_impl(const Template_info_t &template_info, const Query_info_t &query_info, const int scale,
  		std::vector<Detection_t> &detections, const double distThresh, const int offsetX=5, const int offsetY=5);


  //! Key: template id - Value: template info.
  std::map<int, Template_info_t> m_mapOfTemplateInfo;
  //! Spatial rejection
  bool m_useSpatialRejection;
};

} //namespace hog

#endif
