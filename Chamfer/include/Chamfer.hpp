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
#ifndef __ChamferMatching_h__
#define __ChamferMatching_h__

#include <map>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


struct Detection_t {
  //! Detection bounding box
  cv::Rect m_boundingBox;
  //! Corresponding Chamfer distance
  float m_chamferDist;

  Detection_t(const cv::Rect &r, const float dist) : m_boundingBox(r), m_chamferDist(dist) {
  }

  /*
   * Used for std::sort()
   */
  bool operator<(const Detection_t& d) const {
    return m_chamferDist < d.m_chamferDist;
  }
};

struct less_than_area {
  inline bool operator()(const Detection_t& detection1,
      const Detection_t& detection2) {
    return (detection1.m_boundingBox.area() < detection2.m_boundingBox.area());
  }
};

struct Line_info_t {
  double m_length;
  cv::Point m_PointEnd;
  cv::Point m_pointStart;
  double m_rho;
  double m_theta;

  Line_info_t(const double length, const double rho, const double theta, const cv::Point &start, const cv::Point &end)
    : m_length(length), m_PointEnd(end), m_pointStart(start), m_rho(rho), m_theta(theta) {
  }

  friend std::ostream& operator<<(std::ostream& stream, const Line_info_t& line) {
    stream << "Lenght=" << line.m_length << " ; rho=" << line.m_rho << " ; theta="
        << (line.m_theta * 180.0 / M_PI);
    return stream;
  }
};

struct Template_info_t {
  //! List of contours, each contour is a list of points
  std::vector<std::vector<cv::Point> > m_contours;
  //! Distance transform image
  cv::Mat m_distImg;
  //! Corresponding edge orientation for each point for each contour
  std::vector<std::vector<float> > m_edgesOrientation;
  //! Image that contains at each location the edge orientation value of the corresponding
  //! nearest edge for the current pixel location
  cv::Mat m_mapOfEdgeOrientation;
  //! Cluster each line orientation according to his polar angle
  std::map<int, Line_info_t> m_mapOfLines;
  //! Image mask
  cv::Mat m_mask;

  Template_info_t(const std::vector<std::vector<cv::Point> > &contours, const cv::Mat &dist,
      const cv::Mat &edgeOriImg, const std::vector<std::vector<float> > &edgesOri, const cv::Mat &mask)
      : m_contours(contours), m_distImg(dist), m_mapOfEdgeOrientation(edgeOriImg),
        m_edgesOrientation(edgesOri), m_mask(mask) {
  }

  Template_info_t()
      : m_contours(), m_distImg(), m_mapOfEdgeOrientation(), m_edgesOrientation(), m_mask() {
  }
};

struct Query_info_t {
  //! List of contours, each contour is a list of points
  std::vector<std::vector<cv::Point> > m_contours;
  //! Distance transform image
  cv::Mat m_distImg;
  //! Corresponding edge orientation for each point for each contour
  std::vector<std::vector<float> > m_edgesOrientation;
  //! Query Image
  cv::Mat m_img;
  //! Image that contains at each location the edge orientation value of the corresponding
  //! nearest edge for the current pixel location
  cv::Mat m_mapOfEdgeOrientation;
  //! Image that contains the id corresponding to the nearest edge point
  cv::Mat m_mapOfLabels;

  Query_info_t(const std::vector<std::vector<cv::Point> > &contours, const cv::Mat &dist, const cv::Mat &img,
      const cv::Mat &edgeOriImg, const std::vector<std::vector<float> > &edgesOri, const cv::Mat &labels)
      : m_contours(contours), m_distImg(dist), m_edgesOrientation(edgesOri), m_img(img),
        m_mapOfEdgeOrientation(edgeOriImg), m_mapOfLabels(labels) {
  }

  Query_info_t()
      : m_contours(), m_distImg(), m_img(), m_mapOfEdgeOrientation(), m_mapOfLabels() {
  }
};


class ChamferMatching {
public:
  enum MatchingType {
    edgeMatching, edgeForwardBackwardMatching, fullMatching
  };

  ChamferMatching();

  static void computeDistanceTransform(const cv::Mat &img, cv::Mat &dist_img, cv::Mat &labels);

  /*
   * Compute the map that links for each contour id the corresponding indexes i,j in
   * the vector of vectors.
   */
  static void computeEdgeMapIndex(const std::vector<std::vector<cv::Point> > &contours,
      const cv::Mat &labels, std::map<int, std::pair<int, int> > &mapOfIndex);

  /*
   * Create the template mask.
   */
  static void createTemplateMask(const cv::Mat &img, cv::Mat &mask, const double threshold=50.0);

  void detect(const cv::Mat &img_template, const cv::Mat &img_query, std::vector<Detection_t> &bbDetections,
      const bool useOrientation, const float distanceThresh=50.0f);
  void detectMultiScale(const cv::Mat &img_template, const cv::Mat &img_query, std::vector<Detection_t> &bbDetections,
      const bool useOrientation, const float distanceThresh=50.0f, const double scaleFactor=0.1,
      const double minScale=0.4, const double maxScale=2.0);

  static void filterSingleContourPoint(std::vector<std::vector<cv::Point> > &contours, const size_t min=3);

  /*
   * Get the list of contour points.
   */
  static void getContours(const cv::Mat &img, std::vector<std::vector<cv::Point> > &contours, const double threshold=50.0);

  /*
   * Compute for each contour point the corresponding edge orientation.
   * For the current contour point, use the previous and next point to
   * compute the edge orientation.
   */
  static void getContoursOrientation(const std::vector<std::vector<cv::Point> > &contours,
      std::vector<std::vector<float> > &contoursOrientation);

  inline MatchingType getMatchingType(const MatchingType &type) const {
    return m_matchingType;
  }

  inline void setMatchingType(const MatchingType &type) {
    m_matchingType = type;
  }


  double m_cannyThreshold;


private:

  void computeCanny(const cv::Mat &img, cv::Mat &edges, const double threshold);

  double computeChamferDistance(const int offsetX, const int offsetY, cv::Mat &img_res,
      const bool useOrientation=false, const float lambda=1.0f,
      const float weight_foward=1.0f, const float weight_backward=1.0f);

  double computeFullChamferDistance(const int offsetX, const int offsetY, cv::Mat &img_res,
      const bool useOrientation=false, const float lambda=5.0f);

  void computeMatchingMap(cv::Mat &chamferMap, const bool useOrientation=false, const int xStep=5,
      const int yStep=5, float lambda=0.5);

  void createMapOfEdgeOrientations(const cv::Mat &img, const cv::Mat &labels, cv::Mat &mapOfEdgeOrientations);

  void detect_impl(const cv::Mat &img_template, std::vector<Detection_t> &bbDetections,
      const bool useOrientation);

  void groupDetections(const std::vector<Detection_t> &detections, std::vector<Detection_t> &groupedDetections);

  void retainDetections(std::vector<Detection_t> &bbDetections, const float threshold);

  void nonMaximaSuppression(const std::vector<Detection_t> &detections, std::vector<Detection_t> &maximaDetections);

  void prepareQuery(const cv::Mat &img_query);
  void prepareTemplate(const cv::Mat &img_template);


  MatchingType m_matchingType;
  Query_info_t m_query_info;
  Template_info_t m_template_info;
};

#endif
