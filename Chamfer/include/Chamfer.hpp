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
#ifndef __ChamferMatcher_h__
#define __ChamferMatcher_h__

#include <map>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define DEBUG 0


struct Detection_t {
  //! Detection bounding box.
  cv::Rect m_boundingBox;
  //! Corresponding Chamfer distance.
  float m_chamferDist;
  //! Detection scale.
  int m_scale;
  //! Template index.
  int m_templateIndex;

  Detection_t()
  : m_boundingBox(), m_chamferDist(-1), m_scale(-1), m_templateIndex(-1) {
  }

  Detection_t(const cv::Rect &r, const float dist, const int scale)
  : m_boundingBox(r), m_chamferDist(dist), m_scale(scale), m_templateIndex(-1) {
  }

  Detection_t(const cv::Rect &r, const float dist, const int scale, const int index)
  : m_boundingBox(r), m_chamferDist(dist), m_scale(scale), m_templateIndex(index) {
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
  cv::Point m_pointEnd;
  cv::Point m_pointStart;
  double m_rho;
  double m_theta;

  Line_info_t(const double length, const double rho, const double theta, const cv::Point &start, const cv::Point &end)
  : m_length(length), m_pointEnd(end), m_pointStart(start), m_rho(rho), m_theta(theta) {
  }

  friend std::ostream& operator<<(std::ostream& stream, const Line_info_t& line) {
    stream << "Lenght=" << line.m_length << " ; rho=" << line.m_rho << " ; theta="
        << (line.m_theta * 180.0 / M_PI);
    return stream;
  }
};

struct Template_info_t {
  //! List of contours, each contour is a list of points.
  std::vector<std::vector<cv::Point> > m_contours;
  //! Distance transform image.
  cv::Mat m_distImg;
  //! Corresponding edge orientation for each point for each contour.
  std::vector<std::vector<float> > m_edgesOrientation;
  //! Descriptors.
  std::vector<std::pair<float, float> > m_gridDescriptors;
  //! Location of the descriptor on the grid.
  std::vector<cv::Point> m_gridDescriptorsLocations;
  //! Size for the descriptors grid.
  cv::Size m_gridDescriptorsSize;
  //! Image that contains at each location the edge orientation value of the corresponding
  //! nearest edge for the current pixel location.
  cv::Mat m_mapOfEdgeOrientation;
  //  //! Cluster each line orientation according to his polar angle
  //  std::map<int, Line_info_t> m_mapOfLines;
  //! Image mask.
  cv::Mat m_mask;
  //! Query ROI: rectangle in the query image where we want to search the template.
  cv::Rect m_queryROI;
  //! Template location in the query image, used when dealing with template poses (one location = one pose).
  cv::Rect m_templateLocation;
  //! Vector of contours approximated by lines.
  std::vector<std::vector<Line_info_t> > m_vectorOfContourLines;

  Template_info_t(const std::vector<std::vector<cv::Point> > &contours, const cv::Mat &dist,
      const std::vector<std::vector<float> > &edgesOri, const cv::Size &gridDescriptorSize,
      const cv::Mat &edgeOriImg, const cv::Mat &mask, const std::vector<std::vector<Line_info_t> > &contourLines)
  : m_contours(contours), m_distImg(dist), m_edgesOrientation(edgesOri), m_gridDescriptors(), m_gridDescriptorsLocations(),
    m_gridDescriptorsSize(gridDescriptorSize), m_mapOfEdgeOrientation(edgeOriImg), m_mask(mask),
    m_queryROI(0,0,-1,-1), m_templateLocation(0,0,-1,-1), m_vectorOfContourLines(contourLines) {
    computeGridLocations();
  }

  Template_info_t()
  : m_contours(), m_distImg(), m_edgesOrientation(), m_gridDescriptors(), m_gridDescriptorsLocations(),
    m_gridDescriptorsSize(4,4), m_mapOfEdgeOrientation(), m_mask(), m_queryROI(0,0,-1,-1),
    m_templateLocation(0,0,-1,-1), m_vectorOfContourLines() {
  }

private:
  void computeGridLocations() {
    const int nbGridX = m_gridDescriptorsSize.width, nbGridY = m_gridDescriptorsSize.height;
    int min_step_size = 2;

    if( m_distImg.cols / (nbGridX+1) < min_step_size || m_distImg.rows / (nbGridY+1) < min_step_size ) {
      std::cerr << "Template is too small in order to compute the grid locations !" << std::endl;
    } else {
      float step_x = m_distImg.cols / (float) (nbGridX+1);
      float step_y = m_distImg.rows / (float) (nbGridY+1);

      for(int i = 0; i < nbGridY; i++) {
        for(int j = 0; j < nbGridX; j++) {
          cv::Point location( (j+1)*step_x, (i+1)*step_y );
          m_gridDescriptorsLocations.push_back(location);

          float dist = m_distImg.ptr<float>(location.y)[location.x];
          float orientation = m_mapOfEdgeOrientation.ptr<float>(location.y)[location.x];
          m_gridDescriptors.push_back( std::pair<float, float>(dist, orientation) );
        }
      }
    }
  }
};

struct Query_info_t {
  //! List of contours, each contour is a list of points.
  std::vector<std::vector<cv::Point> > m_contours;
  //! Distance transform image.
  cv::Mat m_distImg;
  //! Corresponding edge orientation for each point for each contour.
  std::vector<std::vector<float> > m_edgesOrientation;
  //! Query Image.
  cv::Mat m_img;
  //! Integral distance transform image.
  cv::Mat m_integralDistImg;
  //! Integral edge orientation.
  cv::Mat m_integralEdgeOrientation;
  //! Image that contains at each location the edge orientation value of the corresponding
  //! nearest edge for the current pixel location.
  cv::Mat m_mapOfEdgeOrientation;
  //! Image that contains the id corresponding to the nearest edge point.
  cv::Mat m_mapOfLabels;
  //! Image mask.
  cv::Mat m_mask;
  //! Vector of contours approximated by lines.
  std::vector<std::vector<Line_info_t> > m_vectorOfContourLines;

  Query_info_t(const std::vector<std::vector<cv::Point> > &contours, const cv::Mat &dist, const cv::Mat &img,
      const cv::Mat &integralDistImg, const cv::Mat &integralEdgeOrientation, const cv::Mat &edgeOriImg,
      const std::vector<std::vector<float> > &edgesOri, const cv::Mat &labels, const cv::Mat &mask,
      const std::vector<std::vector<Line_info_t> > &contourLines)
  : m_contours(contours), m_distImg(dist), m_edgesOrientation(edgesOri), m_img(img), m_integralDistImg(integralDistImg),
    m_integralEdgeOrientation(integralEdgeOrientation), m_mapOfEdgeOrientation(edgeOriImg), m_mapOfLabels(labels),
    m_mask(mask), m_vectorOfContourLines(contourLines) {
  }

  Query_info_t()
  : m_contours(), m_distImg(), m_edgesOrientation(), m_img(), m_integralDistImg(), m_integralEdgeOrientation(),
    m_mapOfEdgeOrientation(), m_mapOfLabels(), m_mask(), m_vectorOfContourLines() {
  }
};


class ChamferMatcher {
public:
  enum MatchingType {
    edgeMatching, edgeForwardBackwardMatching, fullMatching, maskMatching, forwardBackwardMaskMatching,
    lineMatching, lineForwardBackwardMatching, lineIntegralMatching
  };

  enum RejectionType {
    noRejection,
    //! Use a grid of reference points to quickly decide if a location should be further processed with Chamfer matching.
    gridDescriptorRejection/*, hogRejection*/
  };

  enum MatchingStrategyType {
    //! Will search the template in a specific area (all the images or in a ROI).
    templateMatching,
    //! Will search the template only at a specific location.
    templatePoseMatching
  };

  enum PyramidType {
    //! Regular Chamfer matching.
    noPyramid,
    //! Precompute the rejection mask on layer+1 (PyrDown) with the result of the computation of the rejection mask.
    //! Warning: RejectionType has to be gridDescriptorRejection
    //! Warning: Worst computation time...
    pyramid1,
    //!Precompute the rejection mask on layer+1 (PyrDown) with the result of the Chamfer matching.
    //! Not working (bad results).
    pyramid2
  };

  ChamferMatcher();

  ChamferMatcher(const std::map<int, cv::Mat> &mapOfTemplateImages,
      const std::map<int, std::pair<cv::Rect, cv::Rect> > &mapOfTemplateRois);

  static void computeCanny(const cv::Mat &img, cv::Mat &edges, const double threshold);

  static void computeDistanceTransform(const cv::Mat &img, cv::Mat &dist_img, cv::Mat &labels);

  /*
   * Compute the map that links for each contour id the corresponding indexes i,j in
   * the vector of vectors.
   */
  static void computeEdgeMapIndex(const std::vector<std::vector<cv::Point> > &contours,
      const cv::Mat &labels, std::map<int, std::pair<int, int> > &mapOfIndex);

  static void computeIntegralDistanceTransform(const cv::Mat &dt, cv::Mat &idt, const int nbClusters,
      const bool useLineIterator=true);

  static void createMapOfEdgeOrientations(const cv::Mat &img, const cv::Mat &labels, cv::Mat &mapOfEdgeOrientations,
      std::vector<std::vector<cv::Point> > &contours, std::vector<std::vector<float> > &edges_orientation);

  /*
   * Create a LUT that maps an angle in degree to the corresponding index.
   */
  static std::vector<int> createOrientationLUT(int nbClusters);

  /*
   * Create the template mask.
   */
  static void createTemplateMask(const cv::Mat &img, cv::Mat &mask, const double threshold=50.0);

  void detect(const cv::Mat &img_query, std::vector<Detection_t> &detections,
      const bool useOrientation, const float distanceThresh=50.0f, const float lambda=5.0f,
      const float weight_forward=1.0f, const float weight_backward=1.0f, const bool useGroupDetections=true);

  void detectMultiScale(const cv::Mat &img_query, std::vector<Detection_t> &detections,
      const bool useOrientation, const float distanceThresh=50.0f, const float lambda=5.0f,
      const float weight_forward=1.0f, const float weight_backward=1.0f,
      const bool useNonMaximaSuppression=true, const bool useGroupDetections=true);

  void displayTemplateData(const int tempo=0);

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

  inline double getCannyThreshold() const {
    return m_cannyThreshold;
  }

  inline cv::Size getGridDescriptorSize() const {
    return m_gridDescriptorSize;
  }

  inline MatchingStrategyType getMatchingStrategyType() const {
    return m_matchingStrategyType;
  }

  inline MatchingType getMatchingType() const {
    return m_matchingType;
  }

  inline float getMaxDescriptorDistanceError() const {
    return m_maxDescriptorDistanceError;
  }

  inline float getMaxDescriptorOrientationError() const {
    return m_maxDescriptorOrientationError;
  }

  inline int getMinNbDescriptorMatches() const {
    return m_minNbDescriptorMatches;
  }

  inline size_t getNbTemplates() const {
    return m_mapOfTemplate_info.size();
  }

  inline PyramidType getPyramidType() const {
    return m_pyramidType;
  }

  inline RejectionType getRejectionType() const {
    return m_rejectionType;
  }

  void loadTemplateData(const std::string &filename);

  void saveTemplateData(const std::string &filename, const bool saveSingleFile=true);

  inline void setCannyThreshold(const double threshold) {
    m_cannyThreshold = threshold;
  }

  inline void setGridDescritorSize(const cv::Size &size) {
    if(size.width > 0 && size.height > 0) {
      m_gridDescriptorSize = size;
    } else {
      std::cerr << "Size is too small !" << std::endl;
    }
  }

  inline void setMatchingStrategyType(const MatchingStrategyType &type) {
    m_matchingStrategyType = type;
  }

  inline void setMatchingType(const MatchingType &type) {
    m_matchingType = type;
  }

  inline void setMaxDescriptorDistanceError(const float error) {
    if(error > 0) {
      m_maxDescriptorDistanceError = error;
    } else {
      std::cerr << "The distance error cannot be negative or null !" << std::endl;
    }
  }

  /*
   * Set the maximal orientation error in radian to match two reference points.
   */
  inline void setMaxDescriptorOrientationError(const float error) {
    if(error > 0) {
      m_maxDescriptorOrientationError = error;
    } else {
      std::cerr << "The orientation error cannot be negative or null !" << std::endl;
    }
  }

  inline void setMinNbDescriptorMatches(const int nb) {
    if(nb > 0 && nb <= m_gridDescriptorSize.width*m_gridDescriptorSize.height) {
      m_minNbDescriptorMatches = nb;
    } else {
      std::cerr << "The minimal number of matches should be > 0 and <= size grid !" << std::endl;
    }
  }

  inline void setPyramidType(const PyramidType &type) {
    m_pyramidType = type;
    //Recompute the scale
    setScale(m_scaleMin, m_scaleMax, m_scaleStep);
  }

  inline void setRejectionType(const RejectionType &type) {
    m_rejectionType = type;
  }

  void setScale(const int min, const int max, const int step);

  void setTemplateImages(const std::map<int, cv::Mat> &mapOfTemplateImages,
      const std::map<int, std::pair<cv::Rect, cv::Rect> > &mapOfTemplateRois, const bool deepCopy=true);

#if DEBUG
  //DEBUG:
  bool m_debug;
#endif


private:

  void approximateContours(const std::vector<std::vector<cv::Point> > &contours,
      std::vector<std::vector<Line_info_t> > &lines, const double epsilon=3.0);

  double computeChamferDistance(const Template_info_t &template_info, const Query_info_t &query_info,
      const int offsetX, const int offsetY,
#if DEBUG
      cv::Mat &img_res,
#endif
      const bool useOrientation=false, const float lambda=5.0f,
      const float weight_forward=1.0f, const float weight_backward=1.0f);

  double computeFullChamferDistance(const Template_info_t &template_info, const Query_info_t &query_info,
      const int offsetX, const int offsetY,
#if DEBUG
      cv::Mat &img_res,
#endif
      const bool useOrientation=false, const float lambda=5.0f);

  void computeMatchingMap(const Template_info_t &template_info, const Query_info_t &query_info, cv::Mat &chamferMap,
      cv::Mat &rejection_mask, const bool useOrientation=false, const int xStep=5, const int yStep=5,
      const float lambda=5.0f, const float weight_forward=1.0f, const float weight_backward=1.0f);

  void computeRejectionMask(const Template_info_t &template_info, const Query_info_t &query_info,
      cv::Mat &rejection_mask, const int startI, const int endI, const int yStep, const int startJ,
      const int endJ, const int xStep);

  void detect_impl(const Template_info_t &template_info, const Query_info_t &query_info, const int scale,
      std::vector<Detection_t> &currentDetections, cv::Mat &rejection_mask, const bool useOrientation,
      const float distanceThresh, const float lambda=5.0f, const float weight_forward=1.0f,
      const float weight_backward=1.0f, const bool useGroupDetections=true);

  void groupDetections(const std::vector<Detection_t> &detections, std::vector<Detection_t> &groupedDetections,
      const double overlapPercentage=0.5);

  void retainDetections(std::vector<Detection_t> &bbDetections, const float threshold);

  void nonMaximaSuppression(const std::vector<Detection_t> &detections, std::vector<Detection_t> &maximaDetections);

  Query_info_t prepareQuery(const cv::Mat &img_query);
  Template_info_t prepareTemplate(const cv::Mat &img_template);


  //! Threshold for Canny edge detection.
  double m_cannyThreshold;
  //! Maximal distance transform error distance to match two reference points.
  float m_maxDescriptorDistanceError;
  //! Maximal orientation error to match two reference points.
  float m_maxDescriptorOrientationError;
  //! Minimal number of matched descriptors to decide if the current location has a probability to contain the template.
  int m_minNbDescriptorMatches;
  //! Grid descriptor size.
  cv::Size m_gridDescriptorSize;
  //! Matching strategy type (search in all the image or at a specific location).
  MatchingStrategyType m_matchingStrategyType;
  //! Matching type.
  MatchingType m_matchingType;
  //! Structure that contains all the information about the query images.
  //  Query_info_t m_query_info;

  //XXX: just for saving video
public:
  //! Map that contains all the information about the template images and at different scales.
  std::map<int, std::map<int, Template_info_t> > m_mapOfTemplate_info;

  //! Map that contains as a key the template id and as a value the template image.
  std::map<int, cv::Mat> m_mapOfTemplateImages;
private:

  //! LUT that maps an angle in degree to a cluster index.
  std::vector<int> m_orientationLUT;
  //! Pyramid type to use.
  PyramidType m_pyramidType;
  //! Rejection type to quickly decide if the current location should be further match with Chamfer method.
  RejectionType m_rejectionType;
  //! Max scale as percentage (200 for example).
  int m_scaleMax;
  //! Min scale as percentage (50 for example).
  int m_scaleMin;
  //! Scale step as percentage (10 for example).
  int m_scaleStep;
  //! Vector of scales to use for the detectMultiScale.
  std::vector<int> m_scaleVector;
};

#endif
