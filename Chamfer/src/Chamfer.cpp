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
#include "../include/Chamfer.hpp"
#include "../include/Utils.hpp"
#include <limits>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>


ChamferMatcher::ChamferMatcher()
    : m_cannyThreshold(50.0), m_matchingType(edgeForwardBackwardMatching),
      m_query_info(), m_mapOfTemplate_info() {
}

ChamferMatcher::ChamferMatcher(const std::map<int, std::pair<cv::Rect, cv::Mat> > &mapOfTemplateImages)
    : m_cannyThreshold(50.0), m_matchingType(edgeForwardBackwardMatching),
      m_query_info(), m_mapOfTemplate_info() {

  for(std::map<int, std::pair<cv::Rect, cv::Mat> >::const_iterator it = mapOfTemplateImages.begin();
      it != mapOfTemplateImages.end(); ++it) {
    //Precompute the template information for scale=1.0
    m_mapOfTemplate_info[it->first][1.0] = prepareTemplate(it->second.second);

    //Set query ROI
    m_mapOfTemplate_info[it->first][1.0].m_queryROI = it->second.first;
  }
}

void ChamferMatcher::approximateContours(const std::vector<std::vector<cv::Point> > &contours,
    std::vector<std::vector<Line_info_t> > contour_lines, const double epsilon) {

  for(size_t i = 0; i < contours.size(); i++) {
    //Approximate the current contour
    std::vector<cv::Point> approx_contour;
    cv::approxPolyDP(contours[i], approx_contour, epsilon, true);

    std::vector<Line_info_t> lines;
    //Compute polar line equation for the approximated contour
    for(size_t j = 0; j < approx_contour.size()-1; j++) {
      double length, rho, theta;
      getPolarLineEquation(approx_contour[j], approx_contour[j+1], theta, rho, length);

      //Add the line
      lines.push_back(Line_info_t(length, rho, theta, approx_contour[j], approx_contour[j+1]));
    }

    //Add the lines
    contour_lines.push_back(lines);
  }
}

/*
 * Detect edges using the Canny method and create and image with edges displayed in black for cv::distanceThreshold
 */
void ChamferMatcher::computeCanny(const cv::Mat &img, cv::Mat &edges, const double threshold) {
  cv::Mat canny_img;
  cv::Canny(img, canny_img, threshold, 3.0*threshold);

  //cv::THRESH_BINARY_INV is used to invert the image as distance transform compute the
  //minimal distance between each pixel to the nearest zero pixel
  cv::threshold(canny_img, edges, 127, 255, cv::THRESH_BINARY_INV);
}

/*
 * Compute the Chamfer distance for each point in the template contour to the nearest edge
 * in the query image.
 */
double ChamferMatcher::computeChamferDistance(const Template_info_t &template_info, const int offsetX, const int offsetY,
    cv::Mat &img_res,
    const bool useOrientation, const float lambda, const float weight_forward, const float weight_backward) {
  double chamfer_dist = 0.0;
  int nbElements = 0;

  img_res = cv::Mat::zeros(template_info.m_distImg.size(), CV_32F);

  if(m_matchingType == lineMatching || m_matchingType == lineForwardBackwardMatching) {
    //"Forward matching" <==> matches lines from template to the nearest lines in the query
    for(size_t i = 0; i < template_info.m_vectorOfContourLines.size(); i++) {
      for(size_t j = 0; j < template_info.m_vectorOfContourLines[i].size(); j++) {
        cv::Point pt1 = template_info.m_vectorOfContourLines[i][j].m_pointStart;
        cv::Point pt2 = template_info.m_vectorOfContourLines[i][j].m_PointEnd;

        cv::LineIterator it_line(m_query_info.m_distImg, pt1, pt2, 8);
        for(int cpt = 0; cpt < it_line.count; cpt++, ++it_line, nbElements++) {

          if(useOrientation) {
            chamfer_dist += weight_forward * ( m_query_info.m_distImg.at<float>(it_line.pos()) + lambda *
                getMinAngleError(template_info.m_mapOfEdgeOrientation.at<float>(it_line.pos()),
                    m_query_info.m_mapOfEdgeOrientation.at<float>(it_line.pos()), false, true) );
          } else {
            chamfer_dist += weight_forward * ( m_query_info.m_distImg.at<float>(it_line.pos()) );
          }
        }
      }
    }

    if(m_matchingType == lineForwardBackwardMatching) {
      //"Backward matching" <==> matches edges from query to the nearest edges in the template
      for(size_t i = 0; i < m_query_info.m_vectorOfContourLines.size(); i++) {
        for(size_t j = 0; j < m_query_info.m_vectorOfContourLines[i].size(); j++) {
          cv::Point pt1 = m_query_info.m_vectorOfContourLines[i][j].m_pointStart;
          cv::Point pt2 = m_query_info.m_vectorOfContourLines[i][j].m_PointEnd;

          cv::LineIterator it_line(template_info.m_distImg, pt1, pt2, 8);
          for(int cpt = 0; cpt < it_line.count; cpt++, ++it_line, nbElements++) {

            if(useOrientation) {
              chamfer_dist += weight_backward * ( template_info.m_distImg.at<float>(it_line.pos()) + lambda *
                  getMinAngleError(m_query_info.m_mapOfEdgeOrientation.at<float>(it_line.pos()),
                      template_info.m_mapOfEdgeOrientation.at<float>(it_line.pos()), false, true) );
            } else {
              chamfer_dist += weight_backward * ( template_info.m_distImg.at<float>(it_line.pos()) );
            }
          }
        }
      }
    }

  } else {
    //"Forward matching" <==> matches edges from template to the nearest edges in the query
    for(size_t i = 0; i < template_info.m_contours.size(); i++) {
      for(size_t j = 0; j < template_info.m_contours[i].size(); j++, nbElements++) {
        int x = template_info.m_contours[i][j].x;
        int y = template_info.m_contours[i][j].y;

        const float *ptr_row_edge_ori = m_query_info.m_mapOfEdgeOrientation.ptr<float>(y + offsetY);

        if(useOrientation) {
          chamfer_dist += weight_forward *( m_query_info.m_distImg.ptr<float>(y + offsetY)[x + offsetX]
            + lambda*(getMinAngleError(template_info.m_edgesOrientation[i][j], ptr_row_edge_ori[x + offsetX], false, true)) );

          //DEBUG:
          float *ptr_row_res = img_res.ptr<float>(y);
          ptr_row_res[x] = m_query_info.m_distImg.ptr<float>(y + offsetY)[x + offsetX];
        } else {
          chamfer_dist += weight_forward * m_query_info.m_distImg.ptr<float>(y + offsetY)[x + offsetX];

          //DEBUG:
          float *ptr_row_res = img_res.ptr<float>(y);
          ptr_row_res[x] = m_query_info.m_distImg.ptr<float>(y + offsetY)[x + offsetX];
        }
      }
    }

    if(m_matchingType == edgeForwardBackwardMatching) {
      //"Backward matching" <==> matches edges from query to the nearest edges in the template
      for(size_t i = 0; i < m_query_info.m_contours.size(); i++) {

        for(size_t j = 0; j < m_query_info.m_contours[i].size(); j++, nbElements) {
          int x = m_query_info.m_contours[i][j].x;
          int y = m_query_info.m_contours[i][j].y;
          const float *ptr_row_edge_ori = template_info.m_mapOfEdgeOrientation.ptr<float>(y-offsetY);

          //Get contours located in the current region
          if(offsetX <= x && x < offsetX+template_info.m_distImg.cols &&
              offsetY <= y && y < offsetY+template_info.m_distImg.rows) {

            if(useOrientation) {
              chamfer_dist += weight_backward * ( template_info.m_distImg.ptr<float>(y-offsetY)[x-offsetX] +
                  lambda*(getMinAngleError(m_query_info.m_edgesOrientation[i][j], ptr_row_edge_ori[x-offsetX], false, true)) );

              //DEBUG:
              float *ptr_row_res = img_res.ptr<float>(y-offsetY);
              ptr_row_res[x-offsetX] = template_info.m_distImg.ptr<float>(y-offsetY)[x-offsetX];
            } else {
              chamfer_dist += template_info.m_distImg.ptr<float>(y)[x];

              //DEBUG:
              float *ptr_row_res = img_res.ptr<float>(y-offsetY);
              ptr_row_res[x-offsetX] = template_info.m_distImg.ptr<float>(y-offsetY)[x-offsetX];
            }
          }
        }
      }
    }
  }

  return chamfer_dist / nbElements;
}

/*
 * Compute distance threshold. Return also an image where each pixel coordinate corresponds to the
 * id of the nearest edge. To get the coordinate of the nearest edge: find the coordinate with the corresponding
 * id and with a distance transform of 0.
 */
void ChamferMatcher::computeDistanceTransform(const cv::Mat &img, cv::Mat &dist_img, cv::Mat &labels) {
  dist_img = cv::Mat(img.size(), CV_32F);

  cv::distanceTransform(img, dist_img, labels, cv::DIST_L2, cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);
}

/*
 * Compute the map that links for each contour id the corresponding indexes i,j in
 * the vector of vectors.
 */
void ChamferMatcher::computeEdgeMapIndex(const std::vector<std::vector<cv::Point> > &contours,
    const cv::Mat &labels, std::map<int, std::pair<int, int> > &mapOfIndex) {

  for(size_t i = 0; i < contours.size(); i++) {
    for(size_t j = 0; j < contours[i].size(); j++) {
      mapOfIndex[labels.ptr<int>(contours[i][j].y)[contours[i][j].x]] = std::pair<int, int>(i,j);
    }
  }
}

/*
 * Compute the "full Chamfer distance" for the given ROI (use all the pixels instead of only edge pixels).
 */
double ChamferMatcher::computeFullChamferDistance(const Template_info_t &template_info, const int offsetX, const int offsetY,
    cv::Mat &img_res,
    const bool useOrientation, const float lambda) {
  double chamfer_dist = 0.0;
  int nbElements = 0;

  img_res = cv::Mat::zeros(template_info.m_distImg.size(), CV_32F);

  if(useOrientation) {
    cv::Mat subDistImg = m_query_info.m_distImg(
        cv::Rect(offsetX, offsetY, template_info.m_distImg.cols, template_info.m_distImg.rows));

    //Get common mask
    cv::Mat common_mask;
    cv::Mat query_mask = m_query_info.m_mask(
        cv::Rect(offsetX, offsetY, template_info.m_distImg.cols, template_info.m_distImg.rows));
    cv::bitwise_or(template_info.m_mask, query_mask, common_mask);


    //Distance Transform
    //Compute the difference only on pixels inside the template mask
    cv::Mat subDistImg_masked;
    subDistImg.copyTo(subDistImg_masked, common_mask);
    cv::Mat templateDistImg_masked;
    template_info.m_distImg.copyTo(templateDistImg_masked, common_mask);

    //DEBUG:
    bool debug = false;
    if(debug) {
      cv::Mat subDistImg_masked_display;
      double minVal, maxVal;
      cv::minMaxLoc(subDistImg_masked, &minVal, &maxVal);
      subDistImg_masked.convertTo(subDistImg_masked_display, CV_8U, 255.0/(maxVal-minVal), -255.0*minVal/(maxVal-minVal));
      cv::imshow("subDistImg_masked_display", subDistImg_masked_display);
    }

//    cv::Mat diffDistTrans = subDistImg - m_template_info.m_distImg;
    cv::Mat diffDistTrans = subDistImg_masked - templateDistImg_masked;
    diffDistTrans = diffDistTrans.mul(diffDistTrans);
    cv::Scalar sqr_sum = cv::sum(diffDistTrans);
    chamfer_dist += sqr_sum.val[0];

    cv::Mat subEdgeOriImg = m_query_info.m_mapOfEdgeOrientation(
        cv::Rect(offsetX, offsetY, template_info.m_distImg.cols, template_info.m_distImg.rows));

    //Orientation
    //Compute the difference only on pixels inside the template mask
    cv::Mat subEdgeOriImg_masked;
    subEdgeOriImg.copyTo(subEdgeOriImg_masked, common_mask);
    cv::Mat templateEdgeOrientation_masked;
    template_info.m_mapOfEdgeOrientation.copyTo(templateEdgeOrientation_masked, common_mask);

//    cv::Mat diffEdgeOri = subEdgeOriImg - m_template_info.m_mapOfEdgeOrientation;
    cv::Mat diffEdgeOri = subEdgeOriImg_masked - templateEdgeOrientation_masked;
    diffEdgeOri = diffEdgeOri.mul(diffEdgeOri);
    sqr_sum = cv::sum(diffEdgeOri);
    chamfer_dist += sqr_sum.val[0] * lambda;

    int length = subDistImg.rows*subDistImg.cols;
    nbElements += 2*length;

    //DEBUG:
    img_res += diffDistTrans + diffEdgeOri;
  } else {
    cv::Mat subDistImg = m_query_info.m_distImg(
        cv::Rect(offsetX, offsetY, template_info.m_distImg.cols, template_info.m_distImg.rows));

    //Get common mask
    cv::Mat common_mask;
    cv::Mat query_mask = m_query_info.m_mask(
        cv::Rect(offsetX, offsetY, template_info.m_distImg.cols, template_info.m_distImg.rows));
    cv::bitwise_or(template_info.m_mask, query_mask, common_mask);

    //Distance Transform
    //Compute the difference only on pixels inside the template mask
    cv::Mat subDistImg_masked;
    subDistImg.copyTo(subDistImg_masked, common_mask);
    cv::Mat templateDistImg_masked;
    template_info.m_distImg.copyTo(templateDistImg_masked, common_mask);

    cv::Mat diffDistTrans = subDistImg_masked - templateDistImg_masked;
    diffDistTrans = diffDistTrans.mul(diffDistTrans);
    cv::Scalar sqr_sum = cv::sum(diffDistTrans);
    chamfer_dist += sqr_sum.val[0];

    int length = subDistImg.rows*subDistImg.cols;
    nbElements += length;

    //DEBUG:
    img_res += diffDistTrans;
  }

  return chamfer_dist / nbElements;
}

/*
 * Compute the image that contains at each pixel location the Chamfer distance.
 */
void ChamferMatcher::computeMatchingMap(const Template_info_t &template_info, cv::Mat &chamferMap, const bool useOrientation,
    const int xStep, const int yStep, const float lambda, const float weight_forward, const float weight_backward) {
  int chamferMapWidth = m_query_info.m_distImg.cols - template_info.m_distImg.cols + 1;
  int chamferMapHeight = m_query_info.m_distImg.rows - template_info.m_distImg.rows + 1;

  //Set the map at the maximum float value
  chamferMap = std::numeric_limits<float>::max()*
      cv::Mat::ones(chamferMapHeight, chamferMapWidth, CV_32F);

  //DEBUG:
  bool display = false;

  int startI = template_info.m_queryROI.y;
  int endI = template_info.m_queryROI.height > 0 ? startI+template_info.m_queryROI.height : chamferMapHeight;
  int startJ = template_info.m_queryROI.x;
  int endJ = template_info.m_queryROI.width > 0 ? startJ+template_info.m_queryROI.width : chamferMapWidth;

#pragma omp parallel for
  for(int i = startI; i < endI; i += yStep) {
    float *ptr_row = chamferMap.ptr<float>(i);
    for(int j = startJ; j < endJ; j += xStep) {
      cv::Mat res;

      switch(m_matchingType) {
      case edgeMatching:
      case edgeForwardBackwardMatching:
        ptr_row[j] = computeChamferDistance(template_info, j, i, res, useOrientation, lambda, weight_forward, weight_backward);
//        std::cout << "cost=" << ptr_row[j] << std::endl;
        break;

      case fullMatching:
      default:
        ptr_row[j] = computeFullChamferDistance(template_info, j, i, res, useOrientation, lambda);
        break;
      }

      //DEBUG:
      if(display) {
//        std::cout << "ptr_row[" << j << "]=" << ptr_row[j] << std::endl;

        cv::Mat query_img_roi = m_query_info.m_img(cv::Rect(j, i, template_info.m_distImg.cols,
            template_info.m_distImg.rows));
        cv::Mat displayEdgeAndChamferDist;
        double threshold = 50;
        cv::Canny(query_img_roi, displayEdgeAndChamferDist, threshold, 3.0*threshold);

        cv::Mat res_8u;
        double min, max;
        cv::minMaxLoc(res, &min, &max);
        res.convertTo(res_8u, CV_8U, 255.0/(max-min), -255.0*min/(max-min));

        displayEdgeAndChamferDist = displayEdgeAndChamferDist + res_8u;

        cv::imshow("displayEdgeAndChamferDist", displayEdgeAndChamferDist);
        cv::imshow("res_8u", res_8u);

        char c = cv::waitKey(0);
        if(c == 27) {
          display = false;
        }
      }
    }
  }
}

/*
 * Create an image that contains at each pixel location the edge orientation corresponding to the nearest edge.
 */
void ChamferMatcher::createMapOfEdgeOrientations(const cv::Mat &img, const cv::Mat &labels, cv::Mat &mapOfEdgeOrientations,
    std::vector<std::vector<cv::Point> > &contours, std::vector<std::vector<float> > &edges_orientation) {
  //Find contours
  getContours(img, contours);

  //Compute orientation for each contour point
  getContoursOrientation(contours, edges_orientation);

  std::map<int, std::pair<int, int> > mapOfIndex;
  computeEdgeMapIndex(contours, labels, mapOfIndex);

  mapOfEdgeOrientations = cv::Mat::zeros(img.size(), CV_32F);
  for(int i = 0; i < img.rows; i++) {
    const int *ptr_row_label = labels.ptr<int>(i);
    float *ptr_row_edgeOri = mapOfEdgeOrientations.ptr<float>(i);

    for(int j = 0; j < img.cols; j++) {
      size_t idx1 = mapOfIndex[ptr_row_label[j]].first;
      size_t idx2 = mapOfIndex[ptr_row_label[j]].second;

      //TODO: add check if there are contours
      ptr_row_edgeOri[j] = edges_orientation[idx1][idx2];
    }
  }
}

/*
 * Create the template mask.
 */
void ChamferMatcher::createTemplateMask(const cv::Mat &img, cv::Mat &mask, const double threshold) {
  std::vector<std::vector<cv::Point> > contours;
  getContours(img, contours, threshold);

  mask = cv::Mat::zeros(img.size(), CV_8U);
  for(int i = 0; i < contours.size(); i++) {
    cv::drawContours(mask, contours, i, cv::Scalar(255), -1);
  }
}

/*
 * Detect an image template in a query image.
 */
void ChamferMatcher::detect_impl(const Template_info_t &template_info, const double scale,
    std::vector<Detection_t> &bbDetections, const bool useOrientation, const float distanceThresh,
    const float lambda, const float weight_forward, const float weight_backward) {
  cv::Mat chamferMap;
  computeMatchingMap(template_info, chamferMap, useOrientation, lambda, weight_forward, weight_backward);

  double minVal, maxVal;
  //Avoid possibility of infinite loop and / or keep a maximum of 100 detections
  int maxLoopIterations = 100, iteration = 0;

  do {
    iteration++;

    //Find the pixel location of the minimal Chamfer distance.
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(chamferMap, &minVal, &maxVal, &minLoc, &maxLoc);

    //"Reset the location" to find other detections
    chamferMap.at<float>(minLoc.y, minLoc.x) = std::numeric_limits<float>::max();

    cv::Point pt1(minLoc.x, minLoc.y);
    cv::Point pt2 = pt1 + cv::Point(template_info.m_distImg.cols, template_info.m_distImg.rows);

    //Add the detection
    cv::Rect detection(pt1, pt2);
    Detection_t detect_t(detection, minVal, scale);
    bbDetections.push_back(detect_t);
  } while( minVal < distanceThresh && iteration <= maxLoopIterations );
}

/*
 * Detect on a single scale.
 */
void ChamferMatcher::detect(const cv::Mat &img_query, std::vector<Detection_t> &detections,
    const bool useOrientation, const float distanceThresh, const float lambda,
    const float weight_forward, const float weight_backward) {
  detections.clear();

  prepareQuery(img_query);

  double scale = 1.0;
  for(std::map<int, std::map<double, Template_info_t> >::const_iterator it = m_mapOfTemplate_info.begin();
      it != m_mapOfTemplate_info.end(); ++it) {
//    std::cout << "id=" << it->first << std::endl;
    std::vector<Detection_t> all_detections;

    std::map<double, Template_info_t>::const_iterator it_template = it->second.find(1.0);
    if(it_template != it->second.end()) {
      detect_impl(it_template->second, scale, all_detections, useOrientation, distanceThresh);

      //Non maxima suppression
      std::vector<Detection_t> all_maxima_detections;
      nonMaximaSuppression(all_detections, all_maxima_detections);

      //Useless, should be already done in detect_impl()
      //Keep only detection with a Chamfer distance below a threshold
      retainDetections(all_maxima_detections, distanceThresh);

      //Group similar detections
      std::vector<Detection_t> current_detections;
      groupDetections(all_maxima_detections, current_detections);

      //Set Template index
      for(std::vector<Detection_t>::iterator it_detection = current_detections.begin();
          it_detection != current_detections.end(); ++it_detection) {
        it_detection->m_templateIndex = it->first;
      }

      detections.insert(detections.end(), current_detections.begin(), current_detections.end());
    }
  }
//  std::cout << std::endl;

  //Sort detections by increasing cost
  std::sort(detections.begin(), detections.end());
}

/*
 * Detect on multiple scales.
 */
void ChamferMatcher::detectMultiScale(const cv::Mat &img_query, std::vector<Detection_t> &detections,
    const bool useOrientation, const float distanceThresh, const double scaleFactor,
    const double minScale, const double maxScale, const float lambda,
    const float weight_forward, const float weight_backward) {
  detections.clear();

  for(std::map<int, std::map<double, Template_info_t> >::iterator it = m_mapOfTemplate_info.begin();
      it != m_mapOfTemplate_info.end(); ++it) {

    //Compute template information for all the scales except scale==1.0 which is already computed in the constructor
    for(double scale = minScale; scale <= maxScale; scale += scaleFactor) {
      cv::Mat img_template_scale;
      cv::resize(m_mapOfTemplate_info[it->first][1.0].m_img, img_template_scale, cv::Size(), scale, scale);

      m_mapOfTemplate_info[it->first][scale] = prepareTemplate(img_template_scale);
    }
  }

  prepareQuery(img_query);

  for(std::map<int, std::map<double, Template_info_t> >::iterator it = m_mapOfTemplate_info.begin();
      it != m_mapOfTemplate_info.end(); ++it) {
    std::vector<Detection_t> all_detections;

    for(double scale = minScale; scale <= maxScale; scale += scaleFactor) {
      std::vector<Detection_t> current_detections;
      detect_impl(it->second[scale], scale, current_detections, useOrientation, distanceThresh);

      all_detections.insert(all_detections.end(), current_detections.begin(), current_detections.end());
    }

    //Non maxima suppression
    std::vector<Detection_t> all_maxima_detections;
    nonMaximaSuppression(all_detections, all_maxima_detections);

    //Useless, should be already done in detect_impl()
    //Keep only detection with a Chamfer distance below a threshold
  //  retainDetections(all_maxima_detections, distanceThresh);

  //  bbDetections = all_detections;
    //Group similar detections
    groupDetections(all_maxima_detections, detections);

    //Sort detections by increasing cost
    std::sort(detections.begin(), detections.end());
  }
}

/*
 * Filter contours that contains less than a specific number of points.
 */
void ChamferMatcher::filterSingleContourPoint(std::vector<std::vector<cv::Point> > &contours, const size_t min) {
  std::vector<std::vector<cv::Point> > contours_filtered;

  for(std::vector<std::vector<cv::Point> >::const_iterator it_contour = contours.begin();
      it_contour != contours.end(); ++it_contour) {

    if(it_contour->size() >= min) {
      contours_filtered.push_back(*it_contour);
    }
  }

  contours = contours_filtered;
}

/*
 * Get the list of contour points.
 */
void ChamferMatcher::getContours(const cv::Mat &img, std::vector<std::vector<cv::Point> > &contours, const double threshold) {
  cv::Mat canny_img;
  cv::Canny(img, canny_img, threshold, 3.0*threshold);

//  std::vector<std::vector<cv::Point> > raw_contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(canny_img, contours, hierarchy, /*CV_RETR_TREE*/CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  //TODO: Keep only contours that are not a hole

  filterSingleContourPoint(contours);
}

/*
 * Compute for each contour point the corresponding edge orientation.
 * For the current contour point, use the previous and next point to
 * compute the edge orientation.
 */
void ChamferMatcher::getContoursOrientation(const std::vector<std::vector<cv::Point> > &contours,
    std::vector<std::vector<float> > &contoursOrientation) {

  for(std::vector<std::vector<cv::Point> >::const_iterator it_contour = contours.begin();
      it_contour != contours.end(); ++it_contour) {
    std::vector<float> orientations;


    //DEBUG:
    bool debug = false;
    int cpt = 0;


    if(it_contour->size() > 2) {
      for(std::vector<cv::Point>::const_iterator it_point = it_contour->begin()+1;
          it_point != it_contour->end(); ++it_point, cpt++) {
        if(debug) {
          //DEBUG:
          if(cpt >= 10 && cpt < 20) {
            cv::Point pt1 = *(it_contour->end()-2);
            cv::Point pt2 = *(it_contour->begin());

            std::cout << "pt1=" << pt1 << " ; pt2=" << pt2 << std::endl;
            double a, b;
            if(getLineEquation(pt1, pt2, a, b)) {

              double theta, rho;
              getPolarLineEquation(a, b, theta, rho);
              std::cout << "a=" << a << " ; b=" << b << " ; theta=" << (theta*180.0/M_PI) << std::endl;
            } else {
              std::cout << "Line !" << std::endl;
            }
            std::cout << std::endl;
          }
        }


#if 1
#if 1
        double rho = 0.0 , angle = 0.0;

        if(it_point == it_contour->begin()+1) {
          getPolarLineEquation(*(it_point-1), *(it_point+1), angle, rho);

          //First point orientation == second point orientation
          orientations.push_back(angle);
          orientations.push_back(angle);
        } else if(it_point == it_contour->end()-1) {
          //Last point
          getPolarLineEquation(*(it_contour->end()-2), *(it_contour->begin()), angle, rho);

          orientations.push_back( *(orientations.end()-1) );
        } else {
          getPolarLineEquation(*(it_point-1), *(it_point+1), angle, rho);

          orientations.push_back((float) angle);
        }
#else
        if(it_point == it_contour->begin()) {
          //First point
          float angle = getAngle(*(it_contour->end()-1), *(it_point+1));

          orientations.push_back(angle);
        } else if(it_point == it_contour->end()-1) {
          //Last point
          float angle = getAngle(*(it_point-1), *(it_contour->begin()));

          orientations.push_back(angle);
        } else {
          float angle = getAngle(*(it_point-1), *(it_point+1));

          orientations.push_back(angle);
        }
#endif
#else
        if(it_point == it_contour->begin()+1) {
          float angle = getAngle(*(it_point-1), *(it_point+1));

          //First point orientation == second point orientation
          orientations.push_back(angle);
          orientations.push_back(angle);
        } else if(it_point == it_contour->end()-1) {
          //Last point
          float angle = getAngle(*(it_contour->end()-2), *(it_contour->begin()));

          orientations.push_back( *(orientations.end()-1) );
        } else {
          float angle = getAngle(*(it_point-1), *(it_point+1));

          orientations.push_back(angle);
        }
#endif
      }
    } else {
      for(std::vector<cv::Point>::const_iterator it_point = it_contour->begin();
          it_point != it_contour->end(); ++it_point) {
        std::cerr << "Not enough contour points !" << std::endl;
        orientations.push_back(0);
      }
    }

    contoursOrientation.push_back(orientations);
  }
}

/*
 * Group similar detections (detections whose the overlapping percentage is above a specific threshold).
 */
void ChamferMatcher::groupDetections(const std::vector<Detection_t> &detections,
    std::vector<Detection_t> &groupedDetections) {
  std::vector<std::vector<Detection_t> > clustered_detections;

  std::vector<bool> already_picked(detections.size(), false);
  for(size_t cpt1 = 0; cpt1 < detections.size(); cpt1++) {
    std::vector<Detection_t> current_detections;

    if(!already_picked[cpt1]) {
      current_detections.push_back(detections[cpt1]);
      already_picked[cpt1] = true;

      for(size_t cpt2 = cpt1+1; cpt2 < detections.size(); cpt2++) {

        if(!already_picked[cpt2]) {
          cv::Rect r_intersect = detections[cpt1].m_boundingBox & detections[cpt2].m_boundingBox;
          double overlapping_percentage = r_intersect.area() /
              (double) (detections[cpt1].m_boundingBox.area() + detections[cpt2].m_boundingBox.area() - r_intersect.area());

          if(overlapping_percentage > 0.65) {
            already_picked[cpt2] = true;
            current_detections.push_back(detections[cpt2]);
          }
        }
      }

      clustered_detections.push_back(current_detections);
    }
  }

  for(std::vector<std::vector<Detection_t> >::const_iterator it1 = clustered_detections.begin();
      it1 != clustered_detections.end(); ++it1) {
    double xMean = 0.0, yMean = 0.0, distMean = 0.0, scaleMean = 0.0;

    for(std::vector<Detection_t>::const_iterator it2 = it1->begin(); it2 != it1->end(); ++it2) {
      xMean += it2->m_boundingBox.x;
      yMean += it2->m_boundingBox.y;
      distMean += it2->m_chamferDist;
      scaleMean += it2->m_scale;
    }

    xMean /= it1->size();
    yMean /= it1->size();
    distMean /= it1->size();
    scaleMean /= it1->size();

    Detection_t detection(cv::Rect(xMean, yMean, it1->begin()->m_boundingBox.width,
        it1->begin()->m_boundingBox.height), distMean, scaleMean);
    groupedDetections.push_back(detection);
  }
}

/*
 * Load template data.
 * Call prepareTemplate for each read template.
 */
void ChamferMatcher::loadTemplateData(const std::string &filename) {
  std::ifstream file(filename.c_str(), std::ifstream::binary);

  if(file.is_open()) {
    //Clean the map
    m_mapOfTemplate_info.clear();

    //Read the number of templates
    int nbTemplates = 0;
    file.read((char *)(&nbTemplates), sizeof(nbTemplates));

    for(int cpt = 0; cpt < nbTemplates; cpt++) {
      //Read the id of the template
      int id = 0;
      file.read((char *)(&id), sizeof(id));


      //Read template image
      //Read the number of rows
      int nbRows = 0;
      file.read((char *)(&nbRows), sizeof(nbRows));

      //Read the number of cols
      int nbCols = 0;
      file.read((char *)(&nbCols), sizeof(nbCols));

      //Read the number of channel
      int nbChannels = 0;
      file.read((char *)(&nbChannels), sizeof(nbChannels));

      //Read image data
      //Allocate array
      char *data = new char[nbRows*nbCols*nbChannels];
      file.read((char *)(data), sizeof(char)*nbRows*nbCols*nbChannels);

      //Copy data to mat
      cv::Mat img;
      if(nbChannels == 3) {
        img = cv::Mat(nbRows, nbCols, CV_8UC3, data);
      } else {
        img = cv::Mat(nbRows, nbCols, CV_8U, data);
      }


      //Read the query ROI
      int x = 0;
      file.read((char *)(&x), sizeof(x));

      int y = 0;
      file.read((char *)(&y), sizeof(y));

      int width = 0;
      file.read((char *)(&width), sizeof(width));

      int height = 0;
      file.read((char *)(&height), sizeof(height));

      //Create query ROI
      cv::Rect queryROI(x, y, width, height);


      //Create Template
      Template_info_t template_info = prepareTemplate(img);
      template_info.m_queryROI = queryROI;

      m_mapOfTemplate_info[id][1.0] = template_info;
    }
  } else {
    std::cerr << "File: " << filename << " cannot be opened !" << std::endl;
  }
}

/*
 * Remove detections inside another detections.
 */
void ChamferMatcher::nonMaximaSuppression(const std::vector<Detection_t> &detections,
    std::vector<Detection_t> &maximaDetections) {
  std::vector<Detection_t> detections_copy = detections;

  //Sort by area
  std::sort(detections_copy.begin(), detections_copy.end(), less_than_area());

  //Discard detections inside another detections
  for(size_t cpt1 = 0; cpt1 < detections_copy.size(); cpt1++) {
    cv::Rect r1 = detections_copy[cpt1].m_boundingBox;
    bool is_inside = false;

    for(size_t cpt2 = cpt1+1; cpt2 < detections_copy.size() &&!is_inside; cpt2++) {
      cv::Rect r2 = detections_copy[cpt2].m_boundingBox;

      if(r1.x+r1.width < r2.x+r2.width && r1.x > r2.x && r1.y+r1.height < r2.y+r2.height && r1.y > r2.y) {
        is_inside = true;
      }
    }

    if(!is_inside) {
      maximaDetections.push_back(detections_copy[cpt1]);
    }
  }
}

/*
 * Compute all the necessary information for the query part.
 */
void ChamferMatcher::prepareQuery(const cv::Mat &img_query) {
  cv::Mat edge_query;
  computeCanny(img_query, edge_query, m_cannyThreshold);

  cv::imshow("edge_query", edge_query);
  //XXX:
//  cv::imwrite("Edge_query.png", edge_query);

  cv::Mat dist_query, img_dist_query, labels_query;
  computeDistanceTransform(edge_query, dist_query, labels_query);

  dist_query.convertTo(img_dist_query, CV_8U);
  cv::imshow("img_dist_query", img_dist_query);

  cv::Mat edge_orientations_query;
  std::vector<std::vector<cv::Point> > contours;
  std::vector<std::vector<float> > edges_orientation;
  createMapOfEdgeOrientations(img_query, labels_query, edge_orientations_query, contours, edges_orientation);

  //Query mask
  cv::Mat mask;
  createTemplateMask(img_query, mask);

  //Contours Lines
  std::vector<std::vector<Line_info_t> > contours_lines;
  approximateContours(contours, contours_lines);

  m_query_info = Query_info_t(contours, dist_query, img_query, edge_orientations_query, edges_orientation,
      labels_query, mask, contours_lines);
}

/*
 * Compute all the necessary information for the template part.
 */
Template_info_t ChamferMatcher::prepareTemplate(const cv::Mat &img_template) {
  cv::Mat edge_template;
  computeCanny(img_template, edge_template, m_cannyThreshold);

  cv::imshow("edge_template", edge_template);
  //XXX:
//  cv::imwrite("Edge_template.png", edge_template);

  cv::Mat dist_template, img_dist_template, labels_template;
  computeDistanceTransform(edge_template, dist_template, labels_template);

  dist_template.convertTo(img_dist_template, CV_8U);
  cv::imshow("img_dist_template", img_dist_template);

  cv::Mat edge_orientations_template;
  std::vector<std::vector<cv::Point> > contours_template;
  std::vector<std::vector<float> > edges_orientation;
  createMapOfEdgeOrientations(img_template, labels_template, edge_orientations_template, contours_template, edges_orientation);


  //DEBUG:
  bool debug = false;
  if(debug) {
    cv::Mat displayFindContours = cv::Mat::zeros(img_template.size(), CV_32F);
    for(int i = 0; i < contours_template.size(); i++) {
      for(int j = 0; j < contours_template[i].size(); j++) {
        displayFindContours.at<float>(contours_template[i][j].y, contours_template[i][j].x) = j;
      }
    }
    std::cout << "\ndisplayFindContours=\n" << displayFindContours << std::endl << std::endl;

    cv::Mat displayContourOrientation = cv::Mat::zeros(img_template.size(), CV_32F);
    for(int i = 0; i < edges_orientation.size(); i++) {
      for(int j = 0; j < edges_orientation[i].size(); j++) {
        float angle = (edges_orientation[i][j] + M_PI_2) * 180.0 / M_PI;
        displayContourOrientation.at<float>(contours_template[i][j].y, contours_template[i][j].x) = angle;
      }
    }
    std::cout << "\ndisplayContourOrientation=\n" << displayContourOrientation << std::endl << std::endl;


    //DEBUG:
    //Display edge orientations
    cv::Mat edgeOrientation = cv::Mat::zeros(img_template.size(), CV_8U);
  //  edge_template.copyTo(edgeOrientation);
  //  cv::bitwise_not ( edgeOrientation, edgeOrientation );

    double line_length = 10.0;
    for(size_t i = 0; i < contours_template.size(); i++) {
      for(size_t j = 0; j < contours_template[i].size(); j+=10) {
        cv::Point pt1 = contours_template[i][j];
        double angle = edges_orientation[i][j] /*+ M_PI_2*/;
        std::cout << "angle=" << (angle * 180.0 / M_PI) << std::endl;
        int x_2 = pt1.x + cos(angle) * line_length;
        int y_2 = pt1.y + sin(angle) * line_length;

        cv::Point pt2(x_2, y_2);
        cv::line(edgeOrientation, pt1, pt2, cv::Scalar(255));
      }
    }
    cv::imshow("edgeOrientation", edgeOrientation);
  }


  //Template mask
  cv::Mat mask;
  createTemplateMask(img_template, mask);

  //Contours Lines
  std::vector<std::vector<Line_info_t> > contours_lines;
  approximateContours(contours_template, contours_lines);

  Template_info_t template_info(contours_template, dist_template, edges_orientation, img_template,
      edge_orientations_template, mask, contours_lines);

  return template_info;
}

/*
 * Keep detections whose the Chamfer distance is below a threshold.
 */
void ChamferMatcher::retainDetections(std::vector<Detection_t> &bbDetections, const float threshold) {
  if(!bbDetections.empty()) {
    //Sort by cost and return only the detection < threshold
    std::sort(bbDetections.begin(), bbDetections.end());

    std::vector<Detection_t> retained_detections;

    for(std::vector<Detection_t>::const_iterator it = bbDetections.begin(); it != bbDetections.end(); ++it) {
      if(it->m_chamferDist < threshold) {
        retained_detections.push_back(*it);
      }
    }

    bbDetections = retained_detections;
  }
}

/*
 * Save template data.
 * Will save only the template image and query ROI as the other information
 * can be computed from this two data.
 */
void ChamferMatcher::saveTemplateData(const std::string &filename) {
  std::ofstream file(filename.c_str(), std::ofstream::binary);

  if(file.is_open()) {
    //Write the number of templates
    int nbTemplates = (int) m_mapOfTemplate_info.size();
    file.write((char *)(&nbTemplates), sizeof(nbTemplates));

    for(std::map<int, std::map<double, Template_info_t> >::const_iterator it = m_mapOfTemplate_info.begin();
        it != m_mapOfTemplate_info.end(); ++it) {
      //Write the id of the template
      int id = it->first;
      file.write((char *)(&id), sizeof(id));


      //Get template object at scale==1.0
      std::map<double, Template_info_t>::const_iterator it_template = it->second.find(1.0);
      if(it_template != it->second.end()) {

        //Save template image
        //Write the number of rows
        int nbRows = it_template->second.m_img.rows;
        file.write((char *)(&nbRows), sizeof(nbRows));

        //Write the number of cols
        int nbCols = it_template->second.m_img.cols;
        file.write((char *)(&nbCols), sizeof(nbCols));

        //Write the number of channel
        int nbChannels = it_template->second.m_img.channels();
        file.write((char *)(&nbChannels), sizeof(nbChannels));

        //Write image data
        file.write((char *)(it_template->second.m_img.data), sizeof(uchar)*nbRows*nbCols*nbChannels);


        //Write the query ROI
        int x = it_template->second.m_queryROI.x;
        file.write((char *)(&x), sizeof(x));

        int y = it_template->second.m_queryROI.y;
        file.write((char *)(&y), sizeof(y));

        int width = it_template->second.m_queryROI.width;
        file.write((char *)(&width), sizeof(width));

        int height = it_template->second.m_queryROI.height;
        file.write((char *)(&height), sizeof(height));
      }
    }

    file.close();
  } else {
    std::cerr << "File: " << filename << " cannot be opened !" << std::endl;
  }
}
