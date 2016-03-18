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
#include <opencv2/highgui/highgui.hpp>


ChamferMatching::ChamferMatching() : m_cannyThreshold(50.0), m_matchingType(edgeForwardBackwardMatching),
    m_query_info(), m_template_info() {
}

/*
 * Detect edges using the Canny method and create and image with edges displayed in black for cv::distanceThreshold
 */
void ChamferMatching::computeCanny(const cv::Mat &img, cv::Mat &edges, const double threshold) {
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
double ChamferMatching::computeChamferDistance(const int offsetX, const int offsetY, cv::Mat &img_res,
    const bool useOrientation, const float lambda, const float weight_foward, const float weight_backward) {
  double chamfer_dist = 0.0;
  int nbElements = 0;

  img_res = cv::Mat::zeros(m_template_info.m_distImg.size(), CV_32F);

  //"Forward matching" <==> matches edges from template to the nearest edges in the query
  for(size_t i = 0; i < m_template_info.m_contours.size(); i++) {
    for(size_t j = 0; j < m_template_info.m_contours[i].size(); j++, nbElements++) {
      int x = m_template_info.m_contours[i][j].x;
      int y = m_template_info.m_contours[i][j].y;

      const float *ptr_row_edge_ori = m_query_info.m_mapOfEdgeOrientation.ptr<float>(y);

      if(useOrientation) {
        chamfer_dist += weight_foward *( m_query_info.m_distImg.ptr<float>(y + offsetY)[x + offsetX]
//         + lambda*(getMinAngleError(template_info.m_edgesOrientation[i][j], ptr_row_edge_ori[j])) ); //Take too long time...
         + lambda*(fabs(m_template_info.m_edgesOrientation[i][j]-ptr_row_edge_ori[x])) ); //Check if there is no angle error pb
//          + lambda*(getMinAngleError(m_template_info.m_edgesOrientation[i][j], ptr_row_edge_ori[x], false, true)) );

        //DEBUG:
        float *ptr_row_res = img_res.ptr<float>(y);
        ptr_row_res[x] = m_query_info.m_distImg.ptr<float>(y + offsetY)[x + offsetX];
      } else {
        chamfer_dist += m_query_info.m_distImg.ptr<float>(y + offsetY)[x + offsetX];

        //DEBUG:
        float *ptr_row_res = img_res.ptr<float>(y);
        ptr_row_res[x] = m_query_info.m_distImg.ptr<float>(y + offsetY)[x + offsetX];
      }
    }
  }

  //"Backward matching" <==> matches edges from query to the nearest edges in the template
  for(size_t i = 0; i < m_query_info.m_contours.size(); i++) {

    for(size_t j = 0; j < m_query_info.m_contours[i].size(); j++, nbElements) {
      int x = m_query_info.m_contours[i][j].x;
      int y = m_query_info.m_contours[i][j].y;
      const float *ptr_row_edge_ori = m_template_info.m_mapOfEdgeOrientation.ptr<float>(y-offsetY);

      if(offsetX <= x && x < offsetX+m_template_info.m_distImg.cols &&
          offsetY <= y && y < offsetY+m_template_info.m_distImg.rows) {
        x -= offsetX;
        y -= offsetY;

        if(useOrientation) {
          chamfer_dist += weight_backward * ( m_template_info.m_distImg.ptr<float>(y)[x] +
              lambda*(fabs(m_query_info.m_edgesOrientation[i][j]-ptr_row_edge_ori[x-offsetX])) );
//              lambda*(getMinAngleError(m_query_info.m_edgesOrientation[i][j], ptr_row_edge_ori[x-offsetX], false, true)) );

          //DEBUG:
          float *ptr_row_res = img_res.ptr<float>(y);
          ptr_row_res[x] = m_template_info.m_distImg.ptr<float>(y)[x];
        } else {
          chamfer_dist += m_template_info.m_distImg.ptr<float>(y)[x];

          //DEBUG:
          float *ptr_row_res = img_res.ptr<float>(y);
          ptr_row_res[x] = m_template_info.m_distImg.ptr<float>(y)[x];
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
void ChamferMatching::computeDistanceTransform(const cv::Mat &img, cv::Mat &dist_img, cv::Mat &labels) {
  dist_img = cv::Mat(img.size(), CV_32F);

  cv::distanceTransform(img, dist_img, labels, cv::DIST_L2, cv::DIST_MASK_5);
}

/*
 * Compute the map that links for each contour id the corresponding indexes i,j in
 * the vector of vectors.
 */
void ChamferMatching::computeEdgeMapIndex(const std::vector<std::vector<cv::Point> > &contours,
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
double ChamferMatching::computeFullChamferDistance(const int offsetX, const int offsetY, cv::Mat &img_res,
    const bool useOrientation, const float lambda) {
  double chamfer_dist = 0.0;
  int nbElements = 0;

  img_res = cv::Mat::zeros(m_template_info.m_distImg.size(), CV_32F);

  if(useOrientation) {
    cv::Mat subDistImg = m_query_info.m_distImg(
        cv::Rect(offsetX, offsetY, m_template_info.m_distImg.cols, m_template_info.m_distImg.rows));
    cv::Mat diffDistTrans = subDistImg - m_template_info.m_distImg;
    diffDistTrans = diffDistTrans.mul(diffDistTrans);
    cv::Scalar sqr_sum = cv::sum(diffDistTrans);
    chamfer_dist += sqr_sum.val[0];

    cv::Mat subEdgeOriImg = m_query_info.m_mapOfEdgeOrientation(
        cv::Rect(offsetX, offsetY, m_template_info.m_distImg.cols, m_template_info.m_distImg.rows));
    cv::Mat diffEdgeOri = subEdgeOriImg - m_template_info.m_mapOfEdgeOrientation;
    diffEdgeOri = diffEdgeOri.mul(diffEdgeOri);
    sqr_sum = cv::sum(diffEdgeOri);
    chamfer_dist += sqr_sum.val[0] * lambda;

    int length = subDistImg.rows*subDistImg.cols;
    nbElements += length;

    //DEBUG:
    img_res += diffDistTrans + diffEdgeOri;
  } else {
    cv::Mat subDistImg = m_query_info.m_distImg(
        cv::Rect(offsetX, offsetY, m_template_info.m_distImg.cols, m_template_info.m_distImg.rows));
    cv::Mat diffDistTrans = subDistImg - m_template_info.m_distImg;
    diffDistTrans = diffDistTrans.mul(diffDistTrans);
    cv::Scalar sqr_sum = cv::sum(diffDistTrans);
    chamfer_dist += sqr_sum.val[0];

    int length = subDistImg.rows*subDistImg.cols;
    nbElements += length;

    //DEBUG:
    img_res += diffDistTrans;
  }

  return chamfer_dist / (2.0*nbElements);
}

/*
 * Compute the image that contains at each pixel location the Chamfer distance.
 */
void ChamferMatching::computeMatchingMap(cv::Mat &chamferMap, const bool useOrientation,
    const int xStep, const int yStep, float lambda) {
  int chamferMapWidth = m_query_info.m_distImg.cols-m_template_info.m_distImg.cols;
  int chamferMapHeight = m_query_info.m_distImg.rows-m_template_info.m_distImg.rows;

  //Set the map at the maximum float value
  chamferMap = std::numeric_limits<float>::max()*
      cv::Mat::ones(chamferMapHeight, chamferMapWidth, CV_32F);

  //DEBUG:
  bool display = false;

//#pragma omp parallel for
  for(int i = 0; i < chamferMapHeight; i += yStep) {
    float *ptr_row = chamferMap.ptr<float>(i);
    for(int j = 0; j < chamferMapWidth; j += xStep) {
      cv::Mat res;

      switch(m_matchingType) {
      case edgeMatching:
      case edgeForwardBackwardMatching:
        ptr_row[j] = computeChamferDistance(j, i, res, useOrientation, 5.0f, 5.0f, 1.0f);
        break;

      case fullMatching:
      default:
        ptr_row[j] = computeFullChamferDistance(j, i, res, useOrientation, lambda);
        break;
      }

      //DEBUG:
      if(display) {
        cv::Mat query_img_roi = m_query_info.m_img(cv::Rect(j, i, m_template_info.m_distImg.cols,
            m_template_info.m_distImg.rows));
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
void ChamferMatching::createMapOfEdgeOrientations(const cv::Mat &img, const cv::Mat &labels, cv::Mat &mapOfEdgeOrientations) {
  std::vector<std::vector<cv::Point> > contours;
  getContours(img, contours);

  std::vector<std::vector<float> > edges_orientation;
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

      ptr_row_edgeOri[j] = edges_orientation[idx1][idx2];
    }
  }
}

/*
 * Create the template mask.
 */
void ChamferMatching::createTemplateMask(const cv::Mat &img, cv::Mat &mask, const double threshold) {
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
void ChamferMatching::detect_impl(const cv::Mat &img_template, std::vector<Detection_t> &bbDetections,
    const bool useOrientation) {
  prepareTemplate(img_template);

  cv::Mat chamferMap;
  float lambda = 10.0;
  computeMatchingMap(chamferMap, useOrientation, lambda);

  //Find the pixel location of the minimal Chamfer distance.
  double minValPrev, minValCur, maxVal;
  cv::Point minLoc, maxLoc;
  cv::minMaxLoc(chamferMap, &minValCur, &maxVal, &minLoc, &maxLoc);

  //Return all the detections in case of similar minimal Chamfer distance.
  int maxLoop = 10, iteration = 0;
  do {
    iteration++;
    minValPrev = minValCur;
    cv::minMaxLoc(chamferMap, &minValCur, &maxVal, &minLoc, &maxLoc);
    chamferMap.at<float>(minLoc.y, minLoc.x) = std::numeric_limits<float>::max();

    cv::Point pt1(minLoc.x, minLoc.y);
    cv::Point pt2 = pt1 + cv::Point(img_template.cols, img_template.rows);

    cv::Rect detection(pt1, pt2);
    Detection_t detect_t(detection, minValCur);
    bbDetections.push_back(detect_t);
  } while( fabs(minValPrev-minValCur) < std::numeric_limits<float>::epsilon() && iteration <= maxLoop );
}

/*
 * Detect on a single scale.
 */
void ChamferMatching::detect(const cv::Mat &img_template, const cv::Mat &img_query, std::vector<Detection_t> &bbDetections,
    const bool useOrientation, const float distanceThresh) {
  prepareQuery(img_query);

  std::vector<Detection_t> all_detections;
  detect_impl(img_template, all_detections, useOrientation);

  //Non maxima suppression
  std::vector<Detection_t> all_maxima_detections;
  nonMaximaSuppression(all_detections, all_maxima_detections);

  //Keep only detection with a Chamfer distance below a threshold
  retainDetections(all_maxima_detections, distanceThresh);

//  bbDetections = all_detections;
  //Group similar detections
  groupDetections(all_maxima_detections, bbDetections);
}

/*
 * Detect on multiple scales.
 */
void ChamferMatching::detectMultiScale(const cv::Mat &img_template, const cv::Mat &img_query,
    std::vector<Detection_t> &bbDetections, const bool useOrientation, const float distanceThresh,
    const double scaleFactor, const double minScale, const double maxScale) {
  prepareQuery(img_query);
  std::vector<Detection_t> all_detections;

  for(double scale = minScale; scale <= maxScale; scale += scaleFactor) {
    cv::Mat img_template_scale;
    cv::resize(img_template, img_template_scale, cv::Size(), scale, scale);

    std::vector<Detection_t> current_detections;
    detect_impl(img_template_scale, current_detections, useOrientation);

    all_detections.insert(all_detections.end(), current_detections.begin(), current_detections.end());
  }

  //Non maxima suppression
  std::vector<Detection_t> all_maxima_detections;
  nonMaximaSuppression(all_detections, all_maxima_detections);

  //Keep only detection with a Chamfer distance below a threshold
  retainDetections(all_maxima_detections, distanceThresh);

//  bbDetections = all_detections;
  //Group similar detections
  groupDetections(all_maxima_detections, bbDetections);
}

/*
 * Filter contours that contains less than a specific number of points.
 */
void ChamferMatching::filterSingleContourPoint(std::vector<std::vector<cv::Point> > &contours, const size_t min) {
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
void ChamferMatching::getContours(const cv::Mat &img, std::vector<std::vector<cv::Point> > &contours, const double threshold) {
  cv::Mat canny_img;
  cv::Canny(img, canny_img, threshold, 3.0*threshold);

  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(canny_img, contours, hierarchy, /*CV_RETR_EXTERNAL*/CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  filterSingleContourPoint(contours);
}

/*
 * Compute for each contour point the corresponding edge orientation.
 * For the current contour point, use the previous and next point to
 * compute the edge orientation.
 */
void ChamferMatching::getContoursOrientation(const std::vector<std::vector<cv::Point> > &contours,
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
void ChamferMatching::groupDetections(const std::vector<Detection_t> &detections,
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
    double xMean = 0.0, yMean = 0.0, distMean = 0.0;

    for(std::vector<Detection_t>::const_iterator it2 = it1->begin(); it2 != it1->end(); ++it2) {
      xMean += it2->m_boundingBox.x;
      yMean += it2->m_boundingBox.y;
      distMean += it2->m_chamferDist;
    }

    xMean /= it1->size();
    yMean /= it1->size();
    distMean /= it1->size();

    Detection_t detection(cv::Rect(xMean, yMean, it1->begin()->m_boundingBox.width,
        it1->begin()->m_boundingBox.height), distMean);
    groupedDetections.push_back(detection);
  }
}

/*
 * Remove detections inside another detections.
 */
void ChamferMatching::nonMaximaSuppression(const std::vector<Detection_t> &detections,
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
void ChamferMatching::prepareQuery(const cv::Mat &img_query) {
  cv::Mat edge_query;
  computeCanny(img_query, edge_query, m_cannyThreshold);

  cv::imshow("edge_query", edge_query);

  cv::Mat dist_query, img_dist_query, labels_query;
  computeDistanceTransform(edge_query, dist_query, labels_query);

  dist_query.convertTo(img_dist_query, CV_8U);
  cv::imshow("img_dist_query", img_dist_query);

  cv::Mat edge_orientations_query;
  createMapOfEdgeOrientations(img_query, labels_query, edge_orientations_query);

  std::vector<std::vector<cv::Point> > contours;
  getContours(img_query, contours);

  std::vector<std::vector<float> > edges_orientation;
  getContoursOrientation(contours, edges_orientation);

  m_query_info = Query_info_t(contours, dist_query, img_query, edge_orientations_query, edges_orientation, labels_query);
}

/*
 * Compute all the necessary information for the template part.
 */
void ChamferMatching::prepareTemplate(const cv::Mat &img_template) {
  cv::Mat edge_template;
  computeCanny(img_template, edge_template, m_cannyThreshold);

  cv::imshow("edge_template", edge_template);

  cv::Mat dist_template, img_dist_template, labels_template;
  computeDistanceTransform(edge_template, dist_template, labels_template);

  dist_template.convertTo(img_dist_template, CV_8U);
  cv::imshow("img_dist_template", img_dist_template);

  cv::Mat edge_orientations_template;
  createMapOfEdgeOrientations(img_template, labels_template, edge_orientations_template);

  std::vector<std::vector<cv::Point> > contours_template;
  getContours(img_template, contours_template);

  std::vector<std::vector<float> > edges_orientation;
  getContoursOrientation(contours_template, edges_orientation);


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


  cv::Mat mask;
  createTemplateMask(img_template, mask);
  m_template_info = Template_info_t(contours_template, dist_template, edge_orientations_template, edges_orientation, mask);
}

/*
 * Keep detections whose the Chamfer distance is below a threshold.
 */
void ChamferMatching::retainDetections(std::vector<Detection_t> &bbDetections, const float threshold) {
  if(!bbDetections.empty()) {
    //Sort by cost and return only the detection < 100
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
