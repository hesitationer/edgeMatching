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

#define DEBUG_LIGHT 1


ChamferMatcher::ChamferMatcher() :
#if DEBUG
      m_debug(false),
#endif
      m_cannyThreshold(50.0), m_maxDescriptorDistanceError(10.0f), m_maxDescriptorOrientationError(0.35f),
      m_minNbDescriptorMatches(5), m_gridDescriptorSize(4,4), m_matchingStrategyType(templateMatching),
      m_matchingType(edgeMatching), /*m_query_info(), */m_mapOfTemplate_info(), m_mapOfTemplateImages(),
      m_orientationLUT(), m_pyramidType(noPyramid), m_rejectionType(gridDescriptorRejection), m_scaleMax(200),
      m_scaleMin(50), m_scaleStep(10), m_scaleVector() {

  int regular_scale = 100;
  m_scaleVector.push_back(regular_scale);

  for(int scale = m_scaleMin; scale <= m_scaleMax; scale += m_scaleStep) {
    if(scale != regular_scale) {
      m_scaleVector.push_back(scale);
    }
  }

  //Sort scales
  std::sort(m_scaleVector.begin(), m_scaleVector.end());
}

ChamferMatcher::ChamferMatcher(const std::map<int, cv::Mat> &mapOfTemplateImages,
    const std::map<int, std::pair<cv::Rect, cv::Rect> > &mapOfTemplateRois) :
#if DEBUG
      m_debug(false),
#endif
      m_cannyThreshold(50.0), m_maxDescriptorDistanceError(10.0f), m_maxDescriptorOrientationError(0.35f),
      m_minNbDescriptorMatches(5), m_gridDescriptorSize(4,4), m_matchingStrategyType(templateMatching),
      m_matchingType(edgeMatching), /*m_query_info(), */m_mapOfTemplate_info(), m_mapOfTemplateImages(),
      m_orientationLUT(), m_pyramidType(noPyramid), m_rejectionType(gridDescriptorRejection), m_scaleMax(200),
      m_scaleMin(50), m_scaleStep(10), m_scaleVector() {

  if(mapOfTemplateImages.size() != mapOfTemplateRois.size()) {
    std::cerr << "Different size between templates and rois!" << std::endl;
    return;
  }

  setTemplateImages(mapOfTemplateImages, mapOfTemplateRois);
}

void ChamferMatcher::approximateContours(const std::vector<std::vector<cv::Point> > &contours,
    std::vector<std::vector<Line_info_t> > &contour_lines, const double epsilon) {

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

    if(!lines.empty()) {
      //Add the lines
      contour_lines.push_back(lines);
    }
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
double ChamferMatcher::computeChamferDistance(const Template_info_t &template_info, const Query_info_t &query_info,
    const int offsetX, const int offsetY,
#if DEBUG
    cv::Mat &img_res,
#endif
    const bool useOrientation, const float lambda, const float weight_forward, const float weight_backward) {
  double chamfer_dist = 0.0;
  int nbElements = 0;

#if DEBUG
  img_res = cv::Mat::zeros(template_info.m_distImg.size(), CV_32F);
#endif

  //TODO: add a normalization step?

  if(m_matchingType == lineMatching || m_matchingType == lineForwardBackwardMatching ||
      m_matchingType == lineIntegralMatching) {
    //Match using approximated lines

    if(m_matchingType == lineIntegralMatching) {

      //Use integral image
      for(size_t i = 0; i < template_info.m_vectorOfContourLines.size(); i++) {
        for(size_t j = 0; j < template_info.m_vectorOfContourLines[i].size(); j++) {
          cv::Point pt1 = template_info.m_vectorOfContourLines[i][j].m_pointStart;
          cv::Point pt2 = template_info.m_vectorOfContourLines[i][j].m_PointEnd;
          cv::Point offset_pt(offsetX, offsetY);

          //Add current window offset
          pt1 += offset_pt;
          pt2 += offset_pt;

          double theta = template_info.m_vectorOfContourLines[i][j].m_theta;
          int use_angle = (int) round((theta-M_PI)*180.0/M_PI) - 90;
          use_angle = use_angle < 0 ? use_angle + 180 : use_angle;

          int h = m_orientationLUT[use_angle];
          const float *ptr_idt_start = query_info.m_integralDistImg.ptr<float>(h) + pt1.y*query_info.m_integralDistImg.size[2];
          const float *ptr_idt_end = query_info.m_integralDistImg.ptr<float>(h) + pt2.y*query_info.m_integralDistImg.size[2];
          float diff_dt = fabs( ptr_idt_start[pt1.x] - ptr_idt_end[pt2.x] );

          if(useOrientation && false) {
//            chamfer_dist += weight_forward * ( query_info.m_distImg.at<float>(it_line.pos()) + lambda *
//                getMinAngleError(template_info.m_mapOfEdgeOrientation.at<float>(it_line.pos()),
//                    query_info.m_mapOfEdgeOrientation.at<float>(it_line.pos()), false, true) );

            const float *ptr_idt_edge_ori_start = query_info.m_integralEdgeOrientation.ptr<float>(h) +
                pt1.y*query_info.m_integralEdgeOrientation.size[2];
            const float *ptr_idt_edge_ori_end = query_info.m_integralEdgeOrientation.ptr<float>(h) +
                pt2.y*query_info.m_integralEdgeOrientation.size[2];
            float diff_edge_ori = fabs( ptr_idt_edge_ori_start[pt1.x] - ptr_idt_edge_ori_end[pt2.x] );

            chamfer_dist += weight_forward * ( diff_dt + lambda *
                            getMinAngleError(theta,
                                diff_edge_ori, false, true) );
          } else {
//            chamfer_dist += weight_forward * ( query_info.m_distImg.at<float>(it_line.pos()) );
            chamfer_dist += weight_forward * ( diff_dt );
          }
        }
      }
    } else {

      //"Forward matching" <==> matches lines from template to the nearest lines in the query
      for(size_t i = 0; i < template_info.m_vectorOfContourLines.size(); i++) {
        for(size_t j = 0; j < template_info.m_vectorOfContourLines[i].size(); j++) {
          cv::Point pt1 = template_info.m_vectorOfContourLines[i][j].m_pointStart;
          cv::Point pt2 = template_info.m_vectorOfContourLines[i][j].m_PointEnd;
          cv::Point offset_pt(offsetX, offsetY);

          //Add current window offset
          pt1 += offset_pt;
          pt2 += offset_pt;

          //Iterate through pixels on line
          cv::LineIterator it_line(query_info.m_distImg, pt1, pt2, 8);
          cv::LineIterator it_line_edge_ori_query(query_info.m_mapOfEdgeOrientation, pt1, pt2, 8);
          cv::LineIterator it_line_edge_ori_template(template_info.m_mapOfEdgeOrientation, pt1-offset_pt, pt2-offset_pt, 8);

          for(int cpt = 0; cpt < it_line.count; cpt++, ++it_line, nbElements++,
              ++it_line_edge_ori_query, ++it_line_edge_ori_template) {
            cv::Point current_pos = it_line.pos();

            if(useOrientation) {
              float value_dist, value_edge_ori_query, value_edge_ori_template;
              memcpy(&value_dist, it_line.ptr, sizeof(float));
              memcpy(&value_edge_ori_query, it_line_edge_ori_query.ptr, sizeof(float));
              memcpy(&value_edge_ori_template, it_line_edge_ori_template.ptr, sizeof(float));

              chamfer_dist += weight_forward * ( value_dist + lambda *
                  getMinAngleError(value_edge_ori_template,
                      value_edge_ori_query, false, true) );
            } else {
              float value;
              memcpy(&value, it_line.ptr, sizeof(float));
              chamfer_dist += weight_forward * ( value );
            }
          }
        }
      }

      if(m_matchingType == lineForwardBackwardMatching) {
        //"Backward matching" <==> matches edges from query to the nearest edges in the template
        for(size_t i = 0; i < query_info.m_vectorOfContourLines.size(); i++) {
          for(size_t j = 0; j < query_info.m_vectorOfContourLines[i].size(); j++) {
            cv::Point pt1 = query_info.m_vectorOfContourLines[i][j].m_pointStart;
            cv::Point pt2 = query_info.m_vectorOfContourLines[i][j].m_PointEnd;
            //TODO:
//            cv::Point offset_pt(offsetX, offsetY);
//
//            //Add current window offset
//            pt1 += offset_pt;
//            pt2 += offset_pt;
//
//            //Iterate through pixels on line
//            cv::LineIterator it_line(template_info.m_distImg, pt1, pt2, 8);
//            for(int cpt = 0; cpt < it_line.count; cpt++, ++it_line, nbElements++) {
//              cv::Point current_pos = it_line.pos();
//
//              if(useOrientation) {
//                chamfer_dist += weight_backward * ( template_info.m_distImg.at<float>(it_line.pos()) + lambda *
//                    getMinAngleError(query_info.m_mapOfEdgeOrientation.at<float>(it_line.pos()),
//                        template_info.m_mapOfEdgeOrientation.at<float>(it_line.pos()), false, true) );
//              } else {
//                chamfer_dist += weight_backward * ( template_info.m_distImg.at<float>(it_line.pos()) );
//                chamfer_dist += weight_backward * ( template_info.m_distImg.at<float>(it_line.pos()) );
//              }
//            }
          }
        }
      }
    }
  } else {
    //Classical edge matching

    //"Forward matching" <==> matches edges from template to the nearest edges in the query
    for(size_t i = 0; i < template_info.m_contours.size(); i++) {
      for(size_t j = 0; j < template_info.m_contours[i].size(); j++, nbElements++) {
        int x = template_info.m_contours[i][j].x;
        int y = template_info.m_contours[i][j].y;

        const float *ptr_row_edge_ori = query_info.m_mapOfEdgeOrientation.ptr<float>(y + offsetY);

        if(useOrientation) {
          chamfer_dist += weight_forward *( query_info.m_distImg.ptr<float>(y + offsetY)[x + offsetX]
             + lambda*(getMinAngleError(template_info.m_edgesOrientation[i][j], ptr_row_edge_ori[x + offsetX], false, true)) );

#if DEBUG
          //DEBUG:
          float *ptr_row_res = img_res.ptr<float>(y);
          ptr_row_res[x] = m_query_info.m_distImg.ptr<float>(y + offsetY)[x + offsetX];
#endif
        } else {
          chamfer_dist += weight_forward * query_info.m_distImg.ptr<float>(y + offsetY)[x + offsetX];

#if DEBUG
          //DEBUG:
          float *ptr_row_res = img_res.ptr<float>(y);
          ptr_row_res[x] = m_query_info.m_distImg.ptr<float>(y + offsetY)[x + offsetX];
#endif
        }
      }
    }

    if(m_matchingType == edgeForwardBackwardMatching) {
      //"Backward matching" <==> matches edges from query to the nearest edges in the template
      for(size_t i = 0; i < query_info.m_contours.size(); i++) {

        for(size_t j = 0; j < query_info.m_contours[i].size(); j++, nbElements) {
          int x = query_info.m_contours[i][j].x;
          int y = query_info.m_contours[i][j].y;
          const float *ptr_row_edge_ori = template_info.m_mapOfEdgeOrientation.ptr<float>(y-offsetY);

          //Get only contours located in the current region
          if(offsetX <= x && x < offsetX+template_info.m_distImg.cols &&
              offsetY <= y && y < offsetY+template_info.m_distImg.rows) {

            if(useOrientation) {
              chamfer_dist += weight_backward * ( template_info.m_distImg.ptr<float>(y-offsetY)[x-offsetX] +
                  lambda*(getMinAngleError(query_info.m_edgesOrientation[i][j], ptr_row_edge_ori[x-offsetX], false, true)) );

#if DEBUG
              //DEBUG:
              float *ptr_row_res = img_res.ptr<float>(y-offsetY);
              ptr_row_res[x-offsetX] = template_info.m_distImg.ptr<float>(y-offsetY)[x-offsetX];
#endif
            } else {
              chamfer_dist += weight_backward * template_info.m_distImg.ptr<float>(y)[x];

#if DEBUG
              //DEBUG:
              float *ptr_row_res = img_res.ptr<float>(y-offsetY);
              ptr_row_res[x-offsetX] = template_info.m_distImg.ptr<float>(y-offsetY)[x-offsetX];
#endif
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
double ChamferMatcher::computeFullChamferDistance(const Template_info_t &template_info, const Query_info_t &query_info,
    const int offsetX, const int offsetY,
#if DEBUG
    cv::Mat &img_res,
#endif
    const bool useOrientation, const float lambda) {
  double chamfer_dist = 0.0;
  int nbElements = 0;

#if DEBUG
  img_res = cv::Mat::zeros(template_info.m_distImg.size(), CV_32F);
#endif

  cv::Mat subDistImg = query_info.m_distImg(
      cv::Rect(offsetX, offsetY, template_info.m_distImg.cols, template_info.m_distImg.rows));

  cv::Mat subEdgeOriImg = query_info.m_mapOfEdgeOrientation(
      cv::Rect(offsetX, offsetY, template_info.m_distImg.cols, template_info.m_distImg.rows));

  if(m_matchingType == fullMatching) {
    //Distance transform
    cv::Mat diffDistTrans;
    cv::absdiff(subDistImg, template_info.m_distImg, diffDistTrans);
    cv::Scalar sqr_sum = cv::sum(diffDistTrans);
    chamfer_dist += sqr_sum.val[0];

    if(useOrientation) {
      //Orientation
      cv::Mat diffEdgeOri;
      cv::absdiff(subEdgeOriImg, template_info.m_mapOfEdgeOrientation, diffEdgeOri);
      sqr_sum = cv::sum(diffEdgeOri);
      chamfer_dist += lambda * sqr_sum.val[0];

#if DEBUG
      //DEBUG:
      img_res += diffDistTrans + diffEdgeOri;
#endif
    }
#if DEBUG
    else {
      //DEBUG:
      img_res += diffDistTrans;
    }
#endif

    int length = subDistImg.rows*subDistImg.cols;
    nbElements += length;
  } else {
    //Get common mask
    cv::Mat common_mask;
    template_info.m_mask.copyTo(common_mask);

    if(m_matchingType == forwardBackwardMaskMatching) {
      cv::Mat query_mask = query_info.m_mask(
          cv::Rect(offsetX, offsetY, template_info.m_distImg.cols, template_info.m_distImg.rows));
      cv::bitwise_or(template_info.m_mask, query_mask, common_mask);
    }

    //Distance Transform
    //Compute the difference only on pixels inside the template mask
    cv::Mat subDistImg_masked;
    subDistImg.copyTo(subDistImg_masked, common_mask);
    cv::Mat templateDistImg_masked;
    template_info.m_distImg.copyTo(templateDistImg_masked, common_mask);

#if DEBUG
    //DEBUG:
    if(m_debug) {
      cv::Mat subDistImg_masked_display;
      double minVal, maxVal;
      cv::minMaxLoc(subDistImg_masked, &minVal, &maxVal);
      subDistImg_masked.convertTo(subDistImg_masked_display, CV_8U, 255.0/(maxVal-minVal), -255.0*minVal/(maxVal-minVal));
      cv::imshow("subDistImg_masked_display", subDistImg_masked_display);
    }
#endif

    cv::Mat diffDistTrans;
    cv::absdiff(subDistImg_masked, templateDistImg_masked, diffDistTrans);
    cv::Scalar sqr_sum = cv::sum(diffDistTrans);
    chamfer_dist += sqr_sum.val[0];

    if(useOrientation) {
      //Orientation
      //Compute the difference only on pixels inside the template mask
      cv::Mat subEdgeOriImg_masked;
      subEdgeOriImg.copyTo(subEdgeOriImg_masked, common_mask);
      cv::Mat templateEdgeOrientation_masked;
      template_info.m_mapOfEdgeOrientation.copyTo(templateEdgeOrientation_masked, common_mask);

      cv::Mat diffEdgeOri;
      cv::absdiff(subEdgeOriImg_masked, templateEdgeOrientation_masked, diffEdgeOri);
      sqr_sum = cv::sum(diffEdgeOri);
      chamfer_dist += lambda * sqr_sum.val[0];

#if DEBUG
      //DEBUG:
      img_res += diffDistTrans + diffEdgeOri;
#endif
    }
#if DEBUG
    else {
      //DEBUG:
      img_res += diffDistTrans;
    }
#endif

    int length = cv::countNonZero(common_mask);
    nbElements += length;
  }

  return chamfer_dist / nbElements;
}

void ChamferMatcher::computeIntegralDistanceTransform(const cv::Mat &dt, cv::Mat &idt, const int nbClusters,
      const bool useLineIterator) {
  int size[3] = {nbClusters, dt.rows, dt.cols};

  idt = cv::Mat(3, size, CV_32F);

  int angle_step = 180 / nbClusters;
  float *ptr_row_idt;
  const float *ptr_row_dt;
  const float *ptr_row_idt_prev;

  std::vector<int> delta_x(nbClusters);
  std::vector<float> delta_s(nbClusters);

  if(!useLineIterator) {
    for(int h = 0; h < nbClusters; h++) {
      int angle = h*angle_step;

      if(angle != 0 && angle != 90) {
        delta_x[h] = round(1 / tan(angle*M_PI/180.0));
        delta_s[h] = 1 / sin(angle*M_PI/180.0);
      }
    }
  }

  for(int h = 0; h < nbClusters; h++) {
    ptr_row_idt = idt.ptr<float>(h);
    ptr_row_dt = dt.ptr<float>(0); //i == 0

    //Initialize first row
    //i == 0
    for(int j = 0; j < dt.cols; j++) {
      ptr_row_idt[j] = ptr_row_dt[j];
    }

    //Initialize first column
    //j == 0
    for(int i = 0; i < dt.rows; i++) {
      ptr_row_idt = idt.ptr<float>(h) + i*dt.cols;
      ptr_row_idt[0] = dt.ptr<float>(i)[0];
    }

    int angle = h*angle_step;
    if(angle == 90) {
      //Vertical lines

      for(int i = 1; i < dt.rows; i++) {
        ptr_row_idt = idt.ptr<float>(h) + i*dt.cols;
        ptr_row_idt_prev = idt.ptr<float>(h) + (i-1)*dt.cols;
        ptr_row_dt = dt.ptr<float>(i);

        for(int j = 0; j < dt.cols; j++) {
          ptr_row_idt[j] = ptr_row_idt_prev[j] + ptr_row_dt[j];
        }
      }
    } else if(angle == 0) {
      //Horizontal lines

      for(int i = 0; i < dt.rows; i++) {
        ptr_row_idt = idt.ptr<float>(h) + i*dt.cols;
        ptr_row_idt_prev = idt.ptr<float>(h) + (i)*dt.cols;
        ptr_row_dt = dt.ptr<float>(i);

        for(int j = 1; j < dt.cols; j++) {
          ptr_row_idt[j] = ptr_row_idt_prev[j-1] + ptr_row_dt[j];
        }
      }
    } else {
      if(useLineIterator) {
        cv::Mat slice_idt(dt.size(), CV_32F, idt.ptr<float>(h));
        cv::Mat slice_idt_mask = cv::Mat::zeros(dt.size(), CV_8U);

        int delta_x = round((dt.rows-1) / tan(angle*M_PI/180.0));
        //Start from top row
        for(int j = 0; j < dt.cols; j++) {
          cv::Point start_diagonal(j, 0), end_diagonal(j+delta_x, dt.rows-1);

          cv::LineIterator it_line_idt(slice_idt, start_diagonal, end_diagonal, 8);
          cv::LineIterator it_line_dt(dt, start_diagonal, end_diagonal, 8);
          cv::LineIterator it_line_idt_mask(slice_idt_mask, start_diagonal, end_diagonal, 8);

          float dt_sum = 0.0f;
          for(int cpt = 0; cpt < it_line_idt.count; cpt++, ++it_line_idt, ++it_line_dt, ++it_line_idt_mask) {
            //Copy current distance transform value
            float value_dt;
            memcpy(&value_dt, it_line_dt.ptr, sizeof(float));

            dt_sum += value_dt;

            //Copy integral distance transform value
            memcpy(it_line_idt.ptr, &dt_sum, sizeof(float));

            //Update IDT mask
            (*it_line_idt_mask.ptr) = 255;
          }
        }

        if(angle < 90) {
          int delta_y = round(tan(angle*M_PI/180.0)*(dt.cols-1));
          //Start from left column
          for(int i = 0; i < dt.rows; i++) {
            cv::Point start_diagonal(0, i), end_diagonal(dt.cols-1, i+delta_y);

            cv::LineIterator it_line_idt(slice_idt, start_diagonal, end_diagonal, 8);
            cv::LineIterator it_line_dt(dt, start_diagonal, end_diagonal, 8);
            cv::LineIterator it_line_idt_mask(slice_idt_mask, start_diagonal, end_diagonal, 8);

            float dt_sum = 0.0f;
            for(int cpt = 0; cpt < it_line_idt.count; cpt++, ++it_line_idt, ++it_line_dt, ++it_line_idt_mask) {
              //Copy current distance transform value
              float value;
              memcpy(&value, it_line_dt.ptr, sizeof(float));

              dt_sum += value;
              //Copy integral distance transform value
              memcpy(it_line_idt.ptr, &dt_sum, sizeof(float));

              //Update IDT mask
              (*it_line_idt_mask.ptr) = 255;
            }
          }
        } else {
          //Initialize last column
          //j == cols-1
          for(int i = 0; i < dt.rows; i++) {
            ptr_row_idt = idt.ptr<float>(h) + i*dt.cols;
            ptr_row_idt[dt.cols-1] = dt.ptr<float>(i)[dt.cols-1];
          }

          int delta_y = round(tan(angle*M_PI/180.0)*( -(dt.cols-1) ));
          //Start from right column
          for(int i = 0; i < dt.rows; i++) {
            cv::Point start_diagonal(dt.cols-1, i), end_diagonal(0, i);
            end_diagonal.y += delta_y;

            cv::LineIterator it_line_idt(slice_idt, start_diagonal, end_diagonal, 8);
            cv::LineIterator it_line_dt(dt, start_diagonal, end_diagonal, 8);
            cv::LineIterator it_line_idt_mask(slice_idt_mask, start_diagonal, end_diagonal, 8);

            float dt_sum = 0.0f;
            for(int cpt = 0; cpt < it_line_idt.count; cpt++, ++it_line_idt, ++it_line_dt, ++it_line_idt_mask) {
              //Copy current distance transform value
              float value;
              memcpy(&value, it_line_dt.ptr, sizeof(float));

              dt_sum += value;
              //Copy integral distance transform value
              memcpy(it_line_idt.ptr, &dt_sum, sizeof(float));

              //Update IDT mask
              (*it_line_idt_mask.ptr) = 255;
            }
          }
        }

        //Fill "holes"
        float *ptr_slice_idt;
        uchar *ptr_slice_idt_mask;

        for(int i = 0; i < slice_idt.rows; i++) {
          ptr_slice_idt_mask = slice_idt_mask.ptr<uchar>(i);
          ptr_slice_idt = slice_idt.ptr<float>(i);

          for(int j = 0; j < slice_idt.cols; j++) {

            //"Hole"
            if(ptr_slice_idt_mask[j] == 0) {
              float nearest_idt;
              bool find_nearest = false;

              for(int a = -1; a <= 1 && !find_nearest; a++) {
                for(int b = -1; b <= 1 && !find_nearest; b++) {
                  int coordX = j + b, coordY = i + a;

                  if(coordX >= 0 && coordX < slice_idt_mask.cols-1 && coordY >= 0 && coordY < slice_idt_mask.rows-1) {
                    if(slice_idt_mask.ptr<uchar>(coordY)[coordX] > 0) {
                      nearest_idt = slice_idt.ptr<float>(coordY)[coordX];
                      find_nearest = true;
                    }
                  }
                }
              }

              //Assign nearest value
              ptr_slice_idt[j] = nearest_idt;
              ptr_slice_idt_mask[j] = 255;
            }
          }
        }

      } else {
        for(int i = 1; i < dt.rows; i++) {
          ptr_row_idt = idt.ptr<float>(h) + i*dt.cols;
          ptr_row_idt_prev = idt.ptr<float>(h) + (i-1)*dt.cols;
          ptr_row_dt = dt.ptr<float>(i);

          for(int j = 0; j < dt.cols; j++) {
            int x_index = j - delta_x[h] >= 0 ? (j - delta_x[h] < dt.cols ? j - delta_x[h] : dt.cols-1) : 0;
            ptr_row_idt[j] = ptr_row_idt_prev[x_index] +
                delta_s[h]*
                ptr_row_dt[j];
          }
        }
      }
    }
  }
}

/*
 * Compute the image that contains at each pixel location the Chamfer distance.
 */
void ChamferMatcher::computeMatchingMap(const Template_info_t &template_info, const Query_info_t &query_info,
    cv::Mat &chamferMap, cv::Mat &rejection_mask, const bool useOrientation, const int xStep, const int yStep,
    const float lambda, const float weight_forward, const float weight_backward) {
  int chamferMapWidth = query_info.m_distImg.cols - template_info.m_distImg.cols + 1;
  int chamferMapHeight = query_info.m_distImg.rows - template_info.m_distImg.rows + 1;

  if(chamferMapWidth <= 0 || chamferMapHeight <= 0) {
    return;
  }

  //Set the map at the maximum float value
  chamferMap = std::numeric_limits<float>::max()*
      cv::Mat::ones(chamferMapHeight, chamferMapWidth, CV_32F);

#if DEBUG
  //DEBUG:
  bool display = true;
#endif

  //Compute the bounding indexes where we want to perform the matching
  int startI = template_info.m_queryROI.y;
  int endI = template_info.m_queryROI.height > 0 ? startI+template_info.m_queryROI.height : chamferMapHeight;
  int startJ = template_info.m_queryROI.x;
  int endJ = template_info.m_queryROI.width > 0 ? startJ+template_info.m_queryROI.width : chamferMapWidth;

  if(m_matchingStrategyType == templatePoseMatching) {
    //Only one Chamfer computation at the location where the template was extracted
    startI = template_info.m_templateLocation.y;
    endI = startI + 1;
    startJ = template_info.m_templateLocation.x;
    endJ = startJ + 1;
  }

  computeRejectionMask(template_info, query_info, rejection_mask, startI, endI, yStep, startJ, endJ, xStep);


#pragma omp parallel for
  for(int i = startI; i < endI; i += yStep) {
    float *ptr_row = chamferMap.ptr<float>(i);
    uchar *ptr_row_rejection_mask = rejection_mask.ptr<uchar>(i);

    for(int j = startJ; j < endJ; j += xStep) {
      if(ptr_row_rejection_mask[j] == 0) {
        continue;
      }

#if DEBUG
      //DEBUG:
      cv::Mat res;
#endif

      switch(m_matchingType) {
      case fullMatching:
      case maskMatching:
      case forwardBackwardMaskMatching:
        ptr_row[j] = computeFullChamferDistance(template_info, query_info, j, i,
#if DEBUG
            res,
#endif
            useOrientation, lambda);
        break;

      case edgeMatching:
      case edgeForwardBackwardMatching:
      case lineMatching:
      case lineForwardBackwardMatching:
      case lineIntegralMatching:
      default:
        ptr_row[j] = computeChamferDistance(template_info, query_info, j, i,
#if DEBUG
            res,
#endif
            useOrientation, lambda, weight_forward, weight_backward);
        break;
      }

#if DEBUG
      //DEBUG:
      if(m_debug && display) {
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
#endif
    }
  }
}

void ChamferMatcher::computeRejectionMask(const Template_info_t &template_info, const Query_info_t &query_info,
    cv::Mat &rejection_mask, const int startI, const int endI, const int yStep, const int startJ,
    const int endJ, const int xStep) {

  if(m_rejectionType == gridDescriptorRejection) {
#pragma omp parallel for
    for(int i = startI; i < endI; i += yStep) {
      uchar *ptr_row_rejection_mask = rejection_mask.ptr<uchar>(i);

      for(int j = startJ; j < endJ; j += xStep) {

        if(ptr_row_rejection_mask[j]) {
          int nbMatches = 0;
          for(size_t cpt = 0; cpt < template_info.m_gridDescriptorsLocations.size(); cpt++) {
            cv::Point location = template_info.m_gridDescriptorsLocations[cpt] + cv::Point(j, i);

            float query_dist = query_info.m_distImg.ptr<float>(location.y)[location.x];
            float query_orientation = query_info.m_mapOfEdgeOrientation.ptr<float>(location.y)[location.x];

            float template_dist = template_info.m_gridDescriptors[cpt].first;
            float template_orientation = template_info.m_gridDescriptors[cpt].second;

            if( std::fabs(query_dist-template_dist) < m_maxDescriptorDistanceError
                && std::fabs(query_orientation-template_orientation) < m_maxDescriptorOrientationError ) {
              nbMatches++;
            }
          }

          if(nbMatches < m_minNbDescriptorMatches) {
            ptr_row_rejection_mask[j] = 0;
          }
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
 * Create a LUT that maps an angle in degree to the corresponding index.
 */
std::vector<int> ChamferMatcher::createOrientationLUT(int nbClusters) {
  std::vector<int> orientationLUT;
  int maxAngle = 180;
  int step = maxAngle / (double) nbClusters;

  for(int i = 0; i < nbClusters; i++) {
    for(size_t j = 0; j < step; j++) {
      orientationLUT.push_back(i);
    }
  }

  //Last cluster
  for(int i = nbClusters*step; i < maxAngle; i++) {
    orientationLUT.push_back(nbClusters-1);
  }

  return orientationLUT;
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
void ChamferMatcher::detect_impl(const Template_info_t &template_info, const Query_info_t &query_info, const int scale,
    std::vector<Detection_t> &currentDetections, cv::Mat &rejection_mask, const bool useOrientation,
    const float distanceThresh, const float lambda, const float weight_forward,
    const float weight_backward, const bool useGroupDetections) {

  cv::Mat chamferMap;
  computeMatchingMap(template_info, query_info, chamferMap, rejection_mask, useOrientation, 5, 5, lambda,
      weight_forward, weight_backward);

  if(!chamferMap.empty()) {
    double minVal, maxVal;
    //Avoid possibility of infinite loop and / or keep a maximum of 100 detections
    int maxLoopIterations = 100, iteration = 0;

    std::vector<Detection_t> all_detections;
    do {
      iteration++;

      //Find the pixel location of the minimal Chamfer distance.
      cv::Point minLoc, maxLoc;
      cv::minMaxLoc(chamferMap, &minVal, &maxVal, &minLoc, &maxLoc);

      //"Reset the location" to find other detections
      chamferMap.at<float>(minLoc.y, minLoc.x) = std::numeric_limits<float>::max();

      cv::Point pt1(minLoc.x, minLoc.y);
      cv::Point pt2 = pt1 + cv::Point(template_info.m_distImg.cols, template_info.m_distImg.rows);

      if(minVal < distanceThresh) {
        //Add the detection
        cv::Rect detection(pt1, pt2);
        Detection_t detect_t(detection, minVal, scale);
        all_detections.push_back(detect_t);
      }
    } while( minVal < distanceThresh && iteration <= maxLoopIterations );

    //Group similar detections
    if(useGroupDetections) {
      groupDetections(all_detections, currentDetections);
    } else {
      currentDetections = all_detections;
    }

    //Sort detections by increasing cost
    std::sort(currentDetections.begin(), currentDetections.end());

    if(m_pyramidType == pyramid2) {
      rejection_mask = (chamferMap < distanceThresh);
    }
  }
}

/*
 * Detect on a single scale.
 */
void ChamferMatcher::detect(const cv::Mat &img_query, std::vector<Detection_t> &detections, const bool useOrientation,
    const float distanceThresh, const float lambda, const float weight_forward, const float weight_backward,
    const bool useGroupDetections) {
  detections.clear();


  int half_scale = 50, regular_scale = 100;

  cv::Mat half_query;
  Query_info_t half_query_info;
  if(m_pyramidType != noPyramid) {
    //PyrDown
    cv::pyrDown(img_query, half_query);
    half_query_info = prepareQuery(half_query);
  }

  Query_info_t query_info = prepareQuery(img_query);

  for(std::map<int, std::map<int, Template_info_t> >::const_iterator it = m_mapOfTemplate_info.begin();
      it != m_mapOfTemplate_info.end(); ++it) {
    std::vector<Detection_t> all_detections;

    std::map<int, Template_info_t>::const_iterator it_template = it->second.find(regular_scale);
    if(it_template != it->second.end()) {
      int chamferMapWidth = query_info.m_distImg.cols - it_template->second.m_distImg.cols + 1;
      int chamferMapHeight = query_info.m_distImg.rows - it_template->second.m_distImg.rows + 1;

      if(chamferMapWidth > 0 && chamferMapHeight > 0) {
        cv::Mat rejection_mask = cv::Mat::ones(chamferMapHeight, chamferMapWidth, CV_8U);


        if(m_pyramidType != noPyramid && m_matchingStrategyType != templatePoseMatching) {
          //Half size
          std::map<int, Template_info_t>::const_iterator it_template_half = it->second.find(half_scale);

          if(it_template_half != it->second.end()) {
            int half_chamferMapWidth = half_query_info.m_distImg.cols - it_template_half->second.m_distImg.cols + 1;
            int half_chamferMapHeight = half_query_info.m_distImg.rows - it_template_half->second.m_distImg.rows + 1;

            if(half_chamferMapWidth > 0 && half_chamferMapHeight > 0) {
              cv::Mat half_rejection_mask = cv::Mat::ones(half_chamferMapHeight, half_chamferMapWidth, CV_8U);

              //Compute the bounding indexes where we want to perform the matching
              int startI = it_template_half->second.m_queryROI.y;
              int endI = it_template_half->second.m_queryROI.height > 0 ?
                  startI + it_template_half->second.m_queryROI.height/2 : half_chamferMapHeight;
              int startJ = it_template_half->second.m_queryROI.x;
              int endJ = it_template_half->second.m_queryROI.width > 0 ?
                  startJ + it_template_half->second.m_queryROI.width/2 : half_chamferMapWidth;

              if(m_pyramidType == pyramid1) {
                computeRejectionMask(it_template_half->second, half_query_info, half_rejection_mask,
                    startI, endI, 5, startJ, endJ, 5);
              } else {
                std::vector<Detection_t> half_detections;
                detect_impl(it_template_half->second, half_query_info, half_scale, half_detections, half_rejection_mask,
                    useOrientation, distanceThresh, lambda, weight_forward, weight_backward, useGroupDetections);
              }

              //Use dilate to increase regions of interest
              int dilation_size = 5;
              cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                  cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                  cv::Point( dilation_size, dilation_size ) );

              cv::dilate(half_rejection_mask, half_rejection_mask, element);

              //Resize the mask to the current size
              cv::resize(half_rejection_mask, rejection_mask, cv::Size(chamferMapWidth, chamferMapHeight),
                  0.0, 0.0, cv::INTER_NEAREST);
            }
          }
        }

#if DEBUG
        rejection_mask *= 255;
        std::stringstream ss;
        ss << "rejection_mask_" << it->first;
        cv::imshow(ss.str(), rejection_mask);
        cv::waitKey(30);
#endif

        //Regular scale
        detect_impl(it_template->second, query_info, regular_scale, all_detections, rejection_mask, useOrientation,
            distanceThresh, lambda, weight_forward, weight_backward, useGroupDetections);

        //Set Template index
        for(std::vector<Detection_t>::iterator it_detection = all_detections.begin();
            it_detection != all_detections.end(); ++it_detection) {
          it_detection->m_templateIndex = it->first;
        }

        detections.insert(detections.end(), all_detections.begin(), all_detections.end());
      }
    } else {
      std::cerr << "Cannot find template at regular scale!" << std::endl;
      return;
    }
  }

  //Sort detections by increasing cost
  std::sort(detections.begin(), detections.end());
}

/*
 * Detect on multiple scales.
 */
void ChamferMatcher::detectMultiScale(const cv::Mat &img_query, std::vector<Detection_t> &detections,
    const bool useOrientation, const float distanceThresh, const float lambda, const float weight_forward,
    const float weight_backward, const bool useNonMaximaSuppression, const bool useGroupDetections) {
  detections.clear();

  if(m_matchingStrategyType == templatePoseMatching) {
    std::cerr << "Cannot detect on multiple scales with the matching strategy=templatePoseMatching!" << std::endl;
    return;
  }

  cv::Mat half_query;
  Query_info_t half_query_info;
  if(m_pyramidType != noPyramid) {
    //PyrDown
    cv::pyrDown(img_query, half_query);
    half_query_info = prepareQuery(half_query);
  }

  Query_info_t query_info = prepareQuery(img_query);

  for(std::map<int, std::map<int, Template_info_t> >::iterator it1 = m_mapOfTemplate_info.begin();
      it1 != m_mapOfTemplate_info.end(); ++it1) {
    std::vector<Detection_t> all_detections;

    for(std::vector<int>::const_iterator it2 = m_scaleVector.begin(); it2 != m_scaleVector.end(); ++it2) {
      std::map<int, Template_info_t>::const_iterator it_tpl_scale = it1->second.find(*it2);

      if(it_tpl_scale != it1->second.end()) {
        std::vector<Detection_t> current_detections;

        int chamferMapWidth = query_info.m_distImg.cols - it_tpl_scale->second.m_distImg.cols + 1;
        int chamferMapHeight = query_info.m_distImg.rows - it_tpl_scale->second.m_distImg.rows + 1;

        if(chamferMapWidth > 0 && chamferMapHeight > 0) {
          cv::Mat rejection_mask = cv::Mat::ones(chamferMapHeight, chamferMapWidth, CV_8U);


          int half_scale = (*it2) / 2;
          if(m_pyramidType != noPyramid && half_scale > 0 && m_matchingStrategyType != templatePoseMatching) {
            //Half size
            std::map<int, Template_info_t>::const_iterator it_template_half = it1->second.find(half_scale);

            if(it_template_half != it1->second.end()) {
              int half_chamferMapWidth = half_query_info.m_distImg.cols - it_template_half->second.m_distImg.cols + 1;
              int half_chamferMapHeight = half_query_info.m_distImg.rows - it_template_half->second.m_distImg.rows + 1;

              if(half_chamferMapWidth > 0 && half_chamferMapHeight > 0) {
                cv::Mat half_rejection_mask = cv::Mat::ones(half_chamferMapHeight, half_chamferMapWidth, CV_8U);

                //Compute the bounding indexes where we want to perform the matching
                int startI = it_template_half->second.m_queryROI.y;
                int endI = it_template_half->second.m_queryROI.height > 0 ?
                    startI + it_template_half->second.m_queryROI.height/2 : half_chamferMapHeight;
                int startJ = it_template_half->second.m_queryROI.x;
                int endJ = it_template_half->second.m_queryROI.width > 0 ?
                    startJ + it_template_half->second.m_queryROI.width/2 : half_chamferMapWidth;

                if(m_pyramidType == pyramid1) {
                  computeRejectionMask(it_template_half->second, half_query_info, rejection_mask, startI, endI, 5, startJ, endJ, 5);
                } else {
                  std::vector<Detection_t> half_detections;
                  detect_impl(it_template_half->second, half_query_info, half_scale, half_detections, half_rejection_mask,
                      useOrientation, distanceThresh, lambda, weight_forward, weight_backward, useGroupDetections);
                }

                //Use dilate to increase regions of interest
                int dilation_size = 5;
                cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                    cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                    cv::Point( dilation_size, dilation_size ) );

                cv::dilate(half_rejection_mask, half_rejection_mask, element);

                //Resize the mask to the current size
                cv::resize(half_rejection_mask, rejection_mask, cv::Size(chamferMapWidth, chamferMapHeight),
                    0.0, 0.0, cv::INTER_NEAREST);
              }
            }
          }


          detect_impl(it_tpl_scale->second, query_info, it_tpl_scale->first, current_detections, rejection_mask, useOrientation,
              distanceThresh, lambda, weight_forward, weight_backward, useGroupDetections);

          //Set Template index
          for(std::vector<Detection_t>::iterator it_detection = current_detections.begin();
              it_detection != current_detections.end(); ++it_detection) {
            it_detection->m_templateIndex = it1->first;
          }

          all_detections.insert(all_detections.end(), current_detections.begin(), current_detections.end());
        }
      }
    }

//    //Non maxima suppression
//    std::vector<Detection_t> all_maxima_detections;
//    if(useNonMaximaSuppression) {
//      nonMaximaSuppression(all_detections, all_maxima_detections);
//    } else {
//      all_maxima_detections = all_detections;
//    }
//
//    //Group similar detections
//    if(useGroupDetections) {
//      groupDetections(all_maxima_detections, detections);
//    } else {
//      detections.insert(detections.end(), all_maxima_detections.begin(), all_maxima_detections.end());
//    }
    detections.insert(detections.end(), all_detections.begin(), all_detections.end());
  }

  //Sort detections by increasing cost
  std::sort(detections.begin(), detections.end());
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

    if(it_contour->size() > 2) {
      for(std::vector<cv::Point>::const_iterator it_point = it_contour->begin()+1;
          it_point != it_contour->end(); ++it_point) {
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
    std::vector<Detection_t> &groupedDetections, const double overlapPercentage) {
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

          if(overlapping_percentage > overlapPercentage) {
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

    std::map<int, int> mapOfOccurrences;
    for(std::vector<Detection_t>::const_iterator it2 = it1->begin(); it2 != it1->end(); ++it2) {
      xMean += it2->m_boundingBox.x;
      yMean += it2->m_boundingBox.y;
      distMean += it2->m_chamferDist;
      scaleMean += it2->m_scale;

      mapOfOccurrences[it2->m_templateIndex]++;
    }

    xMean /= it1->size();
    yMean /= it1->size();
    distMean /= it1->size();
    scaleMean /= it1->size();

    int maxOccurrenceIndex = -1, maxOccurrence = 0;
    for(std::map<int, int>::const_iterator it_index = mapOfOccurrences.begin();
        it_index != mapOfOccurrences.end(); ++it_index) {
      if(maxOccurrence < it_index->second) {
        maxOccurrence = it_index->second;
        maxOccurrenceIndex = it_index->first;
      }
    }

    Detection_t detection(cv::Rect(xMean, yMean, it1->begin()->m_boundingBox.width,
        it1->begin()->m_boundingBox.height), distMean, scaleMean, maxOccurrenceIndex);
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
    //Clean the maps
    m_mapOfTemplate_info.clear();
    m_mapOfTemplateImages.clear();

#define COMPUTE_AFTER_READ 1

#if COMPUTE_AFTER_READ
    std::map<int, cv::Mat> mapOfTemplateImages;
    std::map<int, std::pair<cv::Rect, cv::Rect> > mapOfTemplateRois;
#endif

    //Read the type of the save
    int saveType = 0;
    file.read((char *)(&saveType), sizeof(saveType));

    bool isSavedSingleFile = saveType != 0;


    //Read the number of templates
    int nbTemplates = 0;
    file.read((char *)(&nbTemplates), sizeof(nbTemplates));

    for(int cpt = 0; cpt < nbTemplates; cpt++) {
      //Read the id of the template
      int id = 0;
      file.read((char *)(&id), sizeof(id));


      cv::Mat img;
      if(isSavedSingleFile) {
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
        if(nbChannels == 3) {
          img = cv::Mat(nbRows, nbCols, CV_8UC3, data);
        } else {
          img = cv::Mat(nbRows, nbCols, CV_8U, data);
        }
      } else {
        //Read image path
        int path_length = 0;
        file.read((char *)(&path_length), sizeof(path_length));

        char* image_path = new char[path_length + 1];
        for(int cpt = 0; cpt < path_length; cpt++) {
          char c;
          file.read((char *)(&c), sizeof(c));
          image_path[cpt] = c;
        }
        image_path[path_length] = '\0';

        img = cv::imread(std::string(image_path));
      }

      //Add image
#if COMPUTE_AFTER_READ
      mapOfTemplateImages[id] = img;
#else
      m_mapOfTemplateImages[id] = img;
#endif


      //Read the template location and size
      int x_tpl = 0;
      file.read((char *)(&x_tpl), sizeof(x_tpl));

      int y_tpl = 0;
      file.read((char *)(&y_tpl), sizeof(y_tpl));

      int width_tpl = 0;
      file.read((char *)(&width_tpl), sizeof(width_tpl));

      int height_tpl = 0;
      file.read((char *)(&height_tpl), sizeof(height_tpl));

      //Create template location and size
      cv::Rect templateLocation(x_tpl, y_tpl, width_tpl, height_tpl);


      //Read the query ROI
      int x_roi = 0;
      file.read((char *)(&x_roi), sizeof(x_roi));

      int y_roi = 0;
      file.read((char *)(&y_roi), sizeof(y_roi));

      int width_roi = 0;
      file.read((char *)(&width_roi), sizeof(width_roi));

      int height_roi = 0;
      file.read((char *)(&height_roi), sizeof(height_roi));

      //Create query ROI
      cv::Rect queryROI(x_roi, y_roi, width_roi, height_roi);

#if COMPUTE_AFTER_READ
      mapOfTemplateRois[id] = std::pair<cv::Rect, cv::Rect>(templateLocation, queryROI);
#else
      //Create Template
      Template_info_t template_info = prepareTemplate(img);
      template_info.m_queryROI = queryROI;
      template_info.m_templateLocation = templateLocation;

      int regular_scale = 100;
      m_mapOfTemplate_info[id][regular_scale] = template_info;
#endif
    }

    file.close();

#if COMPUTE_AFTER_READ
    bool deepCopy = false;
    setTemplateImages(mapOfTemplateImages, mapOfTemplateRois, deepCopy);
#else
    //Compute template information for all the scales between [m_scaleMin ; m_scaleMax]
    setScale(m_scaleMin, m_scaleMax, m_scaleStep);
#endif
  } else {
    std::cerr << "File: " << filename << " cannot be opened !" << std::endl;
    return;
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
Query_info_t ChamferMatcher::prepareQuery(const cv::Mat &img_query) {
  cv::Mat edge_query;
  computeCanny(img_query, edge_query, m_cannyThreshold);

#if DEBUG_LIGHT
  cv::imshow("edge_query", edge_query);
#endif
  //XXX:
  //  cv::imwrite("Edge_query.png", edge_query);

  cv::Mat dist_query, img_dist_query, labels_query;
  computeDistanceTransform(edge_query, dist_query, labels_query);

#if DEBUG_LIGHT
  dist_query.convertTo(img_dist_query, CV_8U);
  cv::imshow("img_dist_query", img_dist_query);
#endif

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

  int nbClusters = 12;
  //Compute IDT
  cv::Mat query_idt, query_idt_edge_ori;
  ChamferMatcher::computeIntegralDistanceTransform(dist_query, query_idt, nbClusters, true);
  ChamferMatcher::computeIntegralDistanceTransform(edge_orientations_query, query_idt_edge_ori, nbClusters, true);

  //Create orientation LUT
  m_orientationLUT = ChamferMatcher::createOrientationLUT(nbClusters);

  return Query_info_t(contours, dist_query, img_query, query_idt, query_idt_edge_ori, edge_orientations_query,
      edges_orientation, labels_query, mask, contours_lines);
}

/*
 * Compute all the necessary information for the template part.
 */
Template_info_t ChamferMatcher::prepareTemplate(const cv::Mat &img_template) {
  cv::Mat edge_template;
  computeCanny(img_template, edge_template, m_cannyThreshold);

#if DEBUG_LIGHT
  cv::imshow("edge_template", edge_template);
#endif
  //XXX:
  //  cv::imwrite("Edge_template.png", edge_template);

  cv::Mat dist_template, img_dist_template, labels_template;
  computeDistanceTransform(edge_template, dist_template, labels_template);

#if DEBUG_LIGHT
  dist_template.convertTo(img_dist_template, CV_8U);
  cv::imshow("img_dist_template", img_dist_template);
#endif

  cv::Mat edge_orientations_template;
  std::vector<std::vector<cv::Point> > contours_template;
  std::vector<std::vector<float> > edges_orientation;
  createMapOfEdgeOrientations(img_template, labels_template, edge_orientations_template, contours_template, edges_orientation);


#if DEBUG
  //DEBUG:
  if(m_debug) {
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
#endif


  //Template mask
  cv::Mat mask;
  createTemplateMask(img_template, mask);

  //Contours Lines
  std::vector<std::vector<Line_info_t> > contours_lines;
  approximateContours(contours_template, contours_lines);

  Template_info_t template_info(contours_template, dist_template, edges_orientation, m_gridDescriptorSize,
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
void ChamferMatcher::saveTemplateData(const std::string &filename, const bool saveSingleFile) {
  std::ofstream file(filename.c_str(), std::ofstream::binary);

  if(file.is_open()) {
    //Write the type of the save
    int saveType = saveSingleFile;
    file.write((char *)(&saveType), sizeof(saveType));

    //Write the number of templates
    int nbTemplates = (int) m_mapOfTemplate_info.size();
    file.write((char *)(&nbTemplates), sizeof(nbTemplates));

    int cpt_img = 0;
    for(std::map<int, std::map<int, Template_info_t> >::const_iterator it = m_mapOfTemplate_info.begin();
        it != m_mapOfTemplate_info.end(); ++it, cpt_img++) {
      //Write the id of the template
      int id = it->first;
      file.write((char *)(&id), sizeof(id));


      //Get template object at scale==100
      int regular_scale = 100;
      std::map<int, Template_info_t>::const_iterator it_template = it->second.find(regular_scale);

      //Get template image
      std::map<int, cv::Mat>::const_iterator it_image = m_mapOfTemplateImages.find(it->first);

      if(it_template != it->second.end() && it_image != m_mapOfTemplateImages.end()) {

        if(saveSingleFile) {
          //Save template image
          //Write the number of rows
          int nbRows = it_image->second.rows;
          file.write((char *)(&nbRows), sizeof(nbRows));

          //Write the number of cols
          int nbCols = it_image->second.cols;
          file.write((char *)(&nbCols), sizeof(nbCols));

          //Write the number of channel
          int nbChannels = it_image->second.channels();
          file.write((char *)(&nbChannels), sizeof(nbChannels));

          //Write image data
          //Warning: works only with continuous cv::Mat
          file.write((char *)(it_image->second.data), sizeof(uchar)*nbRows*nbCols*nbChannels);
        } else {
          std::stringstream ss;
          std::string parent = filename.find("/") != std::string::npos ? filename.substr( 0, filename.find_last_of("/") ) : "";
          ss << parent << "/";

          char buffer[20];
          sprintf (buffer, "template_%04d.png", cpt_img);
          std::string image_filename = buffer;
          ss << image_filename;

          //Save image on disk
          std::string filepath = ss.str();
          cv::imwrite(filepath, it_image->second);

          //Save path length
          int path_length = (int) filepath.length();
          file.write((char *)(&path_length), sizeof(path_length));

          //Save image path
          for(int cpt = 0; cpt < path_length; cpt++) {
            file.write((char *) (&filepath[(size_t) cpt]), sizeof(filepath[(size_t) cpt]));
          }
        }


        //Write the template location and size
        int x_tpl = it_template->second.m_templateLocation.x;
        file.write((char *)(&x_tpl), sizeof(x_tpl));

        int y_tpl = it_template->second.m_templateLocation.y;
        file.write((char *)(&y_tpl), sizeof(y_tpl));

        int width_tpl = it_template->second.m_templateLocation.width;
        file.write((char *)(&width_tpl), sizeof(width_tpl));

        int height_tpl = it_template->second.m_templateLocation.height;
        file.write((char *)(&height_tpl), sizeof(height_tpl));


        //Write the query ROI
        int x_roi = it_template->second.m_queryROI.x;
        file.write((char *)(&x_roi), sizeof(x_roi));

        int y_roi = it_template->second.m_queryROI.y;
        file.write((char *)(&y_roi), sizeof(y_roi));

        int width_roi = it_template->second.m_queryROI.width;
        file.write((char *)(&width_roi), sizeof(width_roi));

        int height_roi = it_template->second.m_queryROI.height;
        file.write((char *)(&height_roi), sizeof(height_roi));
      } else {
        std::cerr << "Cannot find the template info for scale=1 or cannot find the template image!" << std::endl;
      }
    }

    file.close();
  } else {
    std::cerr << "File: " << filename << " cannot be opened !" << std::endl;
  }
}

void ChamferMatcher::setScale(const int min, const int max, const int step) {
  if(min > 0 && max > 0 && max >= min && step > 0) {
    m_scaleMin = min;
    m_scaleMax = max;
    m_scaleStep = step;

    m_scaleVector.clear();
    int regular_scale = 100;
    m_scaleVector.push_back(regular_scale);

    for(int scale = m_scaleMin; scale <= m_scaleMax; scale += m_scaleStep) {
      if(scale != regular_scale) {
        m_scaleVector.push_back(scale);
      }
    }

    //Sort scales
    std::sort(m_scaleVector.begin(), m_scaleVector.end());


    for(std::map<int, std::map<int, Template_info_t> >::iterator it_tpl = m_mapOfTemplate_info.begin();
        it_tpl != m_mapOfTemplate_info.end(); ++it_tpl) {

      //Get the template image
      std::map<int, cv::Mat>::const_iterator it_image = m_mapOfTemplateImages.find(it_tpl->first);

      //Get the template at regular scale
      std::map<int, Template_info_t>::const_iterator it_tpl_reg_scale = it_tpl->second.find(regular_scale);

      if(it_image != m_mapOfTemplateImages.end() && it_tpl_reg_scale != it_tpl->second.end()) {

        //Compute template information for all the scales between [m_scaleMin ; m_scaleMax]
        for(int scale = m_scaleMin; scale <= m_scaleMax; scale += m_scaleStep) {

          std::map<int, Template_info_t>::const_iterator it_scale = m_mapOfTemplate_info[it_tpl->first].find(scale);
          if(scale != regular_scale && it_scale == m_mapOfTemplate_info[it_tpl->first].end()) {
            //The scale is not present and is different of 100
            cv::Mat img_template_scale;
            cv::resize(it_image->second, img_template_scale, cv::Size(), scale/100.0, scale/100.0);

            m_mapOfTemplate_info[it_tpl->first][scale] = prepareTemplate(img_template_scale);

            //Set query ROI
            m_mapOfTemplate_info[it_tpl->first][scale].m_queryROI = it_tpl_reg_scale->second.m_queryROI;
          }

          if(m_pyramidType != noPyramid) {
            int half_scale = scale / 2;

            if(half_scale > 0) {
              it_scale = m_mapOfTemplate_info[it_tpl->first].find(half_scale);

              if(it_scale == m_mapOfTemplate_info[it_tpl->first].end()) {
                cv::Mat img_template_scale;
                cv::resize(it_image->second, img_template_scale, cv::Size(), half_scale/100.0, half_scale/100.0);

                m_mapOfTemplate_info[it_tpl->first][half_scale] = prepareTemplate(img_template_scale);
              }
            }
          }
        }
      } else {
        std::cerr << "Cannot find the template image!" << std::endl;
      }
    }

    //Vector of valid scales (multi-scales + pyramid half scale if sets)
    std::vector<int> vectorOfScales;
    vectorOfScales.push_back(regular_scale);
    for(int scale = m_scaleMin; scale <= m_scaleMax; scale += m_scaleStep) {
      vectorOfScales.push_back(scale);

      if(m_pyramidType != noPyramid) {
        if(scale / 2 > 0) {
          //Add half scale
          vectorOfScales.push_back(scale / 2);
        }
      }
    }

    //Remove obsolete scales
    for(std::map<int, std::map<int, Template_info_t> >::iterator it_tpl = m_mapOfTemplate_info.begin();
        it_tpl != m_mapOfTemplate_info.end(); ++it_tpl) {

      for(std::map<int, Template_info_t>::iterator it_tpl_scale = it_tpl->second.begin();
          it_tpl_scale != it_tpl->second.end(); ) {
        if( std::find(vectorOfScales.begin(), vectorOfScales.end(), it_tpl_scale->first) == vectorOfScales.end() ) {
          //Delete obsolete scale
          it_tpl->second.erase(it_tpl_scale++);
        } else {
          ++it_tpl_scale;
        }
      }
    }
  } else {
    std::cerr << "Invalid scale parameter !" << std::endl;
  }
}

void ChamferMatcher::setTemplateImages(const std::map<int, cv::Mat> &mapOfTemplateImages,
    const std::map<int, std::pair<cv::Rect, cv::Rect> > &mapOfTemplateRois, const bool deepCopy) {
  m_mapOfTemplate_info.clear();
  m_mapOfTemplateImages.clear();

  if(mapOfTemplateImages.size() != mapOfTemplateRois.size()) {
    std::cerr << "Different size between templates and rois!" << std::endl;
    return;
  }

  int regular_scale = 100;
  for(std::map<int, cv::Mat>::const_iterator it_tpl = mapOfTemplateImages.begin();
      it_tpl != mapOfTemplateImages.end(); ++it_tpl) {

    //Set template image
    if(deepCopy) {
      m_mapOfTemplateImages[it_tpl->first] = it_tpl->second.clone(); //Clone to avoid modification problem
    } else {
      m_mapOfTemplateImages[it_tpl->first] = it_tpl->second;
    }

    //Precompute the template information for scale=100
    m_mapOfTemplate_info[it_tpl->first][regular_scale] = prepareTemplate(it_tpl->second);

    std::map<int, std::pair<cv::Rect, cv::Rect> >::const_iterator it_roi = mapOfTemplateRois.find(it_tpl->first);
    if(it_roi == mapOfTemplateRois.end()) {
      std::cerr << "The id: " << it_roi->first << " does not exist in template rois!" << std::endl;
      return;
    }

    //Set template location
    m_mapOfTemplate_info[it_tpl->first][regular_scale].m_templateLocation = it_roi->second.first;

    //Set query ROI
    m_mapOfTemplate_info[it_tpl->first][regular_scale].m_queryROI = it_roi->second.second;
  }

  setScale(m_scaleMin, m_scaleMax, m_scaleStep);
}
