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
#include <iostream>
#include <limits>

#include <opencv2/opencv.hpp>
#include "../Chamfer/include/Chamfer.hpp"
#include "../Chamfer/include/Utils.hpp"

std::string DATA_LOCATION_PREFIX = DATA_DIR;


std::vector<int> createOrientationLUT(int nbCluster) {
  std::vector<int> orientationLUT;
  int maxAngle = 180;
  int step = maxAngle / (double) nbCluster;

  for(int i = 0; i < nbCluster; i++) {
    for(size_t j = 0; j < step; j++) {
      orientationLUT.push_back(i);
    }
  }

  //Last cluster
  for(int i = nbCluster*step; i < maxAngle; i++) {
    orientationLUT.push_back(nbCluster-1);
  }

  return orientationLUT;
}

std::map<int, std::vector<Line_info_t> > clusterLines(const std::vector<std::vector<Line_info_t> > &contours_lines_info) {
  std::vector<int> orientationLUT = createOrientationLUT(12);
  std::map<int, std::vector<Line_info_t> > mapOfLines;

  for(std::vector<std::vector<Line_info_t> >::const_iterator it1 = contours_lines_info.begin();
      it1 != contours_lines_info.end(); ++it1) {
    for(std::vector<Line_info_t>::const_iterator it2 = it1->begin(); it2 != it1->end(); ++it2) {
      double angle = (it2->m_theta-M_PI);
      mapOfLines[ orientationLUT[ (int) ( angle*180.0/M_PI ) ] ].push_back(*it2);
    }
  }

  return mapOfLines;
}

void getLines(const std::vector<std::vector<cv::Point> > &contours, std::vector<std::vector<Line_info_t> > &contours_lines_info) {
  for(size_t i = 0; i < contours.size(); i++) {
    std::vector<Line_info_t> list_lines_info;

    for(size_t j = 0; j < contours[i].size()-1; j++) {
      cv::Point pt1 = contours[i][j];
      cv::Point pt2 = contours[i][j+1];

      double theta, rho, length;
      getPolarLineEquation(pt1, pt2, theta, rho, length);

      Line_info_t line_info(length, rho, theta, pt1, pt2);
      list_lines_info.push_back(line_info);
    }

    contours_lines_info.push_back(list_lines_info);
  }
}

void createOrientedDistanceTransformImage(const cv::Mat &img, cv::Mat &img_dt, const double threshold=50.0) {
  int nbCluster = 12;
  int size[3] = {nbCluster, img.rows, img.cols};
  img_dt = cv::Mat::zeros(3, size, CV_32F);

  //Detect edges
  cv::Mat canny_img;
  cv::Canny(img, canny_img, threshold, 3.0*threshold);

  std::vector<std::vector<cv::Point> > contours, approximated_contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(canny_img, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_TC89_L1);

  //Get lines
  std::vector<std::vector<Line_info_t> > contours_lines_info;
  getLines(contours, contours_lines_info);

  //Cluster lines
  std::map<int, std::vector<Line_info_t> > mapOfLines = clusterLines(contours_lines_info);

//  //Display clustered lines
//  for(std::map<int, std::vector<Line_info_t> >::const_iterator it1 = mapOfLines.begin(); it1 != mapOfLines.end(); ++it1) {
//    cv::Mat img_lines = cv::Mat::zeros(img.size(), CV_8U);
//
//    for(std::vector<Line_info_t>::const_iterator it2 = it1->second.begin(); it2 != it1->second.end(); ++it2) {
//      cv::line(img_lines, it2->m_pointStart, it2->m_PointEnd, cv::Scalar(255));
//    }
//
//    std::stringstream ss;
//    ss << "Cluster=" << it1->first;
//    cv::imshow(ss.str(), img_lines);
//    cv::waitKey(0);
//  }

  //Create multi dimensional distance transform
  std::vector<cv::Mat> vectorOfCurrentDistance;
  for(int h = 0; h < nbCluster; h++) {
    float *ptr_row_dt = img_dt.ptr<float>(h);

    //Create edge image corresponding to the current line orientation
    cv::Mat edge_img = cv::Mat::ones(img.size(), CV_8U)*255;
    for(std::vector<Line_info_t>::const_iterator it = mapOfLines[h].begin(); it != mapOfLines[h].end(); ++it) {
      cv::line(edge_img, it->m_pointStart, it->m_PointEnd, cv::Scalar(0));
    }

    cv::Mat current_dt = cv::Mat(img.size(), CV_32F);
    cv::distanceTransform(edge_img, current_dt, cv::DIST_L2, cv::DIST_MASK_5);
    vectorOfCurrentDistance.push_back(current_dt);

    memcpy(ptr_row_dt, current_dt.data, current_dt.rows*current_dt.cols*sizeof(float));

    cv::Mat dt_display = cv::Mat::zeros(current_dt.size(), CV_8U);
    current_dt.convertTo(dt_display, CV_8U);
    cv::imshow("dt_display", dt_display);
    cv::imshow("edge_img", edge_img);
    cv::waitKey(0);
  }
}
//
//void createIntegralDistanceTransformImage(const cv::Mat &img_dt, cv::Mat &img_idt) {
//  int size[3] = {12, img_dt.rows, img_dt.cols};
//
//  img_idt = cv::Mat::zeros(3, size, CV_32F);
//  for(int h = 0; h < 12; h++) {
//    float *ptr_row_idt = img_idt.ptr<float>(0) + h*img_dt.rows;
//    const float *ptr_row_dt = img_dt.ptr<float>(0);
//
//    //i == 0
//    for(int j = 0; j < img_dt.cols; j++) {
//      ptr_row_idt[j] = ptr_row_dt[j];
//    }
//
//    for(int i = 1; i < img_dt.rows; i++) {
//      ptr_row_idt = img_idt.ptr<float>(i) + h*img_dt.rows;
//      ptr_row_dt = img_dt.ptr<float>(i);
//
//      for(int j = 0; j < img_dt.cols; j++) {
////        float delta_x = tan(M_PI_2 - )
//      }
//
//    }
//  }
//}

void testReconstructEdges(cv::Mat &img, const std::vector<std::vector<cv::Point> > &contours,
    const float scale=1.0f, const float rotation=0.0f) {
  cv::RNG rng(12345);
  std::vector<std::vector<Line_info_t> > contours_lines_info;
  getLines(contours, contours_lines_info);

  cv::Point center = cv::Point( img.cols/2, img.rows/2 );
  cv::Mat affine_transformation = cv::getRotationMatrix2D(center, rotation, scale);

  for(std::vector<std::vector<Line_info_t> >::const_iterator it1 = contours_lines_info.begin();
      it1 != contours_lines_info.end(); ++it1) {
    cv::Scalar color = randomColor(rng);

    for(std::vector<Line_info_t>::const_iterator it2 = it1->begin(); it2 != it1->end(); ++it2) {
      cv::Mat start_point(3, 1, CV_64F), new_start_point(3, 1, CV_64F);
      start_point.at<double>(0) = it2->m_pointStart.x;
      start_point.at<double>(1) = it2->m_pointStart.y;
      start_point.at<double>(2) = 1.0f;

      new_start_point = affine_transformation*start_point;

      cv::Mat end_point(3, 1, CV_64F), new_end_point(3, 1, CV_64F);
      end_point.at<double>(0) = it2->m_PointEnd.x;
      end_point.at<double>(1) = it2->m_PointEnd.y;
      end_point.at<double>(2) = 1.0f;

      new_end_point = affine_transformation*end_point;

      cv::Point2f pt1(new_start_point.at<double>(0), new_start_point.at<double>(1)),
          pt2(new_end_point.at<double>(0), new_end_point.at<double>(1));

      cv::line(img, pt1, pt2, color);
    }
  }
}

void testDecomposeEdgesWithLines(const cv::Mat &img, const double threshold=50.0) {
  cv::RNG rng(12345);

  cv::Mat canny_img;
  cv::Canny(img, canny_img, threshold, 3.0*threshold);

  std::vector<std::vector<cv::Point> > contours, approximated_contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(canny_img, contours, hierarchy, /*CV_RETR_EXTERNAL*/CV_RETR_LIST, CV_CHAIN_APPROX_TC89_L1);

  cv::Mat edge_img = cv::Mat::zeros(img.size(), CV_8UC3);
  cv::Mat approx_poly_img = cv::Mat::zeros(img.size(), CV_8UC3);

  for(size_t i = 0; i < contours.size(); i++) {
    cv::Scalar color = randomColor(rng);
    for(size_t j = 0; j < contours[i].size(); j++) {
      cv::circle(edge_img, contours[i][j], 4, color, 1);
    }

    cv::polylines(edge_img, contours[i], true, color);

    std::vector<cv::Point> approx_contour;
    cv::approxPolyDP(contours[i], approx_contour, 3.0, true);
    approximated_contours.push_back(approx_contour);
    cv::polylines(approx_poly_img, approx_contour, true, color);
  }

  cv::Mat reconstruct_edge = cv::Mat::zeros(img.size(), CV_8UC3);
  testReconstructEdges(reconstruct_edge, approximated_contours, 0.5, 125.0);

  cv::imshow("edge_img", edge_img);
  cv::imshow("approx_poly_img", approx_poly_img);
  cv::imshow("reconstruct_edge", reconstruct_edge);
  cv::waitKey(0);
}

int main(int argc, char **argv) {
//  cv::Mat img_template = cv::imread(DATA_LOCATION_PREFIX + "Template_rectangle.png");
//  cv::Mat img_template = cv::imread(DATA_LOCATION_PREFIX + "Template_triangle.png");
//    cv::Mat img_template = cv::imread(DATA_LOCATION_PREFIX + "Template_circle.png");
//  cv::Mat img_query = cv::imread("Chamfer/Query.png");
//  cv::Mat img_query = cv::imread("Chamfer/Query2.png");
//  cv::Mat img_query = cv::imread("Chamfer/Query4.png");

//  cv::Mat img_template = cv::imread(DATA_LOCATION_PREFIX + "Inria_logo_template.jpg");
//  cv::Mat img_query = cv::imread("Chamfer/Inria_scene.jpg");
//  cv::Mat img_query = cv::imread("Chamfer/Inria_scene2.jpg");
  cv::Mat img_query = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene3.jpg");


//  testDecomposeEdgesWithLines(img_template);
  testDecomposeEdgesWithLines(img_query);

  cv::Mat img_dt;
//  createOrientedDistanceTransformImage(img_template, img_dt);
  createOrientedDistanceTransformImage(img_query, img_dt);


  return 0;
}
