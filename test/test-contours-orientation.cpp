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
#include "../Chamfer/include/Utils.hpp"
#include "../Chamfer/include/Chamfer.hpp"
#include <opencv2/highgui/highgui.hpp>

std::string DATA_LOCATION_PREFIX = DATA_DIR;


void checkContoursIndex(const cv::Mat &distance_img, const cv::Mat &labels,
    const std::vector<std::vector<cv::Point> > &contours,
    const std::map<int, std::pair<int, int> > &mapOfIndex) {
  std::map<int, cv::Point> mapOfContourIndex;

  //Get the corresponding point for the given label
  for(int i = 0; i < distance_img.rows; i++) {
    for(int j = 0; j < distance_img.cols; j++) {
      if(distance_img.at<float>(i,j) == 0) {
        int label = labels.at<int>(i,j);
        mapOfContourIndex[label] = cv::Point(j,i);
      }
    }
  }

  //Check if we have the same result
  for(std::map<int, std::pair<int, int> >::const_iterator it = mapOfIndex.begin(); it != mapOfIndex.end(); ++it) {
    cv::Point point1 = contours[it->second.first][it->second.second];
    cv::Point point2 = mapOfContourIndex[it->first];

    if(point1 != point2) {
      std::cerr << "Points are different !" << std::endl;
      return;
    }
  }
}

void checkContoursIndex(const std::string &filename) {
  cv::Mat img = cv::imread(filename);

  cv::Mat edge;
  ChamferMatcher::computeCanny(img, edge, 70.0);

  cv::Mat dist_transform, labels;
  ChamferMatcher::computeDistanceTransform(edge, dist_transform, labels);

  //Check
  std::vector<std::vector<cv::Point> > contours;
  ChamferMatcher::getContours(img, contours);

  std::map<int, std::pair<int, int> > mapOfIndex;
  ChamferMatcher::computeEdgeMapIndex(contours, labels, mapOfIndex);

  checkContoursIndex(dist_transform, labels, contours, mapOfIndex);

  std::cout << "Contour labels are OK." << std::endl;
}

void testContoursOrientation(const std::string &filename) {
  cv::Mat img = cv::imread(filename);

  cv::Mat edge;
  ChamferMatcher::computeCanny(img, edge, 70.0);

  cv::Mat dist_transform, labels;
  ChamferMatcher::computeDistanceTransform(edge, dist_transform, labels);

  cv::Mat edge_orientations_img;
  std::vector<std::vector<cv::Point> > contours;
  std::vector<std::vector<float> > edges_orientation;
  ChamferMatcher::createMapOfEdgeOrientations(img, labels, edge_orientations_img, contours, edges_orientation);

  double min, max;
  cv::minMaxLoc(edge_orientations_img, &min, &max);

  cv::Mat edge_orientations_colormap(edge_orientations_img.size(), CV_8UC3);
  for(int i = 0; i < edge_orientations_colormap.rows; i++) {
    for(int j = 0; j < edge_orientations_colormap.cols; j++) {
      float value = edge_orientations_img.at<float>(i,j) * 255.0/(max-min) - 255.0*min/(max-min);
      edge_orientations_colormap.at<cv::Vec3b>(i,j) = cv::Vec3b(value, 255, 255);
      std::cout << edge_orientations_img.at<float>(i,j) << " ; ";
    }
    std::cout << std::endl;
  }

  cv::Mat edge_orientations_colormap_display;
  cv::cvtColor(edge_orientations_colormap, edge_orientations_colormap_display, cv::COLOR_HSV2BGR);


  //Display edge orientation line
  cv::Mat edge_orientation_line_img;
  img.copyTo(edge_orientation_line_img);

  int length = 20;
  for(size_t i = 0; i < contours.size(); i++) {
    for(size_t j = 0; j < contours[i].size(); j+=10) {
      cv::Point start_point = contours[i][j];

      float angle1 = edges_orientation[i][j];
      int x1 = cos(angle1)*length;
      int y1 = sin(angle1)*length;
      cv::Point end_point1 = start_point + cv::Point(x1, y1);

      float angle2 = angle1 + M_PI;
      int x2 = cos(angle2)*length;
      int y2 = sin(angle2)*length;
      cv::Point end_point2 = start_point + cv::Point(x2, y2);

      cv::line(edge_orientation_line_img, start_point, end_point1, cv::Scalar(255,0,0));
      cv::line(edge_orientation_line_img, start_point, end_point2, cv::Scalar(255,0,0));
    }
  }


  cv::imshow("img", img);
  cv::imshow("edge_orientations_colormap_display", edge_orientations_colormap_display);
  cv::imshow("edge_orientation_line_img", edge_orientation_line_img);
  cv::waitKey(0);
}

int main() {
  testContoursOrientation(DATA_LOCATION_PREFIX + "Inria_logo_template.jpg");
//  testContoursOrientation(DATA_LOCATION_PREFIX + "Template_triangle.png");
//  testContoursOrientation(DATA_LOCATION_PREFIX + "Template_rectangle.png");
//  testContoursOrientation(DATA_LOCATION_PREFIX + "Template_circle.png");

//  checkContoursIndex(DATA_LOCATION_PREFIX + "Template_circle.png");
//  checkContoursIndex(DATA_LOCATION_PREFIX + "Inria_logo_template.jpg");

  return 0;
}
