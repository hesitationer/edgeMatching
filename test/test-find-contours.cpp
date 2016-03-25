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


void testContours(const std::string &filename) {
  cv::Mat img = cv::imread(filename);

  std::vector<std::vector<cv::Point> > contours;
  ChamferMatcher::getContours(img, contours);
  cv::Mat displayFindContours = cv::Mat::zeros(img.size(), CV_32F);
  for(int i = 0; i < contours.size(); i++) {
    for(int j = 0; j < contours[i].size(); j++) {
      displayFindContours.at<float>(contours[i][j].y, contours[i][j].x) = j;
    }
  }
  std::cout << "displayFindContours=\n" << displayFindContours << std::endl;

  std::vector<std::vector<float> > edges_orientations;
  ChamferMatcher::getContoursOrientation(contours, edges_orientations);
  cv::Mat displayContoursOrientation = cv::Mat::zeros(img.size(), CV_32F);
  for(int i = 0; i < edges_orientations.size(); i++) {
    for(int j = 0; j < edges_orientations[i].size(); j++) {
      displayContoursOrientation.at<float>(contours[i][j].y, contours[i][j].x) = edges_orientations[i][j];
    }
  }
  std::cout << "\n\ndisplayContoursOrientation=\n" << displayContoursOrientation << std::endl;
}

int main() {
  testContours(DATA_LOCATION_PREFIX + "Inria_logo_template.jpg");
//  testContours(DATA_LOCATION_PREFIX + "Template_triangle.png");

  return 0;
}
