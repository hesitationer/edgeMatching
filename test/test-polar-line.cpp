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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


void testPolarLineEquation() {
  cv::RNG rng(12345);

  cv::Mat img = cv::Mat::zeros(600, 800, CV_8UC3);

  cv::Point center(img.cols/2, img.rows/2);
  float length = 50;
  float dist_text1 = 180, dist_text2 = 270;
  for(int angle = 0, cpt = 0; angle < 360; angle += 5, cpt++) {
    int x2 = cos(angle*M_PI/180.0)*length;
    int y2 = sin(angle*M_PI/180.0)*length;
    cv::Point extremity = center + cv::Point(x2, y2);

    cv::Scalar color = randomColor(rng);
    cv::line(img, center, extremity, color, 1);

    double theta = 0.0, rho = 0.0;
    getPolarLineEquation(center, extremity, theta, rho);

    std::cout << "Angle=" << angle << " ; theta=" << (theta*180.0/M_PI) << std::endl;

    cv::Point posText;
    if(cpt % 2 == 0) {
      posText = cv::Point(cos(angle*M_PI/180.0)*dist_text1, sin(angle*M_PI/180.0)*dist_text1);
    } else {
      posText = cv::Point(cos(angle*M_PI/180.0)*(dist_text1+30), sin(angle*M_PI/180.0)*(dist_text1+30));
    }

    std::stringstream ss;
    int theta_i = theta*180.0/M_PI;
    ss << theta_i;
    cv::putText(img, ss.str(), center+posText, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);

    cv::Point posText2;
    if(cpt % 2 == 0) {
      posText2 = cv::Point(cos(angle*M_PI/180.0)*dist_text2, sin(angle*M_PI/180.0)*dist_text2);
    } else {
      posText2 = cv::Point(cos(angle*M_PI/180.0)*(dist_text2+20), sin(angle*M_PI/180.0)*(dist_text2+20));
    }

    ss.str("");
    ss << angle;
    cv::putText(img, ss.str(), center+posText2, cv::FONT_HERSHEY_SIMPLEX, 0.5, color /*cv::Scalar(0,255,0)*/);
  }

  cv::imshow("Correspondence angle <==> polar theta", img);
  cv::waitKey(0);
}

int main() {
  testPolarLineEquation();

  return 0;
}
