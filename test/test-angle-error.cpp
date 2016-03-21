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
#include <ctime>        // std::time
#include <algorithm>    // std::random_shuffle
#include <opencv2/highgui/highgui.hpp>

std::string DATA_LOCATION_PREFIX = DATA_DIR;


void testAngleError(const int nbIterations, const bool bench=false) {
  cv::RNG rng(12345);

  //Create vector of angle
  std::vector<float> vectorOfAngles;
  for(float angle = 0.0f; angle <=360.0; angle+=0.1f) {
    vectorOfAngles.push_back( angle*M_PI/180.0 );
  }

  int length = 200, dist_text = 250;
  for(int cpt1 = 0; cpt1 < nbIterations; cpt1++) {
    cv::Mat img = cv::Mat::zeros(600, 800, CV_8UC3);

    cv::Point startPt(img.cols/2, img.rows/2);

    //Shuffle
    std::random_shuffle ( vectorOfAngles.begin(), vectorOfAngles.end() );

    float angle1 = vectorOfAngles[0];
    float angle2 = vectorOfAngles[1];
    if(!bench) {
      std::cout << "angle1=" << (angle1*180.0/M_PI) << std::endl;
      std::cout << "angle2=" << (angle2*180.0/M_PI) << std::endl;
    }

    cv::Point endPt1 = startPt + cv::Point(cos(angle1)*length, sin(angle1)*length);
    cv::Point endPt2 = startPt + cv::Point(cos(angle2)*length, sin(angle2)*length);

    //Min angle error - method1
    float angle_error1 = getMinAngleError(angle1, angle2, false);
    if(!bench) {
      std::cout << "angle_error1=" << (angle_error1*180.0/M_PI) << std::endl;
    }

    //Polar line equation
    double theta1, rho1, theta2, rho2;
    getPolarLineEquation(startPt, endPt1, theta1, rho1);
    getPolarLineEquation(startPt, endPt2, theta2, rho2);

    float angle_error2 = getMinAngleError(theta1, theta2, false, true);
    if(!bench) {
      std::cout << "angle_error2=" << (angle_error2*180.0/M_PI) << std::endl << std::endl;
    }

    if(!bench) {
      //Draw lines
      cv::line(img, startPt, endPt1, cv::Scalar(255,0,0));
      cv::line(img, startPt, endPt2, cv::Scalar(0,255,0));

      //Draw angle text
      for(int angle = 0, cpt2 = 0; angle < 360; angle += 5, cpt2++) {
        cv::Scalar color = randomColor(rng);

        cv::Point posText;
        if(cpt2 % 2 == 0) {
          posText = cv::Point(cos(angle*M_PI/180.0)*dist_text, sin(angle*M_PI/180.0)*dist_text);
        } else {
          posText = cv::Point(cos(angle*M_PI/180.0)*(dist_text+20), sin(angle*M_PI/180.0)*(dist_text+20));
        }

        std::stringstream ss;
        ss.str("");
        ss << angle;
        cv::putText(img, ss.str(), startPt+posText, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
      }

      cv::imshow("Img", img);
      cv::waitKey(0);
    } else {
      float compare_angle = std::fabs((angle_error2*180.0/M_PI) - (angle_error1*180.0/M_PI));
      if( compare_angle > 1.0f ) {
        std::cerr << "Difference when comparing the two methods to find the minimal angle error !" << std::endl;
        std::cerr << "angle_error1=" << (angle_error1*180.0/M_PI) << " ; angle_error2="
            << (angle_error2*180.0/M_PI) << std::endl;
        return;
      }
    }
  }

  if(bench) {
    std::cout << "Test is OK !" << std::endl;
  }
}

int main() {
  std::srand ( unsigned ( std::time(0) ) );

  testAngleError(10000, true);

  return 0;
}
