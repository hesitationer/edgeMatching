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


int main() {
  cv::Mat img_template = cv::imread(DATA_LOCATION_PREFIX + "Inria_logo_template.jpg");
//  cv::Mat img_query = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene.jpg");
//  cv::Mat img_query = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene2.jpg");
//  cv::Mat img_query = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene3.jpg");
//  cv::Mat img_query = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene4.jpg");
  cv::Mat img_query = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene5.jpg");


  std::map<int, cv::Mat> mapOfTemplates;
  std::map<int, std::pair<cv::Rect, cv::Rect> > mapOfTemplateRois;
  mapOfTemplates[1] = img_template;
  mapOfTemplateRois[1] = std::pair<cv::Rect, cv::Rect>(cv::Rect(0,0,-1,-1), cv::Rect(0,0,-1,-1));

  ChamferMatcher chamfer(mapOfTemplates, mapOfTemplateRois);
  std::vector<Detection_t> detections;
  bool useOrientation = true;
  float distanceThreshold = 100.0, lambda = 100.0f;
  float weight_forward = 1.0f, weight_backward = 1.0f;
  bool useNonMaximaSuppression = true, useGroupDetections = true;

  chamfer.setCannyThreshold(70.0);
  chamfer.setMatchingType(ChamferMatcher::edgeMatching);
//  chamfer.setMatchingType(ChamferMatcher::edgeForwardBackwardMatching);
//  chamfer.setMatchingType(ChamferMatcher::fullMatching);
//  chamfer.setMatchingType(ChamferMatcher::lineMatching);
//  chamfer.setMatchingType(ChamferMatcher::lineForwardBackwardMatching);

  double t = (double) cv::getTickCount();
//  chamfer.detect(img_query, detections, useOrientation, distanceThreshold, lambda, weight_forward,
//  		weight_backward, useGroupDetections);
  chamfer.detectMultiScale(img_query, detections, useOrientation, distanceThreshold, lambda, weight_forward,
  		weight_backward, useNonMaximaSuppression, useGroupDetections);
  t = ((double) cv::getTickCount() - t) / cv::getTickFrequency() * 1000.0;
  std::cout << "Processing time=" << t << " ms" << std::endl;

  cv::Mat result;
  img_query.convertTo(result, CV_8UC3);

  std::cout << "detections=" << detections.size() << std::endl;
  for(std::vector<Detection_t>::const_iterator it = detections.begin(); it != detections.end(); ++it) {
    cv::rectangle(result, it->m_boundingBox, cv::Scalar(0,0,255), 2);

    std::stringstream ss;
    //Chamfer distance
    ss << it->m_chamferDist;
    cv::Point ptText = it->m_boundingBox.tl() + cv::Point(10, 20);
    cv::putText(result, ss.str(), ptText, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,0,0), 2);

    //Scale
    ss.str("");
    ss << it->m_scale;
    ptText = it->m_boundingBox.tl() + cv::Point(10, 40);
    cv::putText(result, ss.str(), ptText, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,0,0), 2);

    cv::imshow("result", result);
    cv::waitKey(0);
  }

  //XXX:
//  cv::imwrite("Simple_test_result_single_scale.png", result);

  cv::waitKey(0);
  return 0;
}
