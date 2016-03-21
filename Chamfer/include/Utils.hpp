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
#ifndef __Utils_h__
#define __Utils_h__

#include "Chamfer.hpp"


float atan2_approximation2(float y, float x);

void convertToBoundingBox(const std::vector<Detection_t> &detections, std::vector<cv::Rect> &boundingBoxes);

double fastAcos(const double x);

double fastCosine(const float angle);

double fastSine(const float angle);

/*
 * Compute the angle between two points.
 */
float getAngle(const cv::Point &prev, const cv::Point &next);

bool getLineEquation(const cv::Point &pt1, const cv::Point &pt2, double &a, double &b);

void getPolarLineEquation(const double a, const double b, double &theta, double &rho);
void getPolarLineEquation(const cv::Point &pt1, const cv::Point &pt2, double &theta, double &rho);
void getPolarLineEquation(const cv::Point &pt1, const cv::Point &pt2, double &theta, double &rho, double &length);

float getMinAngleError(const float angle1, const float angle2, const bool fast);
float getMinAngleError(const float angle1, const float angle2, const bool degree, const bool customAngle);

static cv::Scalar randomColor(cv::RNG& rng) {
  int icolor = (unsigned) rng;
  return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

#endif
