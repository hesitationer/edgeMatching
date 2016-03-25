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
#include "../include/Utils.hpp"
#include <limits>


#define PI_FLOAT 3.14159265f
#define PIBY2_FLOAT 1.5707963f
// |error| < 0.005
//https://gist.github.com/volkansalma/2972237
float atan2_approximation2(float y, float x) {
  if (x == 0.0f) {
    if (y > 0.0f)
      return PIBY2_FLOAT;
    if (y == 0.0f)
      return 0.0f;
    return -PIBY2_FLOAT;
  }
  float atan;
  float z = y / x;
  if (fabs(z) < 1.0f) {
    atan = z / (1.0f + 0.28f * z * z);
    if (x < 0.0f) {
      if (y < 0.0f)
        return atan - PI_FLOAT;
      return atan + PI_FLOAT;
    }
  } else {
    atan = PIBY2_FLOAT - z / (z * z + 0.28f);
    if (y < 0.0f)
      return atan - PI_FLOAT;
  }
  return atan;
}

void convertToBoundingBox(const std::vector<Detection_t> &detections, std::vector<cv::Rect> &boundingBoxes) {
  for(std::vector<Detection_t>::const_iterator it = detections.begin(); it != detections.end(); ++it) {
    boundingBoxes.push_back(it->m_boundingBox);
  }
}

double fastAcos(const double x) {
  //http://stackoverflow.com/a/3380723
  return (-0.69813170079773212 * x * x - 0.87266462599716477) * x + 1.5707963267948966;
}

double fastCosine(const float angle) {
  return fastSine(M_PI_2 - angle);
}

double fastSine(const float angle) {
  //http://stackoverflow.com/a/18662397
  double angle2 = angle*angle;
  double angle3 = angle*angle*angle;
  double angle5 = angle3*angle2;
  return angle - (angle3 / 6.0) + (angle5 / 120.0) - (angle5*angle2 / 5040.0);
}

/*
 * Compute the angle between two points.
 */
float getAngle(const cv::Point &prev, const cv::Point &next) {
  int dX = next.x - prev.x;
  int dY = next.y - prev.y;

  return atan2f((float) dY, (float) dX) * 180.0 / M_PI;
}

bool getLineEquation(const cv::Point &pt1, const cv::Point &pt2, double &a, double &b) {
  double den = pt2.x - pt1.x;

  if(fabs(den) < std::numeric_limits<double>::epsilon()) {
    return false;
  } else {
    a = (pt2.y - pt1.y) / den;
    b = pt1.y - a*pt1.x;

    return true;
  }
}

void getPolarLineEquation(const double a, const double b, double &theta, double &rho) {
  theta = atan2(-1.0, a);
//  theta = atan2_approximation2(a, -1.0);

  if(theta < 0) {
    theta += 2.0*M_PI;
  }

  rho = -b / sqrt(a*a + 1);
}

void getPolarLineEquation(const cv::Point &pt1, const cv::Point &pt2, double &theta, double &rho) {
  double a, b;
  theta = M_PI;

  if(getLineEquation(pt1, pt2, a, b)) {
    getPolarLineEquation(a, b, theta, rho);
  } else {
    rho = -b;
  }
}

void getPolarLineEquation(const cv::Point &pt1, const cv::Point &pt2, double &theta, double &rho, double &length) {
  double a, b;
  theta = M_PI;

  if(getLineEquation(pt1, pt2, a, b)) {
    getPolarLineEquation(a, b, theta, rho);
  } else {
    rho = -b;
  }

  length = cv::norm(pt1-pt2);
}

/*
 * Get the minimal angle error between two angles [-PI ; PI].
 */
float getMinAngleError(const float angle1, const float angle2, const bool fast) {
  if(fast) {
    cv::Point2f vec1(fastCosine(angle1), fastSine(angle1));
    vec1 /= norm(vec1);

    cv::Point2f vec2(fastCosine(angle2), fastSine(angle2));
    vec2 /= norm(vec2);

    double angleError1 = fastAcos(vec1.dot(vec2));
    double angleError2 = fastAcos(vec1.dot(-vec2));
    return std::min(angleError1, angleError2);
  } else {
    cv::Point2f vec1(cos(angle1), sin(angle1));
    vec1 /= norm(vec1);

    cv::Point2f vec2(cos(angle2), sin(angle2));
    vec2 /= norm(vec2);

    double angleError1 = acos(vec1.dot(vec2));
    double angleError2 = acos(vec1.dot(-vec2));
    return std::min(angleError1, angleError2);
  }
}

float getMinAngleError(const float angle1, const float angle2, const bool degree, const bool customPolarAngle) {
  if(degree) {
    if(customPolarAngle) {
      return std::min( 180 - fabs(angle1-angle2), fabs(angle1-angle2) );
    } else {
      return std::min( 360 - fabs(angle1-angle2), fabs(angle1-angle2) );
    }
  } else {
    if(customPolarAngle) {
      return std::min( M_PI - fabs(angle1-angle2), fabs(angle1-angle2) );
    } else {
      return std::min( 2.0*M_PI - fabs(angle1-angle2), fabs(angle1-angle2) );
    }
  }
}
