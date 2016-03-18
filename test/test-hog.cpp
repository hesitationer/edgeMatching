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
#include <limits>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

std::string DATA_LOCATION_PREFIX = DATA_DIR;


// http://www.juergenwiki.de/work/wiki/doku.php?id=public%3ahog_descriptor_computation_and_visualization
// HOGDescriptor visual_imagealizer
// adapted for arbitrary size of feature sets and training images
cv::Mat get_hogdescriptor_visual_image(cv::Mat& origImg,
    std::vector<float>& descriptorValues, cv::Size winSize, cv::Size cellSize,
    int scaleFactor, double viz_factor) {
  cv::Mat visual_image;
  cv::resize(origImg, visual_image,
      cv::Size(origImg.cols * scaleFactor, origImg.rows * scaleFactor));

  int gradientBinSize = 9;
  // dividing 180° into 9 bins, how large (in rad) is one bin?
  float radRangeForOneBin = 3.14 / (float) gradientBinSize;

  // prepare data structure: 9 orientation / gradient strengths for each cell
  int cells_in_x_dir = winSize.width / cellSize.width;
  int cells_in_y_dir = winSize.height / cellSize.height;
  int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
  float*** gradientStrengths = new float**[cells_in_y_dir];
  int** cellUpdateCounter = new int*[cells_in_y_dir];
  for (int y = 0; y < cells_in_y_dir; y++) {
    gradientStrengths[y] = new float*[cells_in_x_dir];
    cellUpdateCounter[y] = new int[cells_in_x_dir];
    for (int x = 0; x < cells_in_x_dir; x++) {
      gradientStrengths[y][x] = new float[gradientBinSize];
      cellUpdateCounter[y][x] = 0;

      for (int bin = 0; bin < gradientBinSize; bin++)
        gradientStrengths[y][x][bin] = 0.0;
    }
  }

  // nr of blocks = nr of cells - 1
  // since there is a new block on each cell (overlapping blocks!) but the last one
  int blocks_in_x_dir = cells_in_x_dir - 1;
  int blocks_in_y_dir = cells_in_y_dir - 1;

  // compute gradient strengths per cell
  int descriptorDataIdx = 0;
  int cellx = 0;
  int celly = 0;

  for (int blockx = 0; blockx < blocks_in_x_dir; blockx++) {
    for (int blocky = 0; blocky < blocks_in_y_dir; blocky++) {
      // 4 cells per block ...
      for (int cellNr = 0; cellNr < 4; cellNr++) {
        // compute corresponding cell nr
        int cellx = blockx;
        int celly = blocky;
        if (cellNr == 1)
          celly++;
        if (cellNr == 2)
          cellx++;
        if (cellNr == 3) {
          cellx++;
          celly++;
        }

        for (int bin = 0; bin < gradientBinSize; bin++) {
          float gradientStrength = descriptorValues[descriptorDataIdx];
          descriptorDataIdx++;

          gradientStrengths[celly][cellx][bin] += gradientStrength;

        } // for (all bins)

        // note: overlapping blocks lead to multiple updates of this sum!
        // we therefore keep track how often a cell was updated,
        // to compute average gradient strengths
        cellUpdateCounter[celly][cellx]++;

      } // for (all cells)

    } // for (all block x pos)
  } // for (all block y pos)

  // compute average gradient strengths
  for (int celly = 0; celly < cells_in_y_dir; celly++) {
    for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

      float NrUpdatesForThisCell = (float) cellUpdateCounter[celly][cellx];

      // compute average gradient strenghts for each gradient bin direction
      for (int bin = 0; bin < gradientBinSize; bin++) {
        gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
      }
    }
  }

  std::cout << "descriptorDataIdx = " << descriptorDataIdx << std::endl;

  // draw cells
  for (int celly = 0; celly < cells_in_y_dir; celly++) {
    for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
      int drawX = cellx * cellSize.width;
      int drawY = celly * cellSize.height;

      int mx = drawX + cellSize.width / 2;
      int my = drawY + cellSize.height / 2;

      cv::rectangle(visual_image,
          cv::Point(drawX * scaleFactor, drawY * scaleFactor),
          cv::Point((drawX + cellSize.width) * scaleFactor,
              (drawY + cellSize.height) * scaleFactor), CV_RGB(100, 100, 100),
          1);

      // draw in each cell all 9 gradient strengths
      for (int bin = 0; bin < gradientBinSize; bin++) {
        float currentGradStrength = gradientStrengths[celly][cellx][bin];

        // no line to draw?
        if (currentGradStrength == 0)
          continue;

        float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

        float dirVecX = cos(currRad);
        float dirVecY = sin(currRad);
        float maxVecLen = cellSize.width / 2;
        float scale = viz_factor; // just a visual_imagealization scale,
                                  // to see the lines better

        // compute line coordinates
        float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
        float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
        float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
        float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

        // draw gradient visual_imagealization
        cv::line(visual_image, cv::Point(x1 * scaleFactor, y1 * scaleFactor),
            cv::Point(x2 * scaleFactor, y2 * scaleFactor), CV_RGB(0, 0, 255),
            1);

      } // for (all bins)

    } // for (cellx)
  } // for (celly)

  // don't forget to free memory allocated by helper data structures!
  for (int y = 0; y < cells_in_y_dir; y++) {
    for (int x = 0; x < cells_in_x_dir; x++) {
      delete[] gradientStrengths[y][x];
    }
    delete[] gradientStrengths[y];
    delete[] cellUpdateCounter[y];
  }
  delete[] gradientStrengths;
  delete[] cellUpdateCounter;

  return visual_image;
}

void computeHOG(const cv::Mat &img_color) {
  cv::HOGDescriptor d;
  // Size(64,128), //winSize
  // Size(16,16), //blocksize
  // Size(8,8), //blockStride,
  // Size(8,8), //cellSize,
  // 9, //nbins,
  // 0, //derivAper,
  // -1, //winSigma,
  // 0, //histogramNormType,
  // 0.2, //L2HysThresh,
  // 0 //gammal correction,
  // //nlevels=64
  //);

  // void HOGDescriptor::compute(const Mat& img, vector<float>& descriptors,
  //                             Size winStride, Size padding,
  //                             const vector<Point>& locations) const

  int correct_width = img_color.cols - (img_color.cols - d.blockSize.width) % d.blockStride.width;
  int correct_height = img_color.rows - (img_color.rows - d.blockSize.height) % d.blockStride.height;
  d.winSize = cv::Size(correct_width, correct_height);

  //Scale input image to the correct size
  cv::Mat resize_img, resize_img_color;
  cv::resize(img_color, resize_img_color, d.winSize);
  cv::cvtColor(resize_img_color, resize_img, cv::COLOR_BGR2GRAY);

  std::cout << "d: winSize=" << d.winSize << " ; blocksize=" << d.blockSize << " ; blockStride="
      << d.blockStride << " ; cellSize=" << d.cellSize << std::endl;

  std::vector<float> descriptorsValues;
  std::vector<cv::Point> locations;
  d.compute(resize_img, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations);

  std::cout << "HOG descriptor size is " << d.getDescriptorSize() << std::endl;
  std::cout << "img dimensions: " << resize_img.cols << " width x " << resize_img.rows << "height " << std::endl;
  std::cout << "Found " << descriptorsValues.size() << " descriptor values" << std::endl;
  std::cout << "Nr of locations specified : " << locations.size() << std::endl;

  //Display HOG
  cv::Mat display_HOG = get_hogdescriptor_visual_image(resize_img_color, descriptorsValues, d.winSize, d.cellSize, 1, 5.0);

  cv::imshow("display_HOG", display_HOG);
  cv::waitKey(0);
}

cv::Mat computeHOG(const cv::Mat &img_gray, cv::HOGDescriptor &d) {
  int correct_width = img_gray.cols - (img_gray.cols - d.blockSize.width) % d.blockStride.width;
  int correct_height = img_gray.rows - (img_gray.rows - d.blockSize.height) % d.blockStride.height;
  d.winSize = cv::Size(correct_width, correct_height);

  //Scale input image to the correct size
  cv::Mat resize_img;
  cv::resize(img_gray, resize_img, d.winSize);

  std::vector<float> descriptorsValues;
  std::vector<cv::Point> locations;
  d.compute(resize_img, descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations);

  cv::Mat matDescriptorsValues(descriptorsValues, true);
  return matDescriptorsValues;
}

void detectHOG_impl(const cv::Mat &template_img, const cv::Mat &query_img, std::vector<cv::Rect> &detections,
    const int offsetX=10, const int offsetY=10) {
  int maxWidth = query_img.cols-template_img.cols;
  int maxHeight = query_img.rows-template_img.rows;
  cv::Mat matching_cost_map(maxHeight, maxWidth, CV_32F, std::numeric_limits<float>::max());

  cv::Mat template_img_gray, query_img_gray;
  cv::cvtColor(template_img, template_img_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(query_img, query_img_gray, cv::COLOR_BGR2GRAY);

  cv::HOGDescriptor hog_descriptor;
  cv::Mat template_descriptors = computeHOG(template_img_gray, hog_descriptor);

  for(int i = 0; i < matching_cost_map.rows; i += offsetY) {
    for(int j = 0; j < matching_cost_map.cols; j += offsetX) {
      cv::Mat sub_image = query_img_gray(cv::Rect(j, i, template_img.cols, template_img.rows));

      cv::Mat query_descriptors = computeHOG(sub_image, hog_descriptor);
      double dist = cv::norm(template_descriptors, query_descriptors, cv::NORM_L2);

      matching_cost_map.at<float>(i,j) = (float) dist;
    }
  }


  cv::Mat matching_cost_map_display;
  double minVal, maxVal;
  cv::Point minLoc, maxLoc;
  cv::minMaxLoc(matching_cost_map, &minVal, &maxVal, &minLoc, &maxLoc);
  matching_cost_map.convertTo(matching_cost_map_display, CV_8U, 255.0/(maxVal-minVal), -255.0*minVal/(maxVal-minVal));

  detections.push_back(cv::Rect(minLoc.x, minLoc.y, template_img.cols, template_img.rows));

  cv::imshow("matching_cost_map_display", matching_cost_map_display);
//  cv::waitKey(0);
}

void detectHOG(const cv::Mat &template_img, const cv::Mat &query_img, std::vector<cv::Rect> &detections) {
  detectHOG_impl(template_img, query_img, detections);
}

int main() {
//  computeHOG(cv::imread(DATA_LOCATION_PREFIX + "Inria_logo_template.jpg"));
//  computeHOG(cv::imread(DATA_LOCATION_PREFIX + "Template_rectangle.png"));
//  computeHOG(cv::imread(DATA_LOCATION_PREFIX + "Inria_scene3.jpg"));

  cv::Mat template_img = cv::imread(DATA_LOCATION_PREFIX + "Inria_logo_template.jpg");
  cv::Mat query_img = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene.jpg");
//  cv::Mat query_img = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene2.jpg");
//  cv::Mat query_img = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene3.jpg");

  std::vector<cv::Rect> detections;
  detectHOG(template_img, query_img, detections);

  cv::Mat result;
  query_img.copyTo(result);

  for(size_t i = 0; i < detections.size(); i++) {
    cv::rectangle(result, detections[i], cv::Scalar(0,0,255), 2);
  }

  cv::imshow("result", result);
  cv::waitKey(0);

  return 0;
}
