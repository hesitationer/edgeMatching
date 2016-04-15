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
#include "../include/HOGDetector.hpp"
#include <limits>
#include <opencv2/imgproc/imgproc.hpp>

using namespace hog;


HOGDetector::HOGDetector() : m_mapOfTemplateInfo(), m_useSpatialRejection(true) {
}

//@url=https://avresearch.wordpress.com/2011/08/05/integral-histogram-for-fast-hog-feature-calculation/
/*Function to calculate the integral histogram*/
std::vector<cv::Mat> HOGDetector::calculateIntegralHOG(const cv::Mat &_in, const int _nbins) {
  /*Convert the input image to grayscale*/
//  cv::Mat img_gray;

//  cv::cvtColor(_in, img_gray, CV_BGR2GRAY);
//  cv::equalizeHist(img_gray, img_gray);

  /* Calculate the derivates of the grayscale image in the x and y
   directions using a sobel operator and obtain 2 gradient images
   for the x and y directions*/

  cv::Mat xsobel, ysobel;
  cv::Sobel(_in, xsobel, CV_32FC1, 1, 0);
  cv::Sobel(_in, ysobel, CV_32FC1, 0, 1);

  //Gradient magnitude
  cv::Mat gradient_magnitude;
  cv::magnitude(xsobel, ysobel, gradient_magnitude);
  //Gradient orientation
  cv::Mat gradient_orientation;
  bool angleInDegrees = true;
  cv::phase(xsobel, ysobel, gradient_orientation, angleInDegrees);

//  std::cout << "_in=" << _in.rows << "x" << _in.cols << std::endl;
//  std::cout << "gradient_magnitude=" << gradient_magnitude.rows << "x" << gradient_magnitude.cols << std::endl;
//  std::cout << "gradient_orientation=" << gradient_orientation.rows << "x" << gradient_orientation.cols << std::endl;
//
//  double min_angle, max_angle;
//  cv::minMaxLoc(gradient_orientation, &min_angle, &max_angle, NULL, NULL);
//  std::cout << "min_angle=" << min_angle << " ; max_angle=" << max_angle << std::endl;



  /* Create an array of 9 images (9 because I assume bin size 20 degrees
   and unsigned gradient ( 180/20 = 9), one for each bin which will have
   zeroes for all pixels, except for the pixels in the original image
   for which the gradient values correspond to the particular bin.
   These will be referred to as bin images. These bin images will be then
   used to calculate the integral histogram, which will quicken
   the calculation of HOG descriptors */

  std::vector<cv::Mat> bins(_nbins);
  for (int i = 0; i < _nbins; i++) {
    bins[i] = cv::Mat::zeros(_in.size(), CV_32FC1);
  }

  /* Create an array of 9 images ( note the dimensions of the image,
   the cvIntegral() function requires the size to be that), to store
   the integral images calculated from the above bin images.
   These 9 integral images together constitute the integral histogram */

  std::vector<cv::Mat> integrals(_nbins);
  //IplImage** integrals = (IplImage**) malloc(9 * sizeof(IplImage*));
  for (int i = 0; i < _nbins; i++) {
    integrals[i] = cv::Mat(
        cv::Size(_in.size().width + 1, _in.size().height + 1), CV_64FC1);
  }

  /* Calculate the bin images. The magnitude and orientation of the gradient
   at each pixel is calculated using the xsobel and ysobel images.
   {Magnitude = sqrt(sq(xsobel) + sq(ysobel) ), gradient = itan (ysobel/xsobel) }.
   Then according to the orientation of the gradient, the value of the
   corresponding pixel in the corresponding image is set */

  int x, y;
  float temp_gradient, temp_magnitude;
  for (y = 0; y < _in.size().height; y++) {
    /* ptr1 and ptr2 point to beginning of the current row in the xsobel and ysobel images
     respectively.
     ptrs[i] point to the beginning of the current rows in the bin images */

#if 0
    float* xsobelRowPtr = (float*) (xsobel.row(y).data);
    float* ysobelRowPtr = (float*) (ysobel.row(y).data);
    float** binsRowPtrs = new float *[_nbins];

    for (int i = 0; i < _nbins; i++) {
      binsRowPtrs[i] = (float*) (bins[i].row(y).data);
    }
#else
    float* xsobelRowPtr = xsobel.ptr<float>(y);
    float* ysobelRowPtr = ysobel.ptr<float>(y);
    float** binsRowPtrs = new float *[_nbins];

    for (int i = 0; i < _nbins; i++) {
      binsRowPtrs[i] = bins[i].ptr<float>(y);
    }

    float* magnitudeRowPtr = gradient_magnitude.ptr<float>(y);
    float* orientationRowPtr = gradient_orientation.ptr<float>(y);
#endif

    /*For every pixel in a row gradient orientation and magnitude
     are calculated and corresponding values set for the bin images. */
    for (x = 0; x < _in.size().width; x++) {
#if 0
      /* if the xsobel derivative is zero for a pixel, a small value is
       added to it, to avoid division by zero. atan returns values in radians,
       which on being converted to degrees, correspond to values between -90 and 90 degrees.
       90 is added to each orientation, to shift the orientation values range from {-90-90} to {0-180}.
       This is just a matter of convention. {-90-90} values can also be used for the calculation. */
      if (xsobelRowPtr[x] == 0) {
        temp_gradient = ((atan(ysobelRowPtr[x] / (xsobelRowPtr[x] + 0.00001)))
            * (180 / M_PI)) + 90;
      } else {
        temp_gradient = ((atan(ysobelRowPtr[x] / xsobelRowPtr[x])) * (180 / M_PI))
            + 90;
      }
      temp_magnitude = sqrt(
          (xsobelRowPtr[x] * xsobelRowPtr[x])
              + (ysobelRowPtr[x] * ysobelRowPtr[x]));
#else
      temp_magnitude = magnitudeRowPtr[x];
      temp_gradient = orientationRowPtr[x] > 180 ? orientationRowPtr[x]-180 : orientationRowPtr[x];
#endif

      /*The bin image is selected according to the gradient values.
       The corresponding pixel value is made equal to the gradient
       magnitude at that pixel in the corresponding bin image */
      float binStep = 180 / _nbins;

      for (int i = 1; i <= _nbins; i++) {
        if (temp_gradient <= binStep * i) {
          binsRowPtrs[i - 1][x] = temp_magnitude;
          break;
        }
      }
    }

    //Delete binsRowPtrs
    delete[] binsRowPtrs;
  }


  /*Integral images for each of the bin images are calculated*/
  for (int i = 0; i < _nbins; i++) {
    cv::integral(bins[i], integrals[i]);
  }


  /*The function returns an array of 9 images which constitute the integral histogram*/
  return integrals;
}

/*The following demonstrates how the integral histogram calculated using
 the above function can be used to calculate the histogram of oriented
 gradients for any rectangular region in the image:*/

/* The following function takes as input the rectangular cell for which the
 histogram of oriented gradients has to be calculated, a matrix hog_cell
 of dimensions 1x9 to store the bin values for the histogram, the integral histogram,
 and the normalization scheme to be used. No normalization is done if normalization = -1 */

void HOGDetector::calculateHOG_rect(cv::Mat& hogCell, std::vector<cv::Mat> integrals,
    cv::Rect global_roi, int nbins, int nbCellX, int nbCellY/*, int _normalization*/) {
	if(hogCell.empty()) {
		hogCell = cv::Mat(nbCellX*nbCellY*nbins, 1, CV_32F);
	}

//  if (_roi.width == 0 || _roi.height == 0) {
//    _roi.x = 0;
//    _roi.y = 0;
//    _roi.width = _integrals[0].size().width - 1;
//    _roi.height = _integrals[0].size().height - 1;
//  }

  /* Calculate the bin values for each of the bin of the histogram one by one */
#if 0
  for (int i = 0; i < _nbins; i++) {
    std::cout << "_integrals=" << _integrals[i].type() << " ; CV_32F=" << CV_32F << std::endl;

    IplImage intImgIpl = _integrals[i];

    float a =
        ((double*) (intImgIpl.imageData + (_roi.y) * (intImgIpl.widthStep)))[_roi.x];
    float b = ((double*) (intImgIpl.imageData
        + (_roi.y + _roi.height) * (intImgIpl.widthStep)))[_roi.x + _roi.width];
    float c =
        ((double*) (intImgIpl.imageData + (_roi.y) * (intImgIpl.widthStep)))[_roi.x
            + _roi.width];
    float d = ((double*) (intImgIpl.imageData
        + (_roi.y + _roi.height) * (intImgIpl.widthStep)))[_roi.x];

    ((float*) _hogCell.data)[i] = (a + b) - (c + d);
  }
#else
  std::vector<cv::Rect> rois;
  int step_x = global_roi.width / nbCellX;
  int step_y = global_roi.height / nbCellY;
  int last_step_x = global_roi.width - step_x*(nbCellX-1);
  int last_step_y = global_roi.height - step_y*(nbCellY-1);

  for(int i = 0; i < nbCellY; i++) {
    for(int j = 0; j < nbCellX; j++) {
      if(i == nbCellY-1 && j == nbCellX-1) {
        rois.push_back( cv::Rect(global_roi.x + j*step_x, global_roi.y + i*step_y, last_step_x, last_step_y) );
      } else if(i == nbCellY-1) {
        rois.push_back( cv::Rect(global_roi.x + j*step_x, global_roi.y + i*step_y, step_x, last_step_y) );
      } else if(j == nbCellX-1) {
        rois.push_back( cv::Rect(global_roi.x + j*step_x, global_roi.y + i*step_y, last_step_x, step_y) );
      } else {
        rois.push_back( cv::Rect(global_roi.x + j*step_x, global_roi.y + i*step_y, step_x, step_y) );
      }
    }
  }

  for(int i = 0; i < nbCellY; i++) {
    for(int j = 0; j < nbCellX; j++) {
      for (int cpt = 0; cpt < nbins; cpt++) {
        cv::Mat intImg = integrals[cpt];
        cv::Rect roi = rois[i*nbCellY+j];

        float a = (float) intImg.ptr<double>(roi.y)[roi.x];
        float b = (float) intImg.ptr<double>(roi.y + roi.height)[roi.x + roi.width];
        float c = (float) intImg.ptr<double>(roi.y)[roi.x + roi.width];
        float d = (float) intImg.ptr<double>(roi.y + roi.height)[roi.x];

        hogCell.ptr<float>( cpt*nbins + i*nbCellY+j )[0] = (a + b) - (c + d);
      }
    }
  }
#endif

#if 0
  /*Normalize the matrix*/
//  if (_normalization != -1) {
    cv::normalize(hogCell, hogCell, 0, 1, CV_MINMAX);
//  }
#else
  cv::Scalar hog_sum = cv::sum(hogCell);
  hogCell /= hog_sum[0];
#endif
}

void HOGDetector::detect_impl(const Template_info_t &template_info, const Query_info_t &query_info, const int scale,
		std::vector<Detection_t> &detections, const double distThresh, const int offsetX, const int offsetY) {
	int template_width = template_info.m_size.width*scale/100;
	int template_height = template_info.m_size.height*scale/100;

  int maxWidth = query_info.m_size.width - template_width;
  int maxHeight = query_info.m_size.height - template_height;
  if(maxWidth <= 0 || maxHeight <= 0) {
    return;
  }

  cv::Mat matching_cost_map(maxHeight, maxWidth, CV_32F, std::numeric_limits<float>::max());

  cv::Rect query_roi(0, 0, template_width, template_height);
  int nbCellX = 3, nbCellY = 3, nbins = 9;
  cv::Mat query_hog(nbCellX*nbCellY*nbins, 1, CV_32F);


  //Fast rejection
  int rejectionOffsetY = template_height / 2;
  int rejectionOffsetX = template_width / 2;

  int rejection_height = matching_cost_map.rows / rejectionOffsetY + 1;
  int rejection_width = matching_cost_map.cols / rejectionOffsetX + 1;
  cv::Mat rejection_mask = cv::Mat::ones(rejection_height, rejection_width, CV_8U);

  //Compute the detection at each size(template) / 2
  for(int i = 0, indexI = 0; i < matching_cost_map.rows; i += rejectionOffsetY, indexI++) {
  	uchar *ptr_row_rejection = rejection_mask.ptr<uchar>(indexI);

    for(int j = 0, indexJ = 0; j < matching_cost_map.cols; j += rejectionOffsetX, indexJ++) {
    	query_roi.x = j;
    	query_roi.y = i;

      calculateHOG_rect(query_hog, query_info.m_integralHOG, query_roi, nbins, nbCellX, nbCellY);
      double dist = cv::compareHist(template_info.m_hog, query_hog, CV_COMP_BHATTACHARYYA);

      if(isnan(dist) || dist > 2*distThresh) {
      	ptr_row_rejection[indexJ] = 0;
      }
    }
  }


  int indexI = 0;
#pragma omp parallel for private(query_hog, indexI)
  for(int i = 0; i < matching_cost_map.rows; i += offsetY) {
  	float *ptr_row = matching_cost_map.ptr<float>(i);
  	const uchar *ptr_row_rejection = rejection_mask.ptr<uchar>(indexI);

    for(int j = 0, indexJ=0; j < matching_cost_map.cols; j += offsetX) {
    	if(ptr_row_rejection[indexJ]) {
      	query_roi.x = j;
      	query_roi.y = i;

        calculateHOG_rect(query_hog, query_info.m_integralHOG, query_roi, nbins, nbCellX, nbCellY);
        double dist = cv::compareHist(template_info.m_hog, query_hog, CV_COMP_BHATTACHARYYA);

        if(!isnan(dist)) {
          ptr_row[j] = (float) dist;
        }
    	}

      if( j >= (indexJ+1)*rejection_width ) {
      	indexJ++;
      }
    }

    if( i >= (indexI+1)*rejection_height ) {
    	indexI++;
    }
  }


  double minVal, maxVal;
  cv::Point minLoc, maxLoc;
  cv::minMaxLoc(matching_cost_map, &minVal, &maxVal, &minLoc, &maxLoc);

  do {
    cv::minMaxLoc(matching_cost_map, &minVal, &maxVal, &minLoc, &maxLoc);

    //"Reset the location" to find other detections
    matching_cost_map.ptr<float>(minLoc.y)[minLoc.x] = std::numeric_limits<float>::max();

    if(minVal < distThresh) {
      //Add detection
      Detection_t detection(cv::Rect(minLoc.x, minLoc.y, template_width, template_height), minVal);
      detections.push_back(detection);
    }

    //TODO: fix a maximum number of detections?
    //Get best detection only
    break;
  } while(minVal < distThresh);
}

void HOGDetector::detect(const cv::Mat &query_img, std::vector<Detection_t> &detections, const double distThresh,
		const int offsetX, const int offsetY) {
	cv::Mat query_img_gray;

  if(query_img.channels() != 1) {
    cv::cvtColor(query_img, query_img_gray, cv::COLOR_BGR2GRAY);
  } else {
    query_img_gray = query_img;
  }

  //Compute query info
  std::vector<cv::Mat> query_integralHOG = calculateIntegralHOG(query_img_gray, 9);
  Query_info_t query_info(query_integralHOG, query_img_gray.size());

  //Detect for each template
  int regular_scale = 100;
  for(std::map<int, Template_info_t>::const_iterator it_tpl = m_mapOfTemplateInfo.begin();
  		it_tpl != m_mapOfTemplateInfo.end(); ++it_tpl) {
  	std::vector<Detection_t> current_detections;

  	detect_impl(it_tpl->second, query_info, regular_scale, current_detections, distThresh, offsetX, offsetY);

  	//Set scale
  	for(std::vector<Detection_t>::iterator it_detection = current_detections.begin();
  			it_detection != current_detections.end(); ++it_detection) {
  		it_detection->m_scale = regular_scale;
  		it_detection->m_templateIndex = it_tpl->first;
  	}

  	//Append current detections
  	detections.insert(detections.end(), current_detections.begin(), current_detections.end());
  }

  //Sort detections
  std::sort(detections.begin(), detections.end());
}

void HOGDetector::detectMultiScale(const cv::Mat &query_img, std::vector<Detection_t> &detections, const double distThresh,
  		const int offsetX, const int offsetY, const int minScale, const int maxScale, const int scaleStep) {
	if(minScale <= 0 || maxScale <= 0 || scaleStep <= 0 || maxScale < minScale) {
		std::cerr << "Problem with scale parameters!" << std::endl;
		return;
	}

	cv::Mat query_img_gray;

  if(query_img.channels() != 1) {
    cv::cvtColor(query_img, query_img_gray, cv::COLOR_BGR2GRAY);
  } else {
    query_img_gray = query_img;
  }

  //Compute query info
  std::vector<cv::Mat> query_integralHOG = calculateIntegralHOG(query_img_gray, 9);
  Query_info_t query_info(query_integralHOG, query_img_gray.size());


  //Detect for each template
  for(std::map<int, Template_info_t>::const_iterator it_tpl = m_mapOfTemplateInfo.begin();
  		it_tpl != m_mapOfTemplateInfo.end(); ++it_tpl) {

#pragma omp parallel for
  	for(int scale = minScale; scale <= maxScale; scale += scaleStep) {
    	std::vector<Detection_t> current_detections;

    	detect_impl(it_tpl->second, query_info, scale, current_detections, distThresh, offsetX, offsetY);

    	//Set scale
    	for(std::vector<Detection_t>::iterator it_detection = current_detections.begin();
    			it_detection != current_detections.end(); ++it_detection) {
    		it_detection->m_scale = scale;
    		it_detection->m_templateIndex = it_tpl->first;
    	}

#pragma omp critical
    	{
				//Append current detections
				detections.insert(detections.end(), current_detections.begin(), current_detections.end());
			}
  	}
  }

  //Sort detections
  std::sort(detections.begin(), detections.end());
}

// http://www.juergenwiki.de/work/wiki/doku.php?id=public%3ahog_descriptor_computation_and_visualization
// HOGDescriptor visual_imagealizer
// adapted for arbitrary size of feature sets and training images
cv::Mat HOGDetector::get_hogdescriptor_visual_image(cv::Mat& origImg,
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

void HOGDetector::setTemplateImages(const std::map<int, cv::Mat> &mapOfTemplateImages) {
	m_mapOfTemplateInfo.clear();

	for(std::map<int, cv::Mat>::const_iterator it = mapOfTemplateImages.begin(); it != mapOfTemplateImages.end(); ++it) {
		cv::Mat template_img_gray;
		if(template_img_gray.channels() != 1) {
			cv::cvtColor(it->second, template_img_gray, cv::COLOR_BGR2GRAY);
		} else {
			template_img_gray = it->second;
		}

	  std::vector<cv::Mat> integralHOG = calculateIntegralHOG(template_img_gray, 9);
    cv::Rect template_roi(0, 0, template_img_gray.cols, template_img_gray.rows);
	  cv::Mat template_hog;
    calculateHOG_rect(template_hog, integralHOG, template_roi, 9, 3, 3);

	  Template_info_t template_info(template_hog, integralHOG, template_img_gray.size());
	  m_mapOfTemplateInfo[it->first] = template_info;
	}
}
