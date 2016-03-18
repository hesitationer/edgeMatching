# Chamfer-Matching (work in progress)

### Goals:
- Detect a template image in a query image using only edge information:
  - [x] detection at single scale.
  - [x] detection at multiple scales.
  - [ ] detection at multiple scales and with rotation.
  - [ ] multiple detections at multiple scales and with rotation, highly cluttered background.
- Detect in a query image the most probable template image and retrieve the corresponding pose (from the same object, multiple template images at different orientations are saved with the corresponding object pose)
  - [ ] detection at single scale.
  - [ ] detection at multiple scales.
  - [ ] detection at multiple scales and with rotation.
  - [ ] multiple detections at multiple scales and with rotation, highly cluttered background.

## References (non exhaustive):
* [Chamfer Matching (contour based shape matching)] - Course from University of Pennsylvania by Nicu Știurcă.
* [Fast Directional Chamfer Matching]
* [Fast Detection of Multiple Textureless 3-D Objects]
* [Contour-Based Learning for Object Detection]

## Some available implementations in C++:
* [chamfer_matching.cpp] - Implementation by Marius Muja, use the old C OpenCV API.
* [chamfer_matching.cpp (2)] - From the Implementation by Marius Muja, some modifications added.
* [chamfer_matching.cpp (3)] - The original code was written by Marius Muja and later modified and prepared for integration into OpenCV by Antonella Cascitelli, Marco Di Stefano and Stefano Fabri from Univ. of Rome + fixes from Varun Agrawal.
* [chamfer.cpp] - From the code of Antonella Cascitelli, Marco Di Stefano and Stefano Fabri ; Rotation transform added by Aaron Damashek, John Doherty ; for the project: [Detecting Guns Using Parametric Edge Matching].
* [Fast Directional Chamfer Matching] - Used in [Edge Based Tracking library] for the paper: [Robust 3D visual tracking using particle filtering on the special Euclidean group: A combined approach of keypoint and edge features].


## TODO:
* Allow to detect a template image with a different orientation in the query image.
* Speed-up the computation ! (Need to implement a different approach.)
* Test the robustness of the detection.
* Test the detection from multiple template images.


   [Chamfer Matching (contour based shape matching)]: <https://alliance.seas.upenn.edu/~cis581/wiki/Lectures/Fall2013/CIS581-21-13-chamfer-matching.pdf>
   [Fast Directional Chamfer Matching]: <https://www.umiacs.umd.edu/users/vashok/MyPapers/HighlySelectiveConf2010/liu_cvpr2010.pdf>
   [Fast Detection of Multiple Textureless 3-D Objects]: <http://cmp.felk.cvut.cz/~matas/papers/cai-2013-textureless-icvs.pdf>
   [Contour-Based Learning for Object Detection]: <ftp://svr-www.eng.cam.ac.uk/pub/reports/shotton_iccv05.pdf>
   [chamfer_matching.cpp]: <http://robots.stanford.edu/teichman/repos/track_classification/src/ros-pkg/chamfer_matching/src/chamfer_matching.cpp>
   [chamfer_matching.cpp (2)]: <https://github.com/wg-perception/transparent_objects/blob/master/src/chamfer_matching/chamfer_matching.cpp>
   [chamfer_matching.cpp (3)]: <https://github.com/varunagrawal/opencv/blob/2.4/modules/contrib/src/chamfermatching.cpp>
   [chamfer.cpp]: <https://github.com/johndoherty/pistol_detection/blob/master/PistolDetection/chamfer.cpp>
   [Detecting Guns Using Parametric Edge Matching]: <http://cvgl.stanford.edu/teaching/cs231a_winter1415/prev/projects/CS231AGun.pdf>
   [Fast Directional Chamfer Matching]: <https://github.com/CognitiveRobotics/object_tracking_2D/tree/master/3rdparty/Fdcm>
   [Edge Based Tracking library]: <https://github.com/CognitiveRobotics/object_tracking_2D>
   [Robust 3D visual tracking using particle filtering on the special Euclidean group: A combined approach of keypoint and edge features]: <https://people.csail.mit.edu/cchoi/pub/Choi12ijrr.pdf>



