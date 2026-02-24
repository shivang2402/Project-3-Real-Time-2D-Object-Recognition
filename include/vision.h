/**
 * vision.h
 * Shivang Patel (shivang2402)
 * Core vision pipeline: thresholding, morphology, segmentation
 */

#ifndef VISION_H
#define VISION_H

#include <opencv2/opencv.hpp>
#include <vector>

// Task 1: Custom thresholding (from scratch)
// Separates dark objects from white/light background
// dst is a single channel binary image (0 = background, 255 = foreground)
int threshold(cv::Mat &src, cv::Mat &dst);

// Task 2: Morphological cleanup
// Removes noise and fills holes in binary image
int morphCleanup(cv::Mat &src, cv::Mat &dst);

// Task 3: Connected components segmentation
// Returns a region map (labeled image) and stats
// Uses OpenCV connectedComponentsWithStats
int segment(cv::Mat &binary, cv::Mat &regionMap, cv::Mat &stats, cv::Mat &centroids, int minRegionSize);

// Task 3: Colorize region map for display
int colorRegions(cv::Mat &regionMap, cv::Mat &dst, int numLabels);

#endif
// Utilities for Task 9 (from utilities.cpp)
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug);
void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage, int cx, int cy, float theta, float minE1, float maxE1, float minE2, float maxE2, int debug);
