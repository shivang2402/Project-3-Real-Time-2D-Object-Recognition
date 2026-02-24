/**
 * features.h
 * Shivang Patel (shivang2402)
 * Feature computation, training, and classification
 */

#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Struct to hold region features
struct RegionFeatures {
    int regionID;
    double cx, cy;              // centroid
    double theta;               // orientation (axis of least central moment)
    double percentFilled;       // percent of bounding box filled
    double bboxAspectRatio;     // height/width of oriented bounding box
    double mu20, mu02, mu11;    // central moments (normalized, for additional features)
    double minE1, maxE1;        // extent along primary axis
    double minE2, maxE2;        // extent along secondary axis
    std::vector<float> huMoments; // Hu moments (7 values)
};

// Task 4: Compute features for a region
RegionFeatures computeFeatures(cv::Mat &regionMap, int regionID, cv::Mat &stats, cv::Mat &centroids);

// Task 4: Draw oriented bounding box and primary axis on image
int drawRegionInfo(cv::Mat &dst, RegionFeatures &feat);

// Task 5: Save a labeled feature vector to the object DB file
int saveTrainingData(const std::string &filename, const std::string &label, RegionFeatures &feat);

// Task 5: Load all training data from the object DB file
int loadTrainingData(const std::string &filename, std::vector<std::string> &labels, std::vector<std::vector<float>> &featureVectors);

// Task 6: Classify using scaled Euclidean distance (nearest neighbor)
std::string classifyKNN(RegionFeatures &feat, std::vector<std::string> &labels, std::vector<std::vector<float>> &featureVectors, int k);
std::string classify(RegionFeatures &feat, std::vector<std::string> &labels, std::vector<std::vector<float>> &featureVectors, double &minDist);

// Convert RegionFeatures to a flat feature vector for comparison
std::vector<float> featuresToVector(RegionFeatures &feat);

#endif