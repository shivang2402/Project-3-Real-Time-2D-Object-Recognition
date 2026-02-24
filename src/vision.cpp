/**
 * vision.cpp
 * Shivang Patel (shivang2402)
 * Core vision pipeline: thresholding, morphology, segmentation
 */

#include "vision.h"
#include <iostream>
#include <cmath>

/**
 * Task 1: Custom thresholding (written from scratch, no cv::threshold)
 * 
 * Strategy: Convert to HSV. Use both Value (brightness) and Saturation
 * to separate dark/colored objects from white background.
 * Uses dynamic threshold via k-means (ISODATA) with K=2 on sampled pixels.
 * 
 * A pixel is foreground if:
 *   - its brightness (V) is below the computed threshold, OR
 *   - its saturation (S) is high enough (strongly colored objects on white bg)
 */
int threshold(cv::Mat &src, cv::Mat &dst) {
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    int rows = hsv.rows;
    int cols = hsv.cols;

    // Step 1: Sample pixels for ISODATA (use every 4th row and col = 1/16 pixels)
    std::vector<float> samples;
    for (int r = 0; r < rows; r += 4) {
        for (int c = 0; c < cols; c += 4) {
            float v = hsv.at<cv::Vec3b>(r, c)[2]; // Value channel
            float s = hsv.at<cv::Vec3b>(r, c)[1]; // Saturation channel
            // Combine: reduce brightness for saturated pixels (push colored things darker)
            float combined = v - 0.5f * s;
            samples.push_back(combined);
        }
    }

    // Step 2: ISODATA / K-means with K=2 to find two cluster means
    float mean1 = 50.0f;   // initial guess for dark cluster
    float mean2 = 200.0f;  // initial guess for light cluster

    for (int iter = 0; iter < 20; iter++) {
        float sum1 = 0, sum2 = 0;
        int count1 = 0, count2 = 0;

        for (float val : samples) {
            float d1 = std::abs(val - mean1);
            float d2 = std::abs(val - mean2);
            if (d1 < d2) {
                sum1 += val;
                count1++;
            } else {
                sum2 += val;
                count2++;
            }
        }

        float newMean1 = (count1 > 0) ? sum1 / count1 : mean1;
        float newMean2 = (count2 > 0) ? sum2 / count2 : mean2;

        // Check convergence
        if (std::abs(newMean1 - mean1) < 0.5f && std::abs(newMean2 - mean2) < 0.5f)
            break;

        mean1 = newMean1;
        mean2 = newMean2;
    }

    // Threshold is midpoint between the two means
    float thresh = (mean1 + mean2) / 2.0f;

    // Step 3: Apply threshold pixel by pixel
    dst = cv::Mat::zeros(rows, cols, CV_8UC1);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float v = hsv.at<cv::Vec3b>(r, c)[2];
            float s = hsv.at<cv::Vec3b>(r, c)[1];
            float combined = v - 0.5f * s;

            if (combined < thresh) {
                dst.at<uchar>(r, c) = 255; // foreground (object)
            }
        }
    }

    return 0;
}

/**
 * Task 2: Morphological cleanup
 * Strategy: erosion then dilation (opening) to remove small noise,
 * followed by dilation then erosion (closing) to fill small holes.
 */
int morphCleanup(cv::Mat &src, cv::Mat &dst) {
    cv::Mat temp;
    // 5x5 cross shaped kernel
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));

    // Opening: remove small noise
    cv::erode(src, temp, kernel, cv::Point(-1, -1), 1);
    cv::dilate(temp, temp, kernel, cv::Point(-1, -1), 1);

    // Closing: fill small holes
    cv::dilate(temp, temp, kernel, cv::Point(-1, -1), 2);
    cv::erode(temp, dst, kernel, cv::Point(-1, -1), 2);

    return 0;
}

/**
 * Task 3: Connected components segmentation
 * Returns number of labels (including background at label 0)
 */
int segment(cv::Mat &binary, cv::Mat &regionMap, cv::Mat &stats, cv::Mat &centroids, int minRegionSize) {
    int numLabels = cv::connectedComponentsWithStats(binary, regionMap, stats, centroids, 8, CV_32S);
    return numLabels;
}

/**
 * Task 3: Colorize region map for visualization
 * Each region gets a distinct random color; background stays black
 */
int colorRegions(cv::Mat &regionMap, cv::Mat &dst, int numLabels) {
    // Generate random colors for each label
    std::vector<cv::Vec3b> colors(numLabels);
    colors[0] = cv::Vec3b(0, 0, 0); // background = black

    // Fixed seed for consistent colors across frames
    cv::RNG rng(42);
    for (int i = 1; i < numLabels; i++) {
        colors[i] = cv::Vec3b(rng.uniform(50, 256), rng.uniform(50, 256), rng.uniform(50, 256));
    }

    dst = cv::Mat::zeros(regionMap.size(), CV_8UC3);
    for (int r = 0; r < regionMap.rows; r++) {
        for (int c = 0; c < regionMap.cols; c++) {
            int label = regionMap.at<int>(r, c);
            if (label > 0 && label < numLabels) {
                dst.at<cv::Vec3b>(r, c) = colors[label];
            }
        }
    }

    return 0;
}