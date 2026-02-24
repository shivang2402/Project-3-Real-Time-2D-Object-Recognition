/**
 * features.cpp
 * Shivang Patel (shivang2402)
 * Feature computation, training, and classification
 */

#include "features.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>
#include <numeric>

/**
 * Convert RegionFeatures to a flat vector for distance computation
 */
std::vector<float> featuresToVector(RegionFeatures &feat) {
    std::vector<float> v;
    v.push_back((float)feat.percentFilled);
    v.push_back((float)feat.bboxAspectRatio);
    // Add Hu moments (log transformed for better scale)
    for (int i = 0; i < (int)feat.huMoments.size(); i++) {
        float hm = feat.huMoments[i];
        float logHm = (hm != 0) ? -1.0f * copysignf(1.0f, hm) * log10f(std::abs(hm)) : 0.0f;
        v.push_back(logHm);
    }
    return v;
}

/**
 * Task 4: Compute features for a given region
 */
RegionFeatures computeFeatures(cv::Mat &regionMap, int regionID, cv::Mat &stats, cv::Mat &centroids) {
    RegionFeatures feat;
    feat.regionID = regionID;

    // Centroid from connectedComponentsWithStats
    feat.cx = centroids.at<double>(regionID, 0);
    feat.cy = centroids.at<double>(regionID, 1);

    int area = stats.at<int>(regionID, cv::CC_STAT_AREA);

    // Create binary mask for this region
    cv::Mat mask = (regionMap == regionID);

    // Compute moments using OpenCV
    cv::Moments m = cv::moments(mask, true);

    // Orientation: axis of least central moment
    // theta = 0.5 * atan2(2 * mu11, mu20 - mu02)
    feat.mu20 = m.mu20;
    feat.mu02 = m.mu02;
    feat.mu11 = m.mu11;
    feat.theta = 0.5 * atan2(2.0 * m.mu11, m.mu20 - m.mu02);

    // Compute oriented bounding box by projecting region pixels onto primary/secondary axes
    float cosT = cos(feat.theta);
    float sinT = sin(feat.theta);

    float minE1 = 1e9, maxE1 = -1e9;
    float minE2 = 1e9, maxE2 = -1e9;

    for (int r = 0; r < regionMap.rows; r++) {
        for (int c = 0; c < regionMap.cols; c++) {
            if (regionMap.at<int>(r, c) != regionID) continue;

            float dx = c - (float)feat.cx;
            float dy = r - (float)feat.cy;

            // Project onto primary axis (e1) and secondary axis (e2)
            float e1 = dx * cosT + dy * sinT;
            float e2 = -dx * sinT + dy * cosT;

            if (e1 < minE1) minE1 = e1;
            if (e1 > maxE1) maxE1 = e1;
            if (e2 < minE2) minE2 = e2;
            if (e2 > maxE2) maxE2 = e2;
        }
    }

    feat.minE1 = minE1;
    feat.maxE1 = maxE1;
    feat.minE2 = minE2;
    feat.maxE2 = maxE2;

    // Oriented bounding box dimensions
    float obbWidth = maxE1 - minE1;
    float obbHeight = maxE2 - minE2;

    // Percent filled
    float obbArea = obbWidth * obbHeight;
    feat.percentFilled = (obbArea > 0) ? (double)area / obbArea : 0;

    // Aspect ratio (always >= 1 so it's rotation invariant)
    if (obbWidth > 0 && obbHeight > 0) {
        feat.bboxAspectRatio = (obbHeight > obbWidth) ? obbHeight / obbWidth : obbWidth / obbHeight;
    } else {
        feat.bboxAspectRatio = 1.0;
    }

    // Hu moments
    double huArr[7];
    cv::HuMoments(m, huArr);
    feat.huMoments.assign(huArr, huArr + 7);

    return feat;
}

/**
 * Task 4: Draw oriented bounding box and primary axis on the image
 */
int drawRegionInfo(cv::Mat &dst, RegionFeatures &feat) {
    float cosT = cos(feat.theta);
    float sinT = sin(feat.theta);
    float cx = (float)feat.cx;
    float cy = (float)feat.cy;

    // Draw primary axis line
    float axisLen = (feat.maxE1 - feat.minE1) * 0.6f;
    cv::Point p1((int)(cx - axisLen * cosT), (int)(cy - axisLen * sinT));
    cv::Point p2((int)(cx + axisLen * cosT), (int)(cy + axisLen * sinT));
    cv::line(dst, p1, p2, cv::Scalar(0, 0, 255), 2);

    // Draw oriented bounding box (4 corners)
    float e1Vals[2] = { (float)feat.minE1, (float)feat.maxE1 };
    float e2Vals[2] = { (float)feat.minE2, (float)feat.maxE2 };

    cv::Point corners[4];
    int idx = 0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float x = cx + e1Vals[i] * cosT - e2Vals[j] * sinT;
            float y = cy + e1Vals[i] * sinT + e2Vals[j] * cosT;
            corners[idx++] = cv::Point((int)x, (int)y);
        }
    }
    // Connect: 0-1, 1-3, 3-2, 2-0 (the order due to nested loop)
    cv::line(dst, corners[0], corners[1], cv::Scalar(0, 255, 0), 2);
    cv::line(dst, corners[1], corners[3], cv::Scalar(0, 255, 0), 2);
    cv::line(dst, corners[3], corners[2], cv::Scalar(0, 255, 0), 2);
    cv::line(dst, corners[2], corners[0], cv::Scalar(0, 255, 0), 2);

    // Draw centroid
    cv::circle(dst, cv::Point((int)cx, (int)cy), 5, cv::Scalar(255, 0, 0), -1);

    // Display feature values
    char buf[128];
    snprintf(buf, sizeof(buf), "Fill: %.2f  AR: %.2f", feat.percentFilled, feat.bboxAspectRatio);
    cv::putText(dst, buf, cv::Point((int)cx - 50, (int)cy + (int)(feat.maxE2) + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);

    return 0;
}

/**
 * Task 5: Append a labeled feature vector to the object DB CSV
 * Format: label, f1, f2, f3, ...
 */
int saveTrainingData(const std::string &filename, const std::string &label, RegionFeatures &feat) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Cannot open DB file: " << filename << std::endl;
        return 1;
    }

    std::vector<float> fv = featuresToVector(feat);
    file << label;
    for (float val : fv) {
        file << "," << val;
    }
    file << "\n";
    file.close();
    return 0;
}

/**
 * Task 5: Load all training data from CSV
 */
int loadTrainingData(const std::string &filename, std::vector<std::string> &labels, std::vector<std::vector<float>> &featureVectors) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        // File doesn't exist yet, that's fine
        return 0;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string label;
        std::getline(ss, label, ',');
        labels.push_back(label);

        std::vector<float> fv;
        std::string val;
        while (std::getline(ss, val, ',')) {
            fv.push_back(std::stof(val));
        }
        featureVectors.push_back(fv);
    }

    file.close();
    return 0;
}

/**
 * Task 6: Classify using scaled Euclidean distance
 * D = sqrt( sum( ((x_i - y_i) / stdev_i)^2 ) )
 * Returns the label of the nearest neighbor
 */
std::string classify(RegionFeatures &feat, std::vector<std::string> &labels, std::vector<std::vector<float>> &featureVectors, double &minDist) {
    if (labels.empty()) return "unknown";

    std::vector<float> query = featuresToVector(feat);
    int dim = (int)query.size();

    // Compute standard deviation for each feature across all training data
    std::vector<float> means(dim, 0.0f);
    std::vector<float> stdevs(dim, 0.0f);

    for (auto &fv : featureVectors) {
        for (int i = 0; i < dim && i < (int)fv.size(); i++) {
            means[i] += fv[i];
        }
    }
    int n = (int)featureVectors.size();
    for (int i = 0; i < dim; i++) means[i] /= n;

    for (auto &fv : featureVectors) {
        for (int i = 0; i < dim && i < (int)fv.size(); i++) {
            float diff = fv[i] - means[i];
            stdevs[i] += diff * diff;
        }
    }
    for (int i = 0; i < dim; i++) {
        stdevs[i] = sqrt(stdevs[i] / n);
        if (stdevs[i] < 1e-6f) stdevs[i] = 1.0f; // avoid division by zero
    }

    // Find nearest neighbor using scaled Euclidean
    minDist = 1e30;
    int bestIdx = 0;

    for (int j = 0; j < n; j++) {
        double dist = 0;
        for (int i = 0; i < dim && i < (int)featureVectors[j].size(); i++) {
            float d = (query[i] - featureVectors[j][i]) / stdevs[i];
            dist += d * d;
        }
        dist = sqrt(dist);
        if (dist < minDist) {
            minDist = dist;
            bestIdx = j;
        }
    }

    return labels[bestIdx];
}
std::string classifyKNN(RegionFeatures &feat, std::vector<std::string> &labels, std::vector<std::vector<float>> &featureVectors, int k) {
    if (labels.empty()) return "unknown";
    if (k > (int)labels.size()) k = (int)labels.size();
    std::vector<float> query = featuresToVector(feat);
    int dim = (int)query.size();
    int n = (int)featureVectors.size();
    std::vector<float> means(dim, 0.0f), stdevs(dim, 0.0f);
    for (auto &fv : featureVectors)
        for (int i = 0; i < dim && i < (int)fv.size(); i++) means[i] += fv[i];
    for (int i = 0; i < dim; i++) means[i] /= n;
    for (auto &fv : featureVectors)
        for (int i = 0; i < dim && i < (int)fv.size(); i++) { float d = fv[i]-means[i]; stdevs[i] += d*d; }
    for (int i = 0; i < dim; i++) { stdevs[i] = sqrt(stdevs[i]/n); if (stdevs[i]<1e-6f) stdevs[i]=1.0f; }
    std::vector<std::pair<double,int>> dists;
    for (int j = 0; j < n; j++) {
        double dist = 0;
        for (int i = 0; i < dim && i < (int)featureVectors[j].size(); i++) { float d=(query[i]-featureVectors[j][i])/stdevs[i]; dist+=d*d; }
        dists.push_back({sqrt(dist), j});
    }
    std::sort(dists.begin(), dists.end());
    std::map<std::string,int> votes;
    for (int i = 0; i < k; i++) votes[labels[dists[i].second]]++;
    std::string best = labels[dists[0].second];
    int bestCount = 0;
    for (auto &p : votes) if (p.second > bestCount) { bestCount = p.second; best = p.first; }
    return best;
}
