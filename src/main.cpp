/**
 * main.cpp
 * Shivang Patel (shivang2402)
 * Real-time 2D Object Recognition
 * 
 * Modes:
 *   c = color (original)
 *   t = threshold (Task 1)
 *   m = morphological cleanup (Task 2)
 *   s = segmentation/regions (Task 3)
 *   f = features overlay (Task 4)
 *   n = train: save current object to DB (Task 5)
 *   r = classify/recognize (Task 6)
 *   q = quit
 *   w = save screenshot
 */

#include "vision.h"
#include "features.h"
#include <iostream>
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    // Determine input mode: camera (default) or image directory/video
    std::string inputPath = "";
    bool useCamera = true;

    if (argc >= 2) {
        inputPath = argv[1];
        useCamera = false;
    }

    cv::VideoCapture cap;
    std::vector<std::string> imageFiles;
    int imgIndex = 0;

    if (useCamera) {
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open camera. Provide an image path as argument instead." << std::endl;
            return 1;
        }
        std::cout << "Camera opened: "
                  << cap.get(cv::CAP_PROP_FRAME_WIDTH) << " x "
                  << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    } else {
        // Check if inputPath is a video file or directory of images
        if (inputPath.find(".mp4") != std::string::npos ||
            inputPath.find(".avi") != std::string::npos ||
            inputPath.find(".mov") != std::string::npos) {
            cap.open(inputPath);
            if (!cap.isOpened()) {
                std::cerr << "Cannot open video: " << inputPath << std::endl;
                return 1;
            }
        } else {
            // Assume directory of images
            cv::glob(inputPath + "/*.png", imageFiles, false);
            std::vector<std::string> jpgFiles;
            cv::glob(inputPath + "/*.jpg", jpgFiles, false);
            imageFiles.insert(imageFiles.end(), jpgFiles.begin(), jpgFiles.end());
            std::sort(imageFiles.begin(), imageFiles.end());
            if (imageFiles.empty()) {
                std::cerr << "No images found in: " << inputPath << std::endl;
                return 1;
            }
            std::cout << "Loaded " << imageFiles.size() << " images from " << inputPath << std::endl;
        }
    }

    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);

    std::string dbFile = "data/object_db.csv";
    char mode = 'c';
    int screenshotCount = 0;
    int minRegionSize = 5000;

    // Training data (loaded once)
    std::vector<std::string> trainLabels;
    std::vector<std::vector<float>> trainFeatures;
    loadTrainingData(dbFile, trainLabels, trainFeatures);
    std::cout << "Loaded " << trainLabels.size() << " training samples." << std::endl;

    std::cout << "Keys: c=color t=threshold m=morph s=segment f=features n=train r=recognize w=save q=quit" << std::endl;
    if (!imageFiles.empty()) {
        std::cout << "      [/] = prev/next image" << std::endl;
    }

    cv::Mat frame, binary, cleaned, regionMap, stats, centroids, display;

    for (;;) {
        // Get next frame
        if (!imageFiles.empty()) {
            frame = cv::imread(imageFiles[imgIndex]);
            if (frame.empty()) break;
        } else {
            cap >> frame;
            if (frame.empty()) break;
        }

        cv::imshow("Original", frame);

        // Pipeline: always run threshold and cleanup so later stages have data
        threshold(frame, binary);
        morphCleanup(binary, cleaned);

        int numLabels = 0;
        if (mode == 's' || mode == 'f' || mode == 'n' || mode == 'r') {
            numLabels = segment(cleaned, regionMap, stats, centroids, minRegionSize);
        }

        switch (mode) {
            case 'c':
                display = frame.clone();
                break;

            case 't':
                cv::cvtColor(binary, display, cv::COLOR_GRAY2BGR);
                break;

            case 'm':
                cv::cvtColor(cleaned, display, cv::COLOR_GRAY2BGR);
                break;

            case 's': {
                colorRegions(regionMap, display, numLabels);
                break;
            }

            case 'f': {
                display = frame.clone();
                // Draw features for the largest valid region
                for (int i = 1; i < numLabels; i++) {
                    int area = stats.at<int>(i, cv::CC_STAT_AREA);
                    if (area < minRegionSize) continue;
                    RegionFeatures feat = computeFeatures(regionMap, i, stats, centroids);
                    drawRegionInfo(display, feat);
                }
                break;
            }

            case 'n': {
                // Training mode: prompt for label
                display = frame.clone();
                for (int i = 1; i < numLabels; i++) {
                    int area = stats.at<int>(i, cv::CC_STAT_AREA);
                    if (area < minRegionSize) continue;
                    RegionFeatures feat = computeFeatures(regionMap, i, stats, centroids);
                    drawRegionInfo(display, feat);

                    std::cout << "Enter label for this object: ";
                    std::string label;
                    std::getline(std::cin, label);
                    if (!label.empty()) {
                        saveTrainingData(dbFile, label, feat);
                        // Reload training data
                        trainLabels.clear();
                        trainFeatures.clear();
                        loadTrainingData(dbFile, trainLabels, trainFeatures);
                        std::cout << "Saved! Total training samples: " << trainLabels.size() << std::endl;
                    }
                }
                mode = 'f'; // switch back to features mode after training
                break;
            }

            case 'r': {
                display = frame.clone();
                for (int i = 1; i < numLabels; i++) {
                    int area = stats.at<int>(i, cv::CC_STAT_AREA);
                    if (area < minRegionSize) continue;
                    RegionFeatures feat = computeFeatures(regionMap, i, stats, centroids);
                    drawRegionInfo(display, feat);

                    double minDist = 0;
                    std::string label = classify(feat, trainLabels, trainFeatures, minDist);
                    cv::putText(display, label, cv::Point((int)feat.cx - 30, (int)feat.cy - 60),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                }
                break;
            }

            default:
                display = frame.clone();
                break;
        }

        cv::imshow("Output", display);
        // Auto-train mode: press 'b' to batch train all images with known labels
        // Auto-save mode: press 'a' to save t, m, s, f views of ALL images
        char key = (char)cv::waitKey(useCamera ? 10 : 0);

        // Embedding classification mode: press 'e' to classify using ResNet18 embeddings
        if (key == 'e' && !imageFiles.empty()) {
            static cv::dnn::Net net;
            static bool netLoaded = false;
            static std::vector<std::string> embLabels;
            static std::vector<cv::Mat> embVectors;

            if (!netLoaded) {
                std::string modelPath = "resnet18-v2-7.onnx";
                net = cv::dnn::readNet(modelPath);
                if (net.empty()) {
                    std::cout << "Cannot load ResNet18 model from " << modelPath << std::endl;
                } else {
                    netLoaded = true;
                    std::cout << "ResNet18 loaded. Building embedding DB..." << std::endl;

                    std::map<std::string, std::string> labelMap = {
                        {"img1p3", "triangle"}, {"img2P3", "squeegee"},
                        {"img3P3", "allenkey"}, {"img4P3", "chisel"}, {"img5P3", "keyfob"},
                        {"obj1", "chair"}, {"obj2", "mug"}, {"obj3", "stand"},
                        {"obj5", "desk"}
                    };

                    for (int idx = 0; idx < (int)imageFiles.size(); idx++) {
                        cv::Mat img = cv::imread(imageFiles[idx]);
                        if (img.empty()) continue;

                        std::string base = imageFiles[idx];
                        size_t slash = base.find_last_of("/");
                        size_t dot = base.find_last_of(".");
                        std::string name = base.substr(slash + 1, dot - slash - 1);

                        if (labelMap.find(name) == labelMap.end()) continue;

                        cv::Mat bin, cln, rmap, st, cent;
                        threshold(img, bin);
                        morphCleanup(bin, cln);
                        int nl = segment(cln, rmap, st, cent, minRegionSize);

                        int bestRegion = -1, bestArea = 0;
                        for (int i = 1; i < nl; i++) {
                            int area = st.at<int>(i, cv::CC_STAT_AREA);
                            if (area > bestArea && area >= minRegionSize) {
                                bestArea = area;
                                bestRegion = i;
                            }
                        }

                        if (bestRegion >= 0) {
                            RegionFeatures feat = computeFeatures(rmap, bestRegion, st, cent);
                            cv::Mat embImage;
                            prepEmbeddingImage(img, embImage, (int)feat.cx, (int)feat.cy,
                                (float)feat.theta, feat.minE1, feat.maxE1, feat.minE2, feat.maxE2, 0);

                            if (!embImage.empty()) {
                                cv::Mat embedding;
                                getEmbedding(embImage, embedding, net, 0);
                                embLabels.push_back(labelMap[name]);
                                embVectors.push_back(embedding.clone());
                                std::cout << "Embedded: " << name << " -> " << labelMap[name] << std::endl;
                            }
                        }
                    }
                    std::cout << "Embedding DB built with " << embLabels.size() << " samples." << std::endl;
                }
            }

            if (netLoaded && !embVectors.empty()) {
                // Classify current image using embeddings
                cv::Mat img = cv::imread(imageFiles[imgIndex]);
                cv::Mat bin, cln, rmap, st, cent;
                threshold(img, bin);
                morphCleanup(bin, cln);
                int nl = segment(cln, rmap, st, cent, minRegionSize);

                int bestRegion = -1, bestArea = 0;
                for (int i = 1; i < nl; i++) {
                    int area = st.at<int>(i, cv::CC_STAT_AREA);
                    if (area > bestArea && area >= minRegionSize) {
                        bestArea = area;
                        bestRegion = i;
                    }
                }

                if (bestRegion >= 0) {
                    RegionFeatures feat = computeFeatures(rmap, bestRegion, st, cent);
                    cv::Mat embImage;
                    prepEmbeddingImage(img, embImage, (int)feat.cx, (int)feat.cy,
                        (float)feat.theta, feat.minE1, feat.maxE1, feat.minE2, feat.maxE2, 0);

                    if (!embImage.empty()) {
                        cv::Mat queryEmb;
                        getEmbedding(embImage, queryEmb, net, 0);

                        // Find nearest neighbor using SSD
                        double minDist = 1e30;
                        int bestIdx = 0;
                        for (int j = 0; j < (int)embVectors.size(); j++) {
                            double dist = cv::norm(queryEmb, embVectors[j], cv::NORM_L2SQR);
                            if (dist < minDist) {
                                minDist = dist;
                                bestIdx = j;
                            }
                        }

                        display = frame.clone();
                        cv::putText(display, "EMB: " + embLabels[bestIdx],
                            cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 3);
                        cv::imshow("Output", display);
                        std::cout << "Embedding classification: " << embLabels[bestIdx] << " (dist=" << minDist << ")" << std::endl;
                    }
                }
            }
        }

        // Confusion matrix mode: press 'x' to evaluate all labeled images
        if (key == 'x' && !imageFiles.empty()) {
            std::map<std::string, std::string> labelMap = {
                {"img1p3", "triangle"}, {"img2P3", "squeegee"},
                {"img3P3", "allenkey"}, {"img4P3", "chisel"}, {"img5P3", "keyfob"},
                {"obj1", "chair"}, {"obj2", "mug"}, {"obj3", "stand"},
                {"obj5", "desk"}
            };

            // Get unique labels
            std::vector<std::string> categories = {"triangle", "squeegee", "allenkey", "chisel", "keyfob", "chair", "mug", "stand", "desk"};
            int nc = (int)categories.size();
            std::vector<std::vector<int>> confusion(nc, std::vector<int>(nc, 0));

            std::map<std::string, int> catIdx;
            for (int i = 0; i < nc; i++) catIdx[categories[i]] = i;

            for (int idx = 0; idx < (int)imageFiles.size(); idx++) {
                cv::Mat img = cv::imread(imageFiles[idx]);
                if (img.empty()) continue;

                std::string base = imageFiles[idx];
                size_t slash = base.find_last_of("/");
                size_t dot = base.find_last_of(".");
                std::string name = base.substr(slash + 1, dot - slash - 1);

                if (labelMap.find(name) == labelMap.end()) continue;
                std::string trueLabel = labelMap[name];

                cv::Mat bin, cln, rmap, st, cent;
                threshold(img, bin);
                morphCleanup(bin, cln);
                int nl = segment(cln, rmap, st, cent, minRegionSize);

                int bestRegion = -1, bestArea = 0;
                for (int i = 1; i < nl; i++) {
                    int area = st.at<int>(i, cv::CC_STAT_AREA);
                    if (area > bestArea && area >= minRegionSize) {
                        bestArea = area;
                        bestRegion = i;
                    }
                }

                if (bestRegion >= 0) {
                    RegionFeatures feat = computeFeatures(rmap, bestRegion, st, cent);
                    double minDist = 0;
                    std::string predicted = classify(feat, trainLabels, trainFeatures, minDist);
                    std::cout << name << ": true=" << trueLabel << " predicted=" << predicted << " dist=" << minDist << std::endl;

                    if (catIdx.count(trueLabel) && catIdx.count(predicted)) {
                        confusion[catIdx[trueLabel]][catIdx[predicted]]++;
                    }
                }
            }

            // Print confusion matrix
            std::cout << "\nConfusion Matrix:" << std::endl;
            std::cout << "True\\Pred\t";
            for (auto &c : categories) std::cout << c << "\t";
            std::cout << std::endl;
            for (int i = 0; i < nc; i++) {
                std::cout << categories[i] << "\t";
                for (int j = 0; j < nc; j++) {
                    std::cout << confusion[i][j] << "\t";
                }
                std::cout << std::endl;
            }
        }

        if (key == 'b' && !imageFiles.empty()) {
            // Label map: filename prefix -> label
            std::map<std::string, std::string> labelMap = {
                {"obj1", "chair"}, {"obj2", "mug"}, {"obj3", "stand"},
                {"obj5", "desk"},
                {"img1p3", "triangle"}, {"img2P3", "squeegee"},
                {"img3P3", "allenkey"}, {"img4P3", "chisel"}, {"img5P3", "keyfob"}
            };

            // Clear existing DB
            std::ofstream clearFile(dbFile, std::ios::trunc);
            clearFile.close();

            std::cout << "Auto-training..." << std::endl;
            for (int idx = 0; idx < (int)imageFiles.size(); idx++) {
                cv::Mat img = cv::imread(imageFiles[idx]);
                if (img.empty()) continue;

                std::string base = imageFiles[idx];
                size_t slash = base.find_last_of("/");
                size_t dot = base.find_last_of(".");
                std::string name = base.substr(slash + 1, dot - slash - 1);

                // Check if this image has a label
                if (labelMap.find(name) == labelMap.end()) {
                    std::cout << "Skipping (no label): " << name << std::endl;
                    continue;
                }

                cv::Mat bin, cln, rmap, st, cent;
                threshold(img, bin);
                morphCleanup(bin, cln);
                int nl = segment(cln, rmap, st, cent, minRegionSize);

                // Find largest valid region
                int bestRegion = -1, bestArea = 0;
                for (int i = 1; i < nl; i++) {
                    int area = st.at<int>(i, cv::CC_STAT_AREA);
                    if (area > bestArea && area >= minRegionSize) {
                        bestArea = area;
                        bestRegion = i;
                    }
                }

                if (bestRegion >= 0) {
                    RegionFeatures feat = computeFeatures(rmap, bestRegion, st, cent);
                    saveTrainingData(dbFile, labelMap[name], feat);
                    std::cout << "Trained: " << name << " -> " << labelMap[name] << std::endl;
                } else {
                    std::cout << "No region found: " << name << std::endl;
                }
            }

            // Reload
            trainLabels.clear();
            trainFeatures.clear();
            loadTrainingData(dbFile, trainLabels, trainFeatures);
            std::cout << "Done! " << trainLabels.size() << " training samples." << std::endl;
        }

        if (key == 'a' && !imageFiles.empty()) {
            std::cout << "Auto-saving all views..." << std::endl;
            for (int idx = 0; idx < (int)imageFiles.size(); idx++) {
                cv::Mat img = cv::imread(imageFiles[idx]);
                if (img.empty()) continue;

                std::string base = imageFiles[idx];
                size_t slash = base.find_last_of("/");
                size_t dot = base.find_last_of(".");
                std::string name = base.substr(slash + 1, dot - slash - 1);

                cv::Mat bin, cln, rmap, st, cent, out;

                // Threshold
                threshold(img, bin);
                cv::cvtColor(bin, out, cv::COLOR_GRAY2BGR);
                cv::imwrite("data/report_" + name + "_threshold.png", out);

                // Morph
                morphCleanup(bin, cln);
                cv::cvtColor(cln, out, cv::COLOR_GRAY2BGR);
                cv::imwrite("data/report_" + name + "_morph.png", out);

                // Segment
                int nl = segment(cln, rmap, st, cent, minRegionSize);
                colorRegions(rmap, out, nl);
                cv::imwrite("data/report_" + name + "_segment.png", out);

                // Features
                out = img.clone();
                for (int i = 1; i < nl; i++) {
                    int area = st.at<int>(i, cv::CC_STAT_AREA);
                    if (area < minRegionSize) continue;
                    RegionFeatures feat = computeFeatures(rmap, i, st, cent);
                    drawRegionInfo(out, feat);
                }
                cv::imwrite("data/report_" + name + "_features.png", out);

                std::cout << "Saved views for: " << name << std::endl;
            }
            std::cout << "Done! Check data/ for report_*.png files." << std::endl;
        }

        if (key == 'q' || key == 'Q') break;
        if (key == 'w' || key == 'W') {
            std::string fname = "data/screenshot_" + std::to_string(screenshotCount++) + ".png";
            cv::imwrite(fname, display);
            std::cout << "Saved: " << fname << std::endl;
        }
        if (key == ']' && !imageFiles.empty()) {
            imgIndex = (imgIndex + 1) % (int)imageFiles.size();
            std::cout << "Image: " << imageFiles[imgIndex] << std::endl;
        }
        if (key == '[' && !imageFiles.empty()) {
            imgIndex = (imgIndex - 1 + (int)imageFiles.size()) % (int)imageFiles.size();
            std::cout << "Image: " << imageFiles[imgIndex] << std::endl;
        }
        if (std::string("ctmsfnr").find(key) != std::string::npos) {
            mode = key;
            std::cout << "Mode: " << mode << std::endl;
        }
    }

    cv::destroyAllWindows();
    return 0;
}