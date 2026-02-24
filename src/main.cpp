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

    std::string dbFile = "../data/object_db.csv";
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
        char key = (char)cv::waitKey(useCamera ? 10 : 0);

        if (key == 'q' || key == 'Q') break;
        if (key == 'w' || key == 'W') {
            std::string fname = "../data/screenshot_" + std::to_string(screenshotCount++) + ".png";
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