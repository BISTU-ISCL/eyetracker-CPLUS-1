#include "EyeTrackerCalibrator.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <haar_cascade.xml> <lbfmodel.yaml>" << std::endl;
        return 1;
    }

    EyeTrackerCalibrator calibrator;
    if (!calibrator.loadModels(argv[1], argv[2])) {
        std::cerr << "Failed to load detector/landmark models" << std::endl;
        return 1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Unable to open camera" << std::endl;
        return 1;
    }

    // Example 9-point calibration grid
    std::vector<cv::Point2f> targets;
    const int cols = 3, rows = 3;
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            targets.emplace_back((x + 1) * 1920.f / (cols + 1), (y + 1) * 1080.f / (rows + 1));
        }
    }

    std::cout << "Press 'r' to reset calibration, ESC to quit." << std::endl;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        cv::Point2f gaze;
        PupilMetrics metrics;
        cv::Mat debug;
        bool ok = calibrator.processFrame(frame, targets, gaze, metrics, debug);

        if (ok) {
            cv::circle(debug, gaze, 5, {0, 0, 255}, -1);
        } else {
            cv::putText(frame, "Tracking lost", {10, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 255}, 2);
            debug = frame;
        }

        cv::imshow("Eye Tracker", debug);
        int key = cv::waitKey(1);
        if (key == 27) break;
        if (key == 'r') calibrator.resetCalibration();
    }
    return 0;
}

