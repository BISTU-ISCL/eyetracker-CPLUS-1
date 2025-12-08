#pragma once

// Core OpenCV headers for image processing and matrix utilities
#include <opencv2/opencv.hpp>
// The face landmark APIs live in the opencv_contrib "face" module.
// Install OpenCV with opencv_contrib or ensure face.hpp from that module is available.
// Reference: https://github.com/opencv/opencv_contrib/blob/4.x/modules/face/include/opencv2/face.hpp
#include <opencv2/face.hpp>
#include <deque>
#include <optional>
#include <vector>

// A single calibration observation tying eye features to a known screen point.
struct CalibrationSample {
    cv::Point2f pupilCenter;      // pupil center in image coordinates
    cv::Point2f glint;            // specular highlight location on iris
    cv::Point2f screenPoint;      // known screen calibration target
    cv::Vec3d headRotation;       // head rotation vector (Rodrigues)
    cv::Vec3d headTranslation;    // head translation vector
};

// Basic metrics exposed to the application layer.
struct PupilMetrics {
    float diameterPx{};     // approximated pupil diameter in pixels
    float blinkRateHz{};    // blinks per second (rolling window)
};

// EyeTrackerCalibrator ties together face detection, landmark extraction,
// head-pose estimation, dark-pupil pupil/glint detection, and a linear
// regression mapping to screen coordinates.
class EyeTrackerCalibrator {
public:
    EyeTrackerCalibrator();

    // Load the Haar cascade for face detection and the LBF model for landmarks.
    bool loadModels(const std::string &faceDetectorPath,
                    const std::string &landmarkModelPath);

    // Process one video frame and optionally append calibration samples
    // (targets) until the supplied target list is exhausted.
    bool processFrame(const cv::Mat &frame,
                      const std::vector<cv::Point2f> &screenCalibrationTargets,
                      cv::Point2f &estimatedGaze,
                      PupilMetrics &metrics,
                      cv::Mat &debugFrame);

    // Clear any collected calibration samples.
    void resetCalibration();

private:
    // Detect the largest face in the grayscale image.
    bool detectFace(const cv::Mat &gray, cv::Rect &faceBox);
    // Fit facial landmarks within the detected bounding box.
    bool estimateLandmarks(const cv::Mat &gray, const cv::Rect &faceBox,
                           std::vector<cv::Point2f> &landmarks);
    // Solve for head pose using a small subset of 2D landmarks.
    bool estimateHeadPose(const std::vector<cv::Point2f> &landmarks,
                          cv::Vec3d &rvec, cv::Vec3d &tvec);
    // Dark-pupil detection plus specular highlight extraction.
    bool detectPupilAndGlint(const cv::Mat &eyeRoi, cv::Point2f &pupilCenter,
                             cv::Point2f &glint, float &pupilDiameterPx);

    // Map pupil/glint observations (plus head pose) to screen gaze.
    std::optional<cv::Point2f> estimateGazePoint(const cv::Point2f &pupilCenter,
                                                 const cv::Point2f &glint,
                                                 const cv::Vec3d &rvec,
                                                 const cv::Vec3d &tvec);
    // Update rolling blink frequency by monitoring eye-aspect ratio changes.
    void updateBlinkRate(bool eyeClosed);

    cv::CascadeClassifier faceDetector_;
    cv::Ptr<cv::face::Facemark> facemark_;

    std::deque<CalibrationSample> calibrationBuffer_;
    std::deque<int64_t> blinkTimestampsMs_;
    int64_t lastBlinkStateChangeMs_{0};
    bool lastEyeClosed_{false};
};

