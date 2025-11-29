#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <deque>
#include <optional>
#include <vector>

struct CalibrationSample {
    cv::Point2f pupilCenter;      // pupil center in image coordinates
    cv::Point2f glint;            // specular highlight location on iris
    cv::Point2f screenPoint;      // known screen calibration target
    cv::Vec3d headRotation;       // head rotation vector (Rodrigues)
    cv::Vec3d headTranslation;    // head translation vector
};

struct PupilMetrics {
    float diameterPx{};     // approximated pupil diameter in pixels
    float blinkRateHz{};    // blinks per second (rolling window)
};

class EyeTrackerCalibrator {
public:
    EyeTrackerCalibrator();

    bool loadModels(const std::string &faceDetectorPath,
                    const std::string &landmarkModelPath);

    bool processFrame(const cv::Mat &frame,
                      const std::vector<cv::Point2f> &screenCalibrationTargets,
                      cv::Point2f &estimatedGaze,
                      PupilMetrics &metrics,
                      cv::Mat &debugFrame);

    void resetCalibration();

private:
    bool detectFace(const cv::Mat &gray, cv::Rect &faceBox);
    bool estimateLandmarks(const cv::Mat &gray, const cv::Rect &faceBox,
                           std::vector<cv::Point2f> &landmarks);
    bool estimateHeadPose(const std::vector<cv::Point2f> &landmarks,
                          cv::Vec3d &rvec, cv::Vec3d &tvec);
    bool detectPupilAndGlint(const cv::Mat &eyeRoi, cv::Point2f &pupilCenter,
                             cv::Point2f &glint, float &pupilDiameterPx);

    std::optional<cv::Point2f> estimateGazePoint(const cv::Point2f &pupilCenter,
                                                 const cv::Point2f &glint,
                                                 const cv::Vec3d &rvec,
                                                 const cv::Vec3d &tvec);
    void updateBlinkRate(bool eyeClosed);

    cv::CascadeClassifier faceDetector_;
    cv::Ptr<cv::face::Facemark> facemark_;

    std::deque<CalibrationSample> calibrationBuffer_;
    std::deque<int64_t> blinkTimestampsMs_;
    int64_t lastBlinkStateChangeMs_{0};
    bool lastEyeClosed_{false};
};

