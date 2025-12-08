#include "EyeTrackerCalibrator.hpp"

#include <chrono>
#include <numeric>

// Implementation of the calibration pipeline. The logic follows these steps:
// 1. Detect the face and fit 68 landmarks using the LBF model.
// 2. Compute head pose (rotation/translation) from six landmarks.
// 3. Crop an eye ROI, detect the pupil (dark blob) and glint (bright blob).
// 4. Update blink statistics using eye-aspect ratio.
// 5. Collect calibration samples and regress pupil/glint offsets to gaze (dark-pupil setup).

using Clock = std::chrono::steady_clock;

namespace {
constexpr double kEyeFovDegrees = 55.0; // approximate horizontal field of view
constexpr double kIrisToCorneaMm = 4.5; // distance to approximate glint offset

std::vector<cv::Point3d> modelLandmarks()
{
    // Simple 3D facial landmark model (chin, nose, eye corners, mouth corners)
    return {
        {0.0, 0.0, 0.0},            // nose tip
        {0.0, -330.0, -65.0},       // chin
        {-225.0, 170.0, -135.0},    // left eye left corner
        {225.0, 170.0, -135.0},     // right eye right corner
        {-150.0, -150.0, -125.0},   // left mouth corner
        {150.0, -150.0, -125.0}     // right mouth corner
    };
}
}

EyeTrackerCalibrator::EyeTrackerCalibrator()
{
}

bool EyeTrackerCalibrator::loadModels(const std::string &faceDetectorPath,
                                      const std::string &landmarkModelPath)
{
    // Load Haar cascade for face detection.
    if (!faceDetector_.load(faceDetectorPath)) {
        return false;
    }

    // Load the LBF facemark model (requires opencv_contrib's face module).
    facemark_ = cv::face::FacemarkLBF::create();
    if (!facemark_->loadModel(landmarkModelPath)) {
        facemark_.release();
        return false;
    }

    return true;
}

bool EyeTrackerCalibrator::detectFace(const cv::Mat &gray, cv::Rect &faceBox)
{
    // Find all faces and pick the largest (closest to the camera).
    std::vector<cv::Rect> faces;
    faceDetector_.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(80, 80));
    if (faces.empty()) {
        return false;
    }
    faceBox = *std::max_element(faces.begin(), faces.end(),
                                [](const auto &a, const auto &b) { return a.area() < b.area(); });
    return true;
}

bool EyeTrackerCalibrator::estimateLandmarks(const cv::Mat &gray, const cv::Rect &faceBox,
                                             std::vector<cv::Point2f> &landmarks)
{
    // Run landmark fitting constrained to the detected face box.
    std::vector<std::vector<cv::Point2f>> shapes;
    if (!facemark_ || !facemark_->fit(gray, {faceBox}, shapes) || shapes.empty()) {
        return false;
    }
    landmarks = shapes.front();
    return true;
}

bool EyeTrackerCalibrator::estimateHeadPose(const std::vector<cv::Point2f> &landmarks,
                                            cv::Vec3d &rvec, cv::Vec3d &tvec)
{
    // Using six key landmarks matching modelLandmarks order
    if (landmarks.size() < 68) {
        return false;
    }

    std::vector<cv::Point2d> imagePoints = {
        landmarks[30], // nose tip
        landmarks[8],  // chin
        landmarks[36], // left eye left corner
        landmarks[45], // right eye right corner
        landmarks[48], // left mouth corner
        landmarks[54]  // right mouth corner
    };

    auto model = modelLandmarks();

    const double focalLength = 800.0;
    cv::Point2d center(320, 240);
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << focalLength, 0, center.x,
                              0, focalLength, center.y,
                              0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    // SolvePnP returns the rotation/translation aligning model points to image.
    return cv::solvePnP(model, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
}

bool EyeTrackerCalibrator::detectPupilAndGlint(const cv::Mat &eyeRoi, cv::Point2f &pupilCenter,
                                               cv::Point2f &glint, float &pupilDiameterPx)
{
    // Normalize contrast to make thresholding more stable and reduce noise so
    // dark-pupil segmentation does not react to eyelashes or eyebrows.
    cv::Mat normalized;
    cv::equalizeHist(eyeRoi, normalized);
    cv::Mat blurred;
    cv::GaussianBlur(normalized, blurred, cv::Size(5, 5), 0);

    // Pupil: dark blob detection using Otsu to adapt to illumination changes
    // typical in dark-pupil (non retro-reflective) imaging.
    cv::Mat binary;
    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return false;
    }

    auto largest = *std::max_element(contours.begin(), contours.end(),
                                     [](const auto &a, const auto &b) { return cv::contourArea(a) < cv::contourArea(b); });
    cv::RotatedRect ellipse = cv::fitEllipse(largest);
    pupilCenter = ellipse.center;
    pupilDiameterPx = static_cast<float>((ellipse.size.width + ellipse.size.height) * 0.5f);

    // Glint: bright spot detection using thresholding
    // Use the unblurred equalized frame to preserve specular peaks while
    // operating under dark-pupil illumination.
    cv::threshold(normalized, binary, 230, 255, cv::THRESH_BINARY);
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (!contours.empty()) {
        auto bright = *std::max_element(contours.begin(), contours.end(),
                                        [](const auto &a, const auto &b) { return cv::contourArea(a) < cv::contourArea(b); });
        cv::Moments m = cv::moments(bright);
        glint = cv::Point2f(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
    } else {
        glint = cv::Point2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    }
    return true;
}

std::optional<cv::Point2f> EyeTrackerCalibrator::estimateGazePoint(const cv::Point2f &pupilCenter,
                                                                   const cv::Point2f &glint,
                                                                   const cv::Vec3d &rvec,
                                                                   const cv::Vec3d &tvec)
{
    if (calibrationBuffer_.size() < 4) {
        return std::nullopt;
    }

    // Compute eye vector using pupil-glint offset (dark-pupil design)
    cv::Point2f offset = pupilCenter - glint;
    double angleX = (offset.x / 320.0) * (kEyeFovDegrees * CV_PI / 180.0);
    double angleY = (offset.y / 240.0) * (kEyeFovDegrees * CV_PI / 180.0);

    cv::Mat rotMat;
    cv::Rodrigues(rvec, rotMat);
    cv::Mat gazeVec = (cv::Mat_<double>(3, 1) << std::sin(angleX), std::sin(angleY), 1.0);
    gazeVec = rotMat * gazeVec;

    // Linear regression using calibration samples
    cv::Mat A, B;
    for (const auto &sample : calibrationBuffer_) {
        cv::Mat sampleRot;
        cv::Rodrigues(sample.headRotation, sampleRot);
        cv::Mat sampleVec = (cv::Mat_<double>(3, 1) <<
                             sample.pupilCenter.x - sample.glint.x,
                             sample.pupilCenter.y - sample.glint.y,
                             1.0);
        sampleVec = sampleRot * sampleVec;
        A.push_back(sampleVec.t());
        B.push_back((cv::Mat_<double>(1, 2) << sample.screenPoint.x, sample.screenPoint.y));
    }

    cv::Mat coeffs;
    cv::solve(A, B, coeffs, cv::DECOMP_SVD);
    cv::Mat predicted = (gazeVec.t() * coeffs).reshape(1, 1);
    return cv::Point2f(static_cast<float>(predicted.at<double>(0, 0)),
                       static_cast<float>(predicted.at<double>(0, 1)));
}

void EyeTrackerCalibrator::updateBlinkRate(bool eyeClosed)
{
    const int64_t nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
    if (eyeClosed != lastEyeClosed_) {
        if (!eyeClosed && lastEyeClosed_) {
            blinkTimestampsMs_.push_back(nowMs);
        }
        lastEyeClosed_ = eyeClosed;
        lastBlinkStateChangeMs_ = nowMs;
    }

    // keep last 60 seconds
    const int64_t windowMs = 60000;
    while (!blinkTimestampsMs_.empty() && nowMs - blinkTimestampsMs_.front() > windowMs) {
        blinkTimestampsMs_.pop_front();
    }
}

bool EyeTrackerCalibrator::processFrame(const cv::Mat &frame,
                                        const std::vector<cv::Point2f> &screenCalibrationTargets,
                                        cv::Point2f &estimatedGaze,
                                        PupilMetrics &metrics,
                                        cv::Mat &debugFrame)
{
    if (frame.empty()) return false;

    // Convert to grayscale for detection pipelines.
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 1) Face detection.
    cv::Rect faceBox;
    if (!detectFace(gray, faceBox)) return false;

    // 2) Landmark estimation.
    std::vector<cv::Point2f> landmarks;
    if (!estimateLandmarks(gray, faceBox, landmarks)) return false;

    // 3) Head pose from a minimal set of landmarks.
    cv::Vec3d rvec, tvec;
    if (!estimateHeadPose(landmarks, rvec, tvec)) return false;

    // extract right eye region (landmarks 36-41) as example
    cv::Rect eyeRect = cv::boundingRect(std::vector<cv::Point2f>{landmarks.begin() + 36, landmarks.begin() + 42});
    eyeRect &= cv::Rect(0, 0, frame.cols, frame.rows);
    cv::Mat eyeRoi = gray(eyeRect);

    // 4) Detect pupil center, glint, and pupil size.
    cv::Point2f pupil, glint;
    float diameter = 0.f;
    if (!detectPupilAndGlint(eyeRoi, pupil, glint, diameter)) return false;

    // Convert ROI coordinates back to full image space
    pupil += cv::Point2f(static_cast<float>(eyeRect.x), static_cast<float>(eyeRect.y));
    glint += cv::Point2f(static_cast<float>(eyeRect.x), static_cast<float>(eyeRect.y));

    // Eye closure estimation using aspect ratio on eye landmarks
    const double eyeHeight = cv::norm(landmarks[37] - landmarks[41]) + cv::norm(landmarks[38] - landmarks[40]);
    const double eyeWidth = cv::norm(landmarks[36] - landmarks[39]);
    const double ear = eyeHeight / (2.0 * eyeWidth);
    const bool eyeClosed = ear < 0.18;
    updateBlinkRate(eyeClosed);

    metrics.diameterPx = diameter;
    const double windowSec = 60.0;
    metrics.blinkRateHz = blinkTimestampsMs_.empty() ? 0.0f : static_cast<float>(blinkTimestampsMs_.size() / windowSec);

    // collect calibration samples when user looks at known targets
    size_t maxSamples = screenCalibrationTargets.size();
    if (maxSamples > 0 && calibrationBuffer_.size() < maxSamples) {
        calibrationBuffer_.push_back({pupil, glint, screenCalibrationTargets[calibrationBuffer_.size()], rvec, tvec});
    }

    // 5) Predict gaze using accumulated calibration and current observation.
    auto gaze = estimateGazePoint(pupil, glint, rvec, tvec);
    if (!gaze) return false;
    estimatedGaze = *gaze;

    debugFrame = frame.clone();
    cv::rectangle(debugFrame, faceBox, {0, 255, 0}, 2);
    cv::circle(debugFrame, pupil, 3, {255, 0, 0}, -1);
    cv::circle(debugFrame, glint, 3, {0, 255, 255}, -1);
    cv::putText(debugFrame, "Blink Hz: " + std::to_string(metrics.blinkRateHz), {10, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
    cv::putText(debugFrame, "Pupil px: " + std::to_string(metrics.diameterPx), {10, 45}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
    cv::putText(debugFrame, "Gaze: (" + std::to_string(estimatedGaze.x) + "," + std::to_string(estimatedGaze.y) + ")",
                {10, 70}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);

    return true;
}

void EyeTrackerCalibrator::resetCalibration()
{
    calibrationBuffer_.clear();
}

