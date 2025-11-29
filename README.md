# Eye tracker calibration demo

This sample shows how to perform bright-pupil eye tracking calibration with OpenCV. It detects a face, extracts the eye region, finds the pupil center and corneal specular highlight (glint), estimates head pose for compensation, and maps the gaze to screen calibration points while also estimating pupil diameter and blink rate.

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j
```

## Run

Download the OpenCV Haar cascade (e.g., `haarcascade_frontalface_default.xml`) and an LBF landmark model (e.g., `lbfmodel.yaml` from `opencv_contrib`). Then launch:

```bash
./eye_tracker <haar_cascade.xml> <lbfmodel.yaml>
```

A nine-point calibration grid is used by default. Press `r` to reset calibration or `ESC` to exit. The debug window shows:

- Face and eye overlays
- Pupil center (blue) and glint (yellow)
- Estimated gaze point (red)
- Blink frequency (Hz) and pupil diameter (pixels)

