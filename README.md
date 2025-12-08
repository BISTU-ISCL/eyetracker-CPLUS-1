# Eye tracker calibration demo

This sample shows how to perform dark-pupil eye tracking calibration with OpenCV. It detects a face, extracts the eye region, finds the pupil center and corneal specular highlight (glint), estimates head pose for compensation, and maps the gaze to screen calibration points while also estimating pupil diameter and blink rate.

## Build

### 1. Install OpenCV with `opencv_contrib` (for `opencv2/face.hpp`)

The OpenCV face landmark API lives in `opencv_contrib`, so you need a build of OpenCV that includes those extra modules. A helper script is provided:

```bash
# From the repo root
./scripts/build_opencv_contrib.sh

# Optional environment overrides
# OPENCV_VERSION=4.8.1 PREFIX=/opt/opencv ./scripts/build_opencv_contrib.sh
# SRC_DIR=/tmp/opencv-src BUILD_DIR=/tmp/opencv-build ./scripts/build_opencv_contrib.sh
```

The script clones matching tags of `opencv` and `opencv_contrib`, configures CMake with `OPENCV_EXTRA_MODULES_PATH`, and installs to `/usr/local` by default. After installation, ensure `pkg-config` can find it (if needed):

```bash
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
```

If you install to a custom prefix, you can also point CMake at it explicitly:

```bash
cmake -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4 ..
```

### 2. Build the demo

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

> Missing `opencv2/face.hpp`? The facemark API lives in the **opencv_contrib** repository. Build OpenCV with the contrib modules or copy `face.hpp` from the official source: https://github.com/opencv/opencv_contrib/blob/4.x/modules/face/include/opencv2/face.hpp

A nine-point calibration grid is used by default. Press `r` to reset calibration or `ESC` to exit. The debug window shows:

- Face and eye overlays
- Pupil center (blue, detected via dark-pupil segmentation) and glint (yellow)
- Estimated gaze point (red)
- Blink frequency (Hz) and pupil diameter (pixels)

