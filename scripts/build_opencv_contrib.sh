#!/usr/bin/env bash
# Build and install OpenCV with the opencv_contrib modules (required for opencv2/face.hpp).
# Usage:
#   OPENCV_VERSION=4.8.1 PREFIX=/usr/local ./scripts/build_opencv_contrib.sh
# Environment variables:
#   OPENCV_VERSION: Git tag or branch to checkout (default: 4.8.1)
#   PREFIX: Install prefix passed to CMake (default: /usr/local)
#   SRC_DIR: Directory to store cloned sources (default: ./.opencv-src)
#   BUILD_DIR: Directory to hold the build tree (default: ./.opencv-build)
set -euo pipefail

OPENCV_VERSION="${OPENCV_VERSION:-4.8.1}"
PREFIX="${PREFIX:-/usr/local}"
SRC_DIR="${SRC_DIR:-$(pwd)/.opencv-src}"
BUILD_DIR="${BUILD_DIR:-$(pwd)/.opencv-build}"

OPENCV_SRC="$SRC_DIR/opencv"
OPENCV_CONTRIB_SRC="$SRC_DIR/opencv_contrib"

# Ensure required tools are available before starting the long build.
for tool in git cmake; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required tool: $tool" >&2
    exit 1
  fi
done

mkdir -p "$SRC_DIR" "$BUILD_DIR"

clone_repo() {
  local url=$1
  local dest=$2
  local ref=$3

  if [ -d "$dest/.git" ]; then
    echo "Updating existing repository $dest to $ref ..."
    git -C "$dest" fetch --tags origin "$ref"
    git -C "$dest" checkout "$ref"
  else
    echo "Cloning $url@$ref into $dest ..."
    git clone --depth 1 --branch "$ref" "$url" "$dest"
  fi
}

clone_repo https://github.com/opencv/opencv "$OPENCV_SRC" "$OPENCV_VERSION"
clone_repo https://github.com/opencv/opencv_contrib "$OPENCV_CONTRIB_SRC" "$OPENCV_VERSION"

# Configure the build with contrib modules, skipping extra components to speed things up.
cmake -S "$OPENCV_SRC" -B "$BUILD_DIR" \
  -DOPENCV_EXTRA_MODULES_PATH="$OPENCV_CONTRIB_SRC/modules" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DOPENCV_GENERATE_PKGCONFIG=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_opencv_world=OFF

# Build and install.
cmake --build "$BUILD_DIR" -j"$(nproc)"
cmake --install "$BUILD_DIR"

echo
pkgconfig_path="$PREFIX/lib/pkgconfig"
cat <<EONOTE
OpenCV with contrib modules has been installed to $PREFIX.
If pkg-config cannot find it, export:
  export PKG_CONFIG_PATH="$pkgconfig_path:$PKG_CONFIG_PATH"
Then rebuild this project so CMake can locate opencv2/face.hpp.
EONOTE
