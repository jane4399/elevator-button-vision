# Elevator Button Vision

Minimal repo to work with OCR-RCNN-v2 and elevator button datasets.

## Layout
- `third_party/ocr-rcnn-v2` (git submodule): upstream OCR/RCNN implementation
- `ElevatorButtonDataset/` (ignored): put datasets locally; not committed
- `venv-ocr/` (ignored): local Python env

## Quick start (macOS)
```bash
# clone with submodules
git clone --recurse-submodules https://github.com/jane4399/elevator-button-vision.git
cd elevator-button-vision

# Python env (CPU TF)
python3 -m venv venv-ocr && source venv-ocr/bin/activate
pip install --upgrade pip setuptools wheel
pip install tensorflow==2.20.0 numpy pillow matplotlib lxml imageio opencv-python

# Go to OCR-RCNN scripts
cd third_party/ocr-rcnn-v2/src/button_recognition/scripts/ocr_rcnn_lib

# Place frozen models here
# frozen_model/{detection_graph.pb, ocr_graph.pb, button_label_map.pbtxt}

# Put some images
mkdir -p test_panels
cp /absolute/path/to/ElevatorButtonDataset/icra2010/images/*.jpg ./test_panels/

# Run
python inference-visual.py
# or
python inference_640x480.py
```

## Notes
- Datasets and frozen models are excluded to keep the repo small.
- If submodule is empty after clone, run: `git submodule update --init --recursive`.
