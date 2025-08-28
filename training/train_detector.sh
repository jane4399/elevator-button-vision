#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./training/train_detector.sh \
#     /Users/janexie/Documents/robo/elevator_button/ElevatorButtonDataset/iros2018/train_set/images \
#     /Users/janexie/Documents/robo/elevator_button/ElevatorButtonDataset/iros2018/train_set/annotations
# Outputs will go under training/work/

IMAGES_DIR=${1:?images dir}
ANN_DIR=${2:?annotations dir}
WORK_ROOT=$(cd "$(dirname "$0")" && pwd)
PROJ_ROOT=$(cd "$WORK_ROOT/.." && pwd)

mkdir -p "$WORK_ROOT/work/tfrecords" "$WORK_ROOT/work/configs" "$WORK_ROOT/work/training"

# Create train/val lists
python3 "$WORK_ROOT/make_lists.py" --images "$IMAGES_DIR" --annotations "$ANN_DIR" --out_dir "$WORK_ROOT/work"

# Pull TF1 docker
DOCKER_IMAGE=tensorflow/tensorflow:1.15.5-py3

# Run inside docker
docker run --rm -it \
  -v "$PROJ_ROOT":/work \
  -w /work/training \
  $DOCKER_IMAGE bash -lc '
set -e
apt-get update && apt-get install -y --no-install-recommends git protobuf-compiler python3-opencv && rm -rf /var/lib/apt/lists/*
pip install --no-cache-dir tf-slim contextlib2 lxml Cython pillow matplotlib
# Get models if missing
if [ ! -d /work/training/models ]; then
  git clone https://github.com/tensorflow/models.git
fi
cd models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:/work/training/models/research:/work/training/models/research/slim

# Build TFRecords
python object_detection/dataset_tools/create_pascal_tf_record.py \
  --data_dir '"$IMAGES_DIR"'/.. \
  --set train \
  --annotations_dir annotations \
  --year VOC2012 \
  --output_path /work/training/work/tfrecords/train.record \
  --label_map_path /work/training/label_map.pbtxt \
  --image_dir images \
  --examples_path /work/training/work/train.txt

python object_detection/dataset_tools/create_pascal_tf_record.py \
  --data_dir '"$IMAGES_DIR"'/.. \
  --set val \
  --annotations_dir annotations \
  --year VOC2012 \
  --output_path /work/training/work/tfrecords/val.record \
  --label_map_path /work/training/label_map.pbtxt \
  --image_dir images \
  --examples_path /work/training/work/val.txt

# Prepare config
cat > /work/training/work/configs/faster_rcnn_resnet50.config <<CFG
model {
  faster_rcnn {
    num_classes: 1
    image_resizer { fixed_shape_resizer { height: 480 width: 640 } }
    feature_extractor { type: "faster_rcnn_resnet50" }
  }
}
train_config {
  batch_size: 2
  num_steps: 20000
  fine_tune_checkpoint: "/work/training/work/pretrained/model.ckpt"
  from_detection_checkpoint: true
}
train_input_reader {
  label_map_path: "/work/training/label_map.pbtxt"
  tf_record_input_reader { input_path: "/work/training/work/tfrecords/train.record" }
}
eval_config { num_examples: 100 }
eval_input_reader {
  label_map_path: "/work/training/label_map.pbtxt"
  tf_record_input_reader { input_path: "/work/training/work/tfrecords/val.record" }
}
CFG

# Download pretrained checkpoint
mkdir -p /work/training/work/pretrained
cd /work/training/work/pretrained
if [ ! -f model.ckpt.index ]; then
  curl -L -o ckpt.tar.gz https://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
  tar -xzf ckpt.tar.gz --strip-components=1
  mv model.ckpt.data-00000-of-00001 frozen_inference_graph.pb graph.pbtxt model.ckpt.index model.ckpt.meta /work/training/work/pretrained/ 2>/dev/null || true
fi

# Train
python /work/training/models/research/object_detection/model_main.py \
  --pipeline_config_path /work/training/work/configs/faster_rcnn_resnet50.config \
  --model_dir /work/training/work/training \
  --num_train_steps 20000 \
  --sample_1_of_n_eval_examples 1

# Export frozen graph
python /work/training/models/research/object_detection/export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path /work/training/work/configs/faster_rcnn_resnet50.config \
  --trained_checkpoint_prefix /work/training/work/training/model.ckpt-20000 \
  --output_directory /work/training/work/exported
'

echo "Training complete. Exported graph at training/work/exported/frozen_inference_graph.pb"
