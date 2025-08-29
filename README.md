# YOLO Training & Inference Guide

## Part 1: Training

### Native (Local Machine)
> **Note:** Google Colab has limited training resources. Local training is recommended for larger or longer jobs.

#### Installation

```sh
pip install ultralytics
pip install roboflow
```

#### Import Data

```sh
python extract_data.py
```

#### Train the Model

```sh
python train.py
```

### Google Colab

- [Open Colab Notebook](https://colab.research.google.com/drive/1zVDxpoaXnky634DCkhevr_xOIBqnW3aO?usp=sharing)
- Use this notebook to generate the `.pt` model file.

---

## Part 2: Inference

Run live object detection using your webcam:

```sh
python live_object_detection_webcam.py
```

---

## Notes

- Make sure your data is properly extracted before training.
- Adjust script arguments as needed for your dataset