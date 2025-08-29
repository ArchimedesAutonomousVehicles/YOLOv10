import cv2
import numpy as np
import pyzed.sl as sl
import onnxruntime as ort

# Initialize ZED
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit()

# Load ONNX model
session = ort.InferenceSession(r"C:\Users\User\Desktop\AAV\data\best_292_runs.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Get input shape for resizing
input_shape = session.get_inputs()[0].shape  # [1, 3, H, W]
input_height = input_shape[2]
input_width = input_shape[3]

cv2.namedWindow("YOLOv10n + ZED", cv2.WINDOW_NORMAL)

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve image from ZED
        img_zed = sl.Mat()
        zed.retrieve_image(img_zed, sl.VIEW.LEFT)
        img_cv = img_zed.get_data()

        # Resize and preprocess image
        img_resized = cv2.resize(img_cv, (input_width, input_height))
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32)  # HWC to CHW
        img_input = np.expand_dims(img_input, axis=0) / 255.0  # Normalize to 0-1

        # Inference
        outputs = session.run([output_name], {input_name: img_input})[0]

        # Confirm output shape (debug)
        # print(outputs.shape)  # (1, N, 6) or (N, 6)

        detections = outputs[0] if len(outputs.shape) == 3 else outputs  # handle batch dim
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            if conf < 0.4:  # confidence threshold
                continue

            # Scale boxes back to original image size
            x_scale = img_cv.shape[1] / input_width
            y_scale = img_cv.shape[0] / input_height

            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)

            # Draw box and label
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {int(cls_id)}: {conf:.2f}"
            cv2.putText(img_cv, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("YOLOv10n + ZED", img_cv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()
cv2.destroyAllWindows()
