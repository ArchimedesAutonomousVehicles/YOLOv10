from ultralytics import YOLO

def main():
    model = YOLO("Model\yolov10n.pt") 

    train_results = model.train(
        data=r"Private--3\data.yaml", 
        imgsz=720,  
        #device="cpu", 
        workers=4
    )

    metrics = model.val()

    # # Perform object detection on an image
    # results = model("/content/Private--3/train/images/2024-11-04-102241-1-_jpg.rf.7e609a4d215fa0b383f4afb2bbaeeca7.jpg")  # Predict on an image
    # results[0].show()  # Display results

    path = model.export(format="onnx")

if __name__ == "__main__":
    main()

