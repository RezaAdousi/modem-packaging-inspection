
from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        workers=2,
        device=0,
        project="runs/detect",
        name="modem_baseline",
        patience=20,
        plots=True
    )

    print("Training completed!")

if __name__ == "__main__":
    main()
