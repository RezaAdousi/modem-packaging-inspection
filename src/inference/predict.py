
from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/runs/detect/modem_baseline/weights/best.pt")

    model.predict(
        source="test_images/",
        conf=0.25,
        imgsz=640,
        device=0,
        save=True,
        show=False
    )

if __name__ == "__main__":
    main()
