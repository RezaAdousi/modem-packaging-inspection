
from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/runs/detect/modem_baseline/weights/best.pt")

    metrics = model.val(
        data="dataset/data.yaml",
        split="test",
        imgsz=640,
        batch=2,
        device=0
    )

    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
