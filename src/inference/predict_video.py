
from ultralytics import YOLO
from pathlib import Path
import cv2


def main():

    # 1. Load trained model
    model = YOLO(
        "runs/detect/runs/detect/modem_baseline/weights/best.pt"
    )

    # 2. Input video
    video_path = "test_videos/test_video.mp4"

    # 3. Run YOLO prediction
    results = model.predict(
        source=video_path,
        imgsz=640,
        conf=0.25,
        device=0,
        save=True,
        show=False,
        project="experiments/video_tests",
        name="test_001",
        exist_ok=False
    )

    # 4. Get actual output directory
    output_dir = Path(results[0].save_dir)

    print(f"\nYOLO output directory: {output_dir}")

    # 5. Find generated video
    output_videos = (
        list(output_dir.glob("*.avi")) +
        list(output_dir.glob("*.mp4"))
    )

    if not output_videos:
        raise FileNotFoundError(
            f"No output video found in: {output_dir}"
        )

    input_output = output_videos[0]

    # 6. Define MP4 output path
    mp4_output = output_dir / "test_video.mp4"

    # 7. Convert AVI to MP4 if necessary
    if input_output.suffix.lower() == ".mp4":

        print(f"MP4 already exists: {input_output}")

    else:

        cap = cv2.VideoCapture(str(input_output))

        if not cap.isOpened():
            raise RuntimeError("Could not open YOLO output video!")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Invalid video properties!")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(mp4_output),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create MP4 output!")

        frame_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            writer.write(frame)
            frame_count += 1

        cap.release()
        writer.release()

        print(f"\nConversion completed!")
        print(f"Frames converted: {frame_count}")

    # 8. Final result
    print("\nVideo prediction completed!")
    print(f"Output video: {mp4_output.resolve()}")


if __name__ == "__main__":
    main()
