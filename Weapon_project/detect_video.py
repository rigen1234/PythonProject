import argparse
import cv2
from ultralytics import YOLO
from utils import draw_box, FPS, put_fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best.pt")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--out", type=str, default="output.mp4", help="Path to save output video")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weapons", type=str, default="gun,knife,pistol,handgun,rifle")
    args = parser.parse_args()

    weapon_labels = set([x.strip().lower() for x in args.weapons.split(",") if x.strip()])
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file. Check the path.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps_in, (w, h))

    fps_counter = FPS()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
        r = results[0]
        names = r.names

        if r.boxes is not None:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                label = str(names.get(cls_id, cls_id))
                conf = float(b.conf[0])
                xyxy = b.xyxy[0].tolist()

                if label.lower() in weapon_labels:
                    draw_box(frame, xyxy, label, conf, color=(0, 0, 255))

        fps_value = fps_counter.update()
        put_fps(frame, fps_value)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[DONE] Saved: {args.out}")

if __name__ == "__main__":
    main()