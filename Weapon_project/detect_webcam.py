import argparse
import cv2
from ultralytics import YOLO
from utils import draw_box, FPS, put_fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best.pt",
                        help="Path to YOLO .pt weights (custom weapon model recommended).")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--weapons", type=str, default="gun,knife,pistol,handgun,rifle",
                        help="Comma-separated weapon class names to highlight")
    args = parser.parse_args()

    weapon_labels = set([x.strip().lower() for x in args.weapons.split(",") if x.strip()])

    model = YOLO(args.weights)  # works with YOLOv8 weights
    cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try --cam 1 or check camera permissions.")

    fps_counter = FPS()

    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO inference
        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
        r = results[0]
        names = r.names  # class id -> label

        # Draw detections
        if r.boxes is not None:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                label = str(names.get(cls_id, cls_id))
                conf = float(b.conf[0])
                xyxy = b.xyxy[0].tolist()

                # highlight only weapon classes
                if label.lower() in weapon_labels:
                    draw_box(frame, xyxy, label, conf, color=(0, 0, 255))  # red
                else:
                    # Optional: comment this out if you only want weapon boxes
                    # draw_box(frame, xyxy, label, conf, color=(255, 0, 0))  # blue
                    pass

        fps_value = fps_counter.update()
        put_fps(frame, fps_value)

        cv2.imshow("Weapon Detection (YOLO + OpenCV)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()