import argparse
import cv2
from ultralytics import YOLO
from utils import draw_box

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best.pt")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--out", type=str, default="output.jpg", help="Path to save output image")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weapons", type=str, default="gun,knife,pistol,handgun,rifle")
    args = parser.parse_args()

    weapon_labels = set([x.strip().lower() for x in args.weapons.split(",") if x.strip()])
    model = YOLO(args.weights)

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError("Could not read image. Check path/format.")

    results = model.predict(img, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)
    r = results[0]
    names = r.names

    if r.boxes is not None:
        for b in r.boxes:
            cls_id = int(b.cls[0])
            label = str(names.get(cls_id, cls_id))
            conf = float(b.conf[0])
            xyxy = b.xyxy[0].tolist()

            if label.lower() in weapon_labels:
                draw_box(img, xyxy, label, conf, color=(0, 0, 255))

    cv2.imwrite(args.out, img)
    print(f"[DONE] Saved: {args.out}")

if __name__ == "__main__":
    main()