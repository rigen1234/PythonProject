import time
import cv2

def draw_box(img, xyxy, label, conf, color=(0, 0, 255)):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

class FPS:
    def __init__(self):
        self.t0 = time.time()
        self.frames = 0
        self.fps = 0.0

    def update(self):
        self.frames += 1
        dt = time.time() - self.t0
        if dt >= 1.0:
            self.fps = self.frames / dt
            self.t0 = time.time()
            self.frames = 0
        return self.fps

def put_fps(img, fps_value):
    cv2.putText(img, f"FPS: {fps_value:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)