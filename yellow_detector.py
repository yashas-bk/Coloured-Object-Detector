"""Standalone webcam color detector. Defaults to yellow.

Usage:
    python yellow_detector.py                 # detect yellow
    python yellow_detector.py --color red     # any category: red, orange, yellow,
                                              # green, cyan, blue, purple, pink,
                                              # white, gray, black
"""

import argparse

import cv2

from detection import COLOR_CATEGORIES, annotate, detect_category


def main():
    parser = argparse.ArgumentParser(description="Real-time color object detector")
    parser.add_argument(
        "--color",
        default="yellow",
        choices=list(COLOR_CATEGORIES),
        help="Color category to detect (default yellow)",
    )
    parser.add_argument("--min-area", type=int, default=500, help="Minimum contour area in px")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        detections, mask = detect_category(frame, args.color, args.min_area)
        annotated = annotate(frame, detections, label=args.color)

        cv2.imshow("Color Object Detector", annotated)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
