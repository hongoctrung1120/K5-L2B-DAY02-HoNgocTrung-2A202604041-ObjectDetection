#!/usr/bin/env python3
"""Gán nhãn sơ bộ (pre-annotation) bằng model pretrained, xuất nhãn YOLO theo schema của lab.

Schema: 0=car, 1=truck, 2=bus, 3=van.

Hai backend:
  coco   - YOLO pretrained trên COCO (car/truck/bus). COCO không có lớp van,
           nên backend này KHÔNG BAO GIỜ sinh lớp 3; van sẽ ra car/truck/bus.
  world  - YOLO-World (open-vocabulary) với prompt car/truck/bus/van; sinh đủ 4 lớp.

Nhãn sinh ra chỉ là bản nháp: phải mở ảnh preview để rà và sửa tay trước khi nộp.

Ví dụ:
  python tools/auto_label.py --preview outputs/auto_label_preview
  python tools/auto_label.py --backend world --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CLASS_NAMES = ["car", "truck", "bus", "van"]
COLORS = [(0, 200, 0), (0, 140, 255), (255, 80, 0), (200, 0, 200)]  # BGR
COCO_TO_SCHEMA = {2: 0, 7: 1, 5: 2}  # COCO car, truck, bus
DEFAULT_MODELS = {"coco": "yolo11x.pt", "world": "yolov8x-worldv2.pt"}


def load_model(backend: str, weights: str):
    if backend == "world":
        from ultralytics import YOLOWorld

        model = YOLOWorld(weights)
        model.set_classes(CLASS_NAMES)
        return model
    from ultralytics import YOLO

    return YOLO(weights)


def detect(model, backend: str, image: Path, args) -> list[tuple[int, float, float, float, float, float]]:
    """Trả về list (class_id, x_center, y_center, width, height, conf), tọa độ normalized."""
    result = model.predict(
        str(image),
        conf=args.conf,
        iou=args.nms_iou,
        imgsz=args.imgsz,
        agnostic_nms=True,  # một xe chỉ giữ một box, tránh trùng car/truck trên cùng vật thể
        classes=list(COCO_TO_SCHEMA) if backend == "coco" else None,
        max_det=300,
        verbose=False,
    )[0]
    height, width = result.orig_shape
    rows = []
    for xyxy, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        class_id = COCO_TO_SCHEMA[int(cls)] if backend == "coco" else int(cls)
        x1, y1, x2, y2 = xyxy
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        box_w, box_h = (x2 - x1) / width, (y2 - y1) / height
        if box_w * width < args.min_size or box_h * height < args.min_size:
            continue
        rows.append((class_id, (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, box_w, box_h, conf))
    return rows


def save_preview(image: Path, rows, out_dir: Path) -> None:
    import cv2

    frame = cv2.imread(str(image))
    height, width = frame.shape[:2]
    for class_id, xc, yc, w, h, conf in rows:
        p1 = (int((xc - w / 2) * width), int((yc - h / 2) * height))
        p2 = (int((xc + w / 2) * width), int((yc + h / 2) * height))
        cv2.rectangle(frame, p1, p2, COLORS[class_id], 2)
        cv2.putText(frame, f"{CLASS_NAMES[class_id]} {conf:.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[class_id], 1, cv2.LINE_AA)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / image.name), frame)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gán nhãn sơ bộ YOLO cho ảnh giao thông bằng model pretrained.")
    parser.add_argument("--images", type=Path, default=Path("dataset/images/train"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels/train"))
    parser.add_argument("--backend", choices=["coco", "world"], default="coco")
    parser.add_argument("--model", default=None, help="file weights; mặc định theo backend")
    parser.add_argument("--conf", type=float, default=0.25, help="ngưỡng confidence")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="ngưỡng IoU của NMS")
    parser.add_argument("--imgsz", type=int, default=1280, help="kích thước suy luận; lớn hơn giúp bắt xe nhỏ")
    parser.add_argument("--min-size", type=float, default=6.0, help="bỏ box có cạnh nhỏ hơn số pixel này")
    parser.add_argument("--preview", type=Path, default=None, help="thư mục lưu ảnh vẽ box để rà bằng mắt")
    parser.add_argument("--overwrite", action="store_true", help="cho phép ghi đè file nhãn đã có")
    args = parser.parse_args()

    images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in args.images.glob(ext))
    if not images:
        print(f"Không tìm thấy ảnh trong {args.images}")
        return 1
    existing = [p for p in images if (args.labels / f"{p.stem}.txt").exists()]
    if existing and not args.overwrite:
        print(f"Đã có {len(existing)} file nhãn trong {args.labels} (vd {existing[0].stem}.txt). "
              "Thêm --overwrite nếu muốn ghi đè.")
        return 1

    model = load_model(args.backend, args.model or DEFAULT_MODELS[args.backend])
    args.labels.mkdir(parents=True, exist_ok=True)
    counts = [0] * len(CLASS_NAMES)
    for image in images:
        rows = detect(model, args.backend, image, args)
        lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h, _ in rows]
        (args.labels / f"{image.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        for row in rows:
            counts[row[0]] += 1
        if args.preview:
            save_preview(image, rows, args.preview)
        print(f"{image.name}: {len(rows)} box")

    summary = ", ".join(f"{name}={count}" for name, count in zip(CLASS_NAMES, counts))
    print(f"\nXong {len(images)} ảnh -> {args.labels} ({summary})")
    if args.preview:
        print(f"Ảnh preview: {args.preview}")
    print("Đây là nhãn nháp: rà lại bằng mắt và sửa tay trước khi dùng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
