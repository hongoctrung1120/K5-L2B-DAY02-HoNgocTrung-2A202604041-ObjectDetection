#!/usr/bin/env python3
"""Đổi ID lớp từ thứ tự README sang quy ước của gold labels.

README / CVAT project (car, truck, bus, van):  0=car, 1=truck, 2=bus, 3=van
Gold labels (quan sát từ eval và gold train):   0=car, 1=van,   2=truck, 3=bus

Script đọc nhãn theo thứ tự README ở --source và ghi bản đã đổi ID vào --output.
Không sửa tại chỗ, nên chạy lại nhiều lần từ cùng một nguồn vẫn ra cùng kết quả.

Ví dụ (sau khi export YOLO 1.1 từ CVAT và giải nén vào outputs/cvat_export):
  python tools/remap_class_ids.py --source outputs/cvat_export/obj_train_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

README_TO_GOLD = {0: 0, 1: 2, 2: 3, 3: 1}  # car->0, truck->2, bus->3, van->1


def main() -> int:
    parser = argparse.ArgumentParser(description="Đổi ID lớp README -> quy ước gold.")
    parser.add_argument("--source", type=Path, required=True, help="thư mục nhãn theo thứ tự README")
    parser.add_argument("--output", type=Path, default=Path("dataset/labels/train"))
    args = parser.parse_args()

    if args.source.resolve() == args.output.resolve():
        print("--source và --output phải khác nhau để tránh đổi ID hai lần.")
        return 1
    files = sorted(args.source.glob("drive_*.txt"))
    if not files:
        print(f"Không thấy file drive_*.txt trong {args.source}")
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {new: 0 for new in README_TO_GOLD.values()}
    for path in files:
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            new_id = README_TO_GOLD[int(parts[0])]
            counts[new_id] += 1
            lines.append(" ".join([str(new_id), *parts[1:]]))
        (args.output / path.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Đã đổi ID cho {len(files)} file -> {args.output}")
    print(f"Số box theo ID mới: 0(car)={counts[0]}, 1(van)={counts[1]}, 2(truck)={counts[2]}, 3(bus)={counts[3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
