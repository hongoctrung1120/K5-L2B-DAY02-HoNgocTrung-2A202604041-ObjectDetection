# Báo cáo cá nhân — Ngày 2 Object Detection

## 1. Thông tin học viên

- Họ tên: Hồ Ngọc Trung
- MSSV: 2A202604041
- Email: 26ai.trunghn@vinuni.edu.vn
- Lớp: K5 L2B
- Link repo bài làm:(https://github.com/hongoctrung1120/K5-L2B-DAY02-HoNgocTrung-2A202604041-ObjectDetection)
- Link Colab đã chạy: không có; pipeline chạy trên máy cá nhân (xem mục 5).

## 2. Schema và mini guideline

Chi tiết ở [GUIDELINE_MINI.md](../../GUIDELINE_MINI.md). Tên lớp và định nghĩa theo README (`car`, `truck`, `bus`, `van`). ID trong file nhãn nộp theo quy ước quan sát được ở gold: xe con = 0, xe van = 1, xe tải = 2, xe buýt = 3 (xem mục 4).

Các ca mơ hồ đã ghi:

- `drive_002`: xe thân hộp bị cắt ở rìa phải → `van`, box chạm rìa ảnh.
- `drive_009`: xe thùng kín màu xanh bị xe buýt che → `truck`, box quanh phần nhìn thấy.
- `drive_014`: xe cứu hộ công an → `truck`, cần giảng viên xác nhận.
- `drive_016`: xe trắng thân hộp nhỏ → `van`.

Phần cần giảng viên làm rõ: gold eval gán xe buýt lớn là ID 3 và xe tải thùng (`drive_051`) là ID 2, lệch với bảng schema trong README; gold eval cũng bỏ qua nhiều xe ở xa và xe đỗ ven đường.

## 3. Annotation và self-QC

- Số file nhãn train: 50 (827 box: 732 `car`, 16 `truck`, 77 `bus`, 2 `van`) theo tên lớp; nhãn sinh tự động, học viên sửa tay ở 6 ảnh trong CVAT.
- Kết quả checker và lỗi đã xử lý: `python tools/check_yolo_labels.py` báo `ĐẠT cấu trúc: 50/50 file nhãn hợp lệ`; không có lỗi định dạng.
- Danh sách ít nhất 10 ảnh đã rà bằng mắt: Học viên đã tự rà soát trực quan các ảnh từ drive_001 đến drive_018 (kết hợp xem trên ảnh vẽ sẵn bounding box và kiểm tra trên CVAT).
- Lỗi bỏ sót/box/class phát hiện khi rà và cách sửa: đã sửa lớp 3 box (`drive_002`, `drive_009`, `drive_016`). Các lỗi đã biết nhưng chưa sửa: thiếu lớp `van` (xe van bị gán `car`, ví dụ `drive_004`, `drive_008`); sót xe buýt lớn ở tiền cảnh `drive_009`; sót xe van đỏ ở `drive_014` và xe van cam ở `drive_016`.
- Commit khóa nhãn ban đầu: không có; repo git được tạo sau khi đã nhận gold.

## 4. Gold QC và rework

Cột "Trước sửa" là bản nháp sinh tự động theo ID của README; cột "Sau sửa" là sau khi đổi ID lớp và sửa tay trong CVAT:

| Chỉ số | Trước sửa | Sau sửa nếu có |
|---|---:|---:|
| TP | 441 | 499 |
| FP | 387 | 328 |
| FN | 212 | 154 |
| Precision | 0.5326 | 0.6034 |
| Recall | 0.6753 | 0.7642 |
| F1 | 0.5955 | 0.6743 |
| mean_iou_of_matches | 0.8532 | 0.8573 |
| passed | false | false |

Quan sát khi đối chiếu với gold (ghép box không xét lớp, IoU >= 0.50):

- 293 box không có box gold tương ứng, phần lớn là xe ở xa hoặc xe đỗ ven đường mà gold không gán (ví dụ `drive_028`: 13 box thừa, `drive_012`: 16).
- 118 box gold chưa được gán (ví dụ `drive_024`, `drive_027`, `drive_041`: 9 box mỗi ảnh).
- Lệch quy ước ID lớp: 47 box gán `bus` (2) khớp vị trí với gold ID 3; 11 box gán `truck` (1) khớp gold ID 2; 31 box gán `car` (0) khớp gold ID 1. Gold có vẻ dùng 1 = van, 2 = xe tải, 3 = xe buýt, khác bảng schema trong README.

Rework đã làm: đổi ID lớp của toàn bộ nhãn từ thứ tự README sang quy ước quan sát được ở gold (van = 1, xe tải = 2, xe buýt = 3) bằng `tools/remap_class_ids.py`. F1 tăng từ 0.5955 lên 0.6698. Sau đó học viên sửa tay trong CVAT local ở 6 ảnh: xóa 1 box (`drive_006`), chỉnh lại 2 box (`drive_009`, `drive_042`) và đổi lớp 3 box (`drive_012`, `drive_039`, `drive_043`); F1 lên 0.6743.

Bài chưa đạt ngưỡng F1 >= 0.80. Phần còn thiếu là xóa box thừa (328 FP) và vẽ box sót (154 FN). Kết quả lần đầu lưu ở `outputs/annotation_evaluation_initial.json` (không commit), kết quả cuối ở `outputs/annotation_evaluation.json`.

## 5. Kết quả YOLO

Pipeline được chạy trên máy cá nhân, không phải Colab, nên không có link Colab. Script chạy dùng đúng các tham số train/val/predict của `notebooks/day2_annotation_to_yolo.ipynb`. Bộ nhãn train lúc chạy chưa qua ngưỡng gold QC (F1 0.6743).

- Runtime, GPU/CPU, phiên bản Ultralytics và Torch: Windows 11, GPU NVIDIA GeForce RTX 3050 Ti Laptop; Ultralytics 8.4.208, Torch 2.6.0+cu124.
- Các tham số thay đổi so với notebook nếu có và lý do: thêm `workers=0` để tránh lỗi dataloader đa tiến trình trên Windows; `data.yaml` được chép ra bản tạm có đường dẫn tuyệt đối thay vì sửa file gốc. Còn lại giữ nguyên (`yolo26n.pt`, 50 epoch, `imgsz=640`, `batch=-1`, `seed=20260909`).
- Đường dẫn best checkpoint trong phiên chạy: `yolo26_runs/vehicle_annotation/weights/best.pt` (không commit).

| Chỉ số trên 10 ảnh eval | Giá trị |
|---|---:|
| Precision | 0.5748 |
| Recall | 0.5594 |
| mAP50 | 0.4813 |
| mAP50–95 | 0.3063 |

Theo từng ID lớp (tên trong `data.yaml` / loại xe thực tế theo quy ước gold):

| ID | Tên trong `data.yaml` | Loại xe thực tế | Precision | Recall | AP50 |
|---:|---|---|---:|---:|---:|
| 0 | car | xe con | 0.42 | 0.90 | 0.75 |
| 1 | truck | xe van | 1.00 | 0.00 | 0.13 |
| 2 | bus | xe tải | 0.24 | 0.50 | 0.25 |
| 3 | van | xe buýt | 0.63 | 0.83 | 0.79 |

Kết quả kèm theo: ảnh dự đoán ở [assets/eval_predictions/visuals/](assets/eval_predictions/visuals/), log train ở [assets/yolo26_runs/vehicle_annotation/results.csv](assets/yolo26_runs/vehicle_annotation/results.csv), ảnh chỉ số ở [assets/yolo26_runs/vehicle_eval/](assets/yolo26_runs/vehicle_eval/) (ví dụ [confusion matrix](assets/yolo26_runs/vehicle_eval/confusion_matrix.png)).

Ví dụ cụ thể: ở [drive_058](assets/eval_predictions/visuals/drive_058.jpg) model dự đoán 31 box ID 0 và 5 box ID 3, trong khi gold có 6 xe buýt (ID 3) xếp hàng bên trái. Ở [drive_060](assets/eval_predictions/visuals/drive_060.jpg) gold có một xe ID 1 ở phía xa giữa ảnh nhưng model không dự đoán box ID 1 nào.

Trả lời bốn câu hỏi cuối notebook:

1. Lớp recall thấp nhất là ID 1 (recall 0.00): model không dự đoán đúng xe van nào trên eval. Minh họa: `drive_060` và `drive_056` đều có xe ID 1 trong gold nhưng không có dự đoán ID 1. Bộ nhãn train chỉ có 2 box ID 1, trong khi gold train có 46.
2. mAP50 cao nhưng mAP50–95 thấp nghĩa là box tìm đúng vật thể nhưng chưa khít. Cần kiểm tra các cạnh box có lấy thừa bóng đổ/mặt đường hoặc cắt mất phần thân xe không, và box của xe bị che/cắt khung có nhất quán không. Ở bài này mAP50 là 0.48 và mAP50–95 là 0.31, nên vấn đề chính là sót và sai lớp chứ chưa phải độ khít.
3. Lỗi lớp liên quan guideline: xe van bị gán thành xe con. Dữ kiện: khi đối chiếu gold train, 31 box gán `car` trùng vị trí với box gold ID 1; precision của ID 0 trên eval chỉ 0.42. Ngoài ra thứ tự ID trong README lệch với gold, guideline phải ghi rõ quy ước ID đang dùng.
4. Không dùng 10 ảnh eval để train vì khi đó chỉ số đo trên chính dữ liệu model đã thấy, không còn phản ánh khả năng tổng quát. Trong notebook, 10 ảnh này là tập validation: Ultralytics đánh giá sau mỗi epoch và chọn `best.pt` theo chúng, nên đây không phải test set độc lập và chỉ số có phần lạc quan.

## 6. Khó khăn và cách xử lý

| Bước/thời điểm | Lỗi quan sát | Cách đã thử | Kết quả/còn thiếu |
|---|---|---|---|
| Cài CVAT local | Trang `localhost:8080` không tải được dù container chạy | Phát hiện `wslrelay` chiếm cổng 8080; tắt `localhostForwarding` trong `.wslconfig`, khởi động lại WSL và Docker | CVAT chạy được tại `http://localhost:8080` |
| Gán nhãn sơ bộ | Model pretrained COCO không có lớp `van` và bắt cả xe ở xa | Tăng ngưỡng confidence lên 0.6; thử thêm YOLO-World | Trên 10 ảnh eval chỉ đạt F1 0.66 (theo lớp), chưa qua ngưỡng 0.80; cần sửa tay |
| Đối chiếu schema | ID lớp trong gold eval lệch với README (xe buýt = 3) | Ghi vào guideline để hỏi giảng viên | Chưa có trả lời |

## 7. Tự đánh giá theo rubric

- Hoàn thành và định dạng YOLO: có 50 file, checker đạt.
- Đầy đủ vật thể, độ bám sát box, phân lớp: box khớp gold có mean IoU 0.8573, nhưng còn 328 box thừa, 154 box sót và gần như thiếu lớp xe van.
- Mini guideline: có định nghĩa lớp, quy tắc box và 4 ca mơ hồ.
- Gold-set QC: đã chạy, chưa đạt (F1 0.6743 < 0.80, mean IoU 0.8573 đạt); có rework đổi ID lớp và sửa tay 6 ảnh.
- Pipeline YOLO và phân tích: đã chạy trên máy cá nhân (không phải Colab), không dùng eval để train; có metrics, ảnh eval và nhận xét lỗi. Thiếu link Colab.

## 8. Công cụ AI và tài liệu tham khảo

- Công cụ AI đã dùng (Claude Code):
- Hỗ trợ giải thích khái niệm, tra cứu cú pháp Python/Ultralytics và đọc log - lỗi khi cài đặt môi trường (lỗi xung đột cổng WSL/Docker khi dựng CVAT local, lỗi workers của DataLoader trên Windows).
- Góp ý cách diễn đạt, rà soát chính tả và định dạng bảng biểu Markdown cho báo cáo và GUIDELINE_MINI.md.
- Phần học viên tự thực hiện và kiểm tra:
- Tự xây dựng quy tắc gán nhãn, quyết định các ca mơ hồ trong GUIDELINE_MINI.md, tự rà soát trực quan các ảnh và trực tiếp chỉnh sửa nhãn trên CVAT.
- Tự viết, chạy và kiểm tra đầu ra của các script hỗ trợ (tools/auto_label.py, tools/remap_class_ids.py) cũng như toàn bộ pipeline huấn luyện/đánh giá YOLO trên máy cá nhân.
- Tự đối chiếu số liệu thực nghiệm, chọn ảnh minh họa và viết phần phân tích kết quả cùng câu trả lời cho bốn câu hỏi ở mục 5.
- Tài liệu và nguồn tham khảo ngoài repo:
- Weights pretrained yolo11x.pt và tài liệu hướng dẫn của Ultralytics.
- Mã nguồn và tài liệu cài đặt CVAT v2.82.0.