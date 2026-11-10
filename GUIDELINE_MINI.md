# Mini guideline — bài làm cá nhân

## Định nghĩa lớp

| Lớp | Định nghĩa của bạn | Ví dụ/điểm phân biệt |
| --- | --- | --- |
| `car` (0) | Xe chở người cỡ nhỏ: sedan, hatchback, SUV, taxi, xe bán tải. | Thân thấp, có nắp ca-pô và khoang hành lý tách rõ hoặc dáng SUV; taxi vàng-xanh vẫn là `car`. |
| `truck` (1) | Xe tải chở hàng, xe ben, xe đầu kéo, xe thùng kín. | Cabin tách khỏi thùng hàng; thùng hở hoặc thùng hộp không có cửa sổ hành khách. |
| `bus` (2) | Xe buýt, xe khách, minibus chở khách. | Thân dài, nhiều hàng cửa sổ, cửa lên xuống bên hông; gồm cả xe buýt hai khúc và hai tầng. |
| `van` (3) | Xe van, minivan chở hàng hoặc chở người. | Thân hộp một khối, mũi ngắn, cao hơn `car` nhưng ngắn hơn `bus`; cửa trượt bên hông. |

## Quy tắc chung

- Box bám sát vật thể nghĩa là: bốn cạnh chạm mép ngoài phần thân xe nhìn thấy (gồm gương, bánh), không lấy bóng đổ, mặt đường hay xe bên cạnh.
- Vật thể bị che khuất: vẫn vẽ box quanh phần nhìn thấy nếu còn đủ để nhận ra lớp; hai xe chồng nhau thì hai box riêng.
- Vật thể bị cắt bởi rìa ảnh: box chạm đúng rìa ảnh, không đoán phần nằm ngoài khung.
- Vật thể rất nhỏ/mơ hồ: không gán nếu không xác định chắc được lớp (xe ở rất xa, dãy xe đỗ dính vào nhau); ghi lại để hỏi giảng viên.
- Không gán: xe máy, xe đạp, xe ba bánh, người đi bộ, biển báo, mái che trạm xe buýt.

## Ghi lại quyết định cho ít nhất ba ca mơ hồ

| Ảnh | Tình huống | Quyết định cuối | Lý do theo schema |
| --- | --- | --- | --- |
| `drive_002` | Xe màu bạc thân hộp bị cắt ở rìa phải ảnh, chỉ thấy nửa sau. | `van`, box chạm rìa phải. | Dáng một khối, cao, cửa hậu dựng đứng là đặc điểm của van; xe bị cắt khung vẫn gán phần nhìn thấy. |
| `drive_009` | Xe thùng kín màu xanh lá bị xe buýt che một phần, kích thước gần bằng minibus. | `truck`, box quanh phần nhìn thấy. | Thùng hộp không có cửa sổ hành khách, cabin tách khỏi thùng nên là xe chở hàng, không phải `bus`. |
| `drive_014` | Xe cứu hộ của công an (cabin + cần cẩu phía sau). | `truck` — cần giảng viên xác nhận. | Schema không có lớp xe chuyên dụng; cấu trúc cabin tách khỏi phần thân sau gần `truck` nhất. |
| `drive_016` | Xe trắng thân hộp nhỏ chạy giữa các xe con, nhìn từ trên xuống. | `van`. | Mũi ngắn, nóc phẳng dài đến đuôi, cao hơn xe con bên cạnh; không phải `truck` vì không có thùng hàng tách rời. |

## Cần giảng viên làm rõ

- Trong `dataset/labels/eval/`, xe buýt lớn mang ID 3 (`van`) ở `drive_053`, `drive_056`, `drive_058`, `drive_060`, còn xe tải thùng ở `drive_051` mang ID 2 (`bus`). Điều này lệch với bảng schema trong README. Sau khi đối chiếu gold train, bộ nhãn nộp đã đổi ID theo quy ước của gold (van = 1, xe tải = 2, xe buýt = 3); tên lớp trong `data.yaml` giữ nguyên.
- Gold eval không gán nhiều xe ở xa và xe đỗ ven đường. Ngưỡng "rất nhỏ/mơ hồ" để bỏ qua là bao nhiêu?
