# 🎙️ VoiceStudio AI Pro — Lọc Tạp Âm & Đổi Giọng Lồng Tiếng Video

Ứng dụng Desktop chuyên dụng giúp cải thiện chất lượng giọng thu âm, khử tạp âm môi trường, biến đổi giọng nói chân thật (giữ nguyên cấu trúc thanh quản Formant) và ghép trực tiếp vào Video.

## 🚀 Cách khởi động nhanh
Nhấp đúp chuột vào file **[`Chay_Ung_Dung.bat`](file:///c:/Users/tuana/Desktop/Voice/Chay_Ung_Dung.bat)** trong thư mục để mở ngay giao diện ứng dụng.

Hoặc chạy bằng dòng lệnh:
```powershell
.\.venv\Scripts\python.exe app.py
```

---

## ✨ Các tính năng chính

### 1. Thu âm trực tiếp & Nhập xuất đa định dạng
- **Thu âm trực tiếp từ Microphone** kèm đồng hồ đo âm lượng thời gian thực và bộ đếm giờ.
- **Hỗ trợ mở trực tiếp cả file Audio và Video** (`.mp4`, `.mov`, `.mkv`, `.wav`, `.mp3`, `.m4a`...).
- **So sánh trực quan 2 biểu đồ sóng âm (Waveform)**: Bản thu gốc vs Bản đã xử lý Studio.

### 2. Làm sạch giọng & Tiện ích làm Video nhanh (Tab 2)
- **Khử tạp âm nhiều tầng**: tự dò & cắt tiếng ù điện 50/60 Hz cùng các hài (notch pha-không), khử nhiễu nền bằng cổng phổ với mẫu nhiễu lấy từ khoảng nghỉ thật giữa câu, rồi hạ phần nền còn sót trong các quãng nghỉ (có giữ trước/sau từ nên không mất chữ). Thêm tùy chọn **RNNoise (AI)** và chế độ lọc nhiễu biến thiên.
- **Khử vang phòng**: suy giảm đuôi vang muộn trong miền phổ (mô hình suy giảm mũ) kết hợp cắt các dải cộng hưởng phòng.
- **Chuẩn hóa -14 LUFS thật** (ITU-R BS.1770) và limiter đỉnh -1 dBFS đặt ở cuối chuỗi xử lý.
- **Tách lời nói khỏi nhạc nền (Vocal Isolation)**: Tự động tách giọng người nói ra khỏi nhạc nền có sẵn trong video.
- **Tự động cắt khoảng lặng / ngập ngừng (Auto Trim Silence)**: Tự động phát hiện và cắt bỏ các đoạn im lặng, đọc nghỉ quá lâu giữa các câu mà vẫn giữ lại nhịp thở tự nhiên.

### 3. Biến đổi giọng nói chân thật & Bàn trộn Phòng thu (Tab 1)
- **Thuật toán Formant-Preserving Pitch & Timbre Shift**: Thay đổi cao độ dây thanh (`Pitch`) độc lập với kích thước khoang thanh quản (`Formant`), giúp chuyển đổi giọng **Nam Trầm Ấm**, **Nữ Trong Trẻo**, **Kể Chuyện Điện Ảnh**, **Phát Thanh Viên Thời Sự**... chân thật, không bị méo tiếng kiểu "sóc chuột".
- **Bàn trộn Studio DSP**:
  - **3-Band Parametric EQ**: Tăng độ ấm (Bass 160Hz), độ rõ lời (Mid Presence 2.8kHz) và độ sáng (Treble Air 8.5kHz).
  - **Dynamic De-esser**: Làm dịu âm gió/chói (`s`, `x`, `ch`).
  - **Studio Compressor & Loudness Normalization**: Nén đều những chỗ nói to/nhỏ và chuẩn hóa âm lượng chuẩn YouTube / TikTok / CapCut (`-14 LUFS`).

### 4. AI Giọng Đọc Tự Nhiên & ElevenLabs Speech-to-Speech (Tab 3)
- **Đọc kịch bản bằng giọng AI Tiếng Việt miễn phí (Microsoft Edge Neural TTS)**: Hỗ trợ giọng `Nam Minh` (Nam trầm ấm) và `Hoài My` (Nữ truyền cảm) cùng các giọng đa ngôn ngữ chuẩn Studio.
- **ElevenLabs Speech-to-Speech Cloud API**: Chuyển đổi trực tiếp bản thu âm giọng thật sang giọng AI siêu thực nhưng giữ nguyên 100% cảm xúc và ngữ điệu.

### 5. Ghép trực tiếp vào Video gốc
- Nút **🎬 Ghép Giọng Đã Xử Lý Vào Video (.MP4)** tự động thay thế âm thanh mới vào file Video gốc bằng `FFmpeg` (`-c:v copy`) trong vài giây mà không làm giảm chất lượng hình ảnh.
