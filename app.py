import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QLinearGradient, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QCheckBox,
    QTabWidget,
    QFileDialog,
    QProgressBar,
    QMessageBox,
    QFrame,
    QTextEdit,
    QLineEdit,
    QGroupBox,
    QSplitter,
    QScrollArea,
)

from audio_engine import (
    AudioEngine,
    VoiceSettings,
    VOICE_CATEGORIES,
    VOICE_PRESETS,
    EDGE_TTS_VOICES,
    ELEVENLABS_VOICES,
)


# =============================================================================
# WIDGET VẼ SÓNG ÂM (WAVEFORM VISUALIZER)
# =============================================================================
class WaveformWidget(QWidget):
    def __init__(self, title: str, accent_color: str = "#38bdf8", parent=None):
        super().__init__(parent)
        self.title = title
        self.accent_color = QColor(accent_color)
        self.peaks: Optional[np.ndarray] = None
        self.duration_sec: float = 0.0
        self.rms_db: float = -60.0
        self.peak_db: float = -60.0
        self.setMinimumHeight(105)

    def set_audio(self, audio: Optional[np.ndarray], sr: int):
        if audio is None or len(audio) == 0:
            self.peaks = None
            self.duration_sec = 0.0
            self.update()
            return

        self.duration_sec = len(audio) / float(sr)
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-8)
        peak = float(np.max(np.abs(audio)) + 1e-8)
        self.rms_db = 20.0 * np.log10(rms)
        self.peak_db = 20.0 * np.log10(peak)

        num_bars = 260
        if len(audio) < num_bars:
            self.peaks = np.abs(audio)
        else:
            chunk = len(audio) // num_bars
            trimmed = audio[: chunk * num_bars].reshape(num_bars, chunk)
            self.peaks = np.max(np.abs(trimmed), axis=1)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        # Nền khung
        bg_grad = QLinearGradient(0, 0, 0, rect.height())
        bg_grad.setColorAt(0.0, QColor("#0f172a"))
        bg_grad.setColorAt(1.0, QColor("#1e293b"))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 10, 10)

        # Đường trung tâm
        mid_y = rect.height() / 2.0 + 8
        painter.setPen(QPen(QColor("#334155"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(12, int(mid_y), rect.width() - 12, int(mid_y))

        # Tiêu đề & thông số kỹ thuật
        painter.setPen(QColor("#e2e8f0"))
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(14, 22, self.title)

        if self.peaks is not None and len(self.peaks) > 0:
            info_text = f"Thời lượng: {self.duration_sec:.2f}s   |   Độ lớn TB: {self.rms_db:.1f} dB   |   Đỉnh: {self.peak_db:.1f} dB"
            painter.setPen(QColor("#94a3b8"))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(rect.width() - 375, 22, info_text)

            # Vẽ các cột sóng âm
            draw_w = rect.width() - 28
            draw_h = (rect.height() - 38) * 0.46
            n = len(self.peaks)
            bar_w = max(1.5, draw_w / float(n) - 1.2)

            painter.setPen(Qt.PenStyle.NoPen)
            for i, val in enumerate(self.peaks):
                x = 14 + i * (draw_w / float(n))
                h = max(2.0, min(1.0, float(val)) * draw_h)
                bar_color = QColor(self.accent_color)
                if val > 0.92:
                    bar_color = QColor("#f43f5e")
                painter.setBrush(bar_color)
                painter.drawRoundedRect(QRectF(x, mid_y - h, bar_w, h * 2.0), 1.5, 1.5)
        else:
            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Chưa có dữ liệu âm thanh — Hãy Thu âm, Mở file hoặc Tạo giọng mẫu")


# =============================================================================
# LUỒNG XỬ LÝ NỀN (BACKGROUND WORKER THREADS)
# =============================================================================
class ProcessWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, engine: AudioEngine, audio: np.ndarray, sr: int, settings: VoiceSettings):
        super().__init__()
        self.engine = engine
        self.audio = audio
        self.sr = sr
        self.settings = settings

    def run(self):
        try:
            out = self.engine.process_voice(
                self.audio,
                self.sr,
                self.settings,
                progress_cb=lambda p, m: self.progress.emit(p, m),
            )
            self.finished_ok.emit(out)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class AIWorker(QThread):
    finished_ok = pyqtSignal(object, int, str)
    failed = pyqtSignal(str)

    def __init__(self, mode: str, engine: AudioEngine, kwargs: dict):
        super().__init__()
        self.mode = mode
        self.engine = engine
        self.kwargs = kwargs

    def run(self):
        try:
            if self.mode == "edge_tts":
                audio, sr = self.engine.generate_edge_tts(**self.kwargs)
                self.finished_ok.emit(audio, sr, "Đã tạo giọng đọc AI Neural thành công!")
            elif self.mode == "ai_neural_sts":
                audio, sr, rec_text = self.engine.convert_voice_ai_neural_sts(**self.kwargs)
                self.finished_ok.emit(audio, sr, rec_text)
            elif self.mode == "elevenlabs":
                audio, sr = self.engine.convert_voice_elevenlabs_sts(**self.kwargs)
                self.finished_ok.emit(audio, sr, "Đã chuyển đổi giọng bằng ElevenLabs AI thành công!")
        except Exception as e:
            self.failed.emit(str(e))


# =============================================================================
# CỬA SỔ CHÍNH CỦA ỨNG DỤNG (MAIN STUDIO WINDOW)
# =============================================================================
class VoiceStudioApp(QMainWindow):
    sig_mic_level = pyqtSignal(float)
    sig_play_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎙️ VoiceStudio AI Pro — Lọc Tạp Âm & Biến Đổi Giọng Nói Cho Video")
        self.resize(1240, 820)

        self.engine = AudioEngine(target_sr=44100)
        self.settings = VoiceSettings()

        self.orig_audio: Optional[np.ndarray] = None
        self.base_proc_audio: Optional[np.ndarray] = None
        self.proc_audio: Optional[np.ndarray] = None
        self.current_sr: int = 44100
        self.loaded_video_path: Optional[str] = None
        self.record_start_time: float = 0.0

        self.rec_timer = QTimer(self)
        self.rec_timer.timeout.connect(self._update_record_timer)
        self.sig_mic_level.connect(self._on_mic_level)

        self.play_ui_timer = QTimer(self)
        self.play_ui_timer.timeout.connect(self._update_playback_buttons_ui)
        self.sig_play_finished.connect(self._on_playback_finished_ui)

        self._apply_dark_stylesheet()
        self._build_ui()
        self._populate_devices()

    def _apply_dark_stylesheet(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #0b1120;
                color: #f1f5f9;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                background-color: #111827;
                border: 1px solid #1e293b;
                border-radius: 10px;
                margin-top: 10px;
                padding: 12px;
                font-weight: bold;
                color: #38bdf8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px 14px;
                color: #f8fafc;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #475569;
            }
            QPushButton:disabled {
                background-color: #0f172a;
                color: #475569;
                border-color: #1e293b;
            }
            QComboBox, QLineEdit, QTextEdit {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 7px;
                padding: 6px 10px;
                color: #f8fafc;
            }
            QTabWidget::pane {
                border: 1px solid #1e293b;
                background-color: #111827;
                border-radius: 10px;
            }
            QTabBar::tab {
                background-color: #0f172a;
                color: #94a3b8;
                padding: 10px 18px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background-color: #1d4ed8;
                color: #ffffff;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #1e293b;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #38bdf8;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #f8fafc;
                border: 2px solid #0284c7;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QProgressBar {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 7px;
                text-align: center;
                color: #f8fafc;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #0284c7;
                border-radius: 6px;
            }
            """
        )

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        # ---------------------------------------------------------------------
        # HEADER
        # ---------------------------------------------------------------------
        header = QHBoxLayout()
        title_lbl = QLabel("🎙️ VOICESTUDIO AI PRO — XỬ LÝ & ĐỔI GIỌNG LỒNG TIẾNG VIDEO")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: 800; color: #38bdf8;")
        subtitle_lbl = QLabel("Khử ồn AI • Giữ Formant Giọng Thật • Tách Nhạc Nền • Cắt Khoảng Lặng • Ghép Video Trực Tiếp")
        subtitle_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        header.addWidget(title_lbl)
        header.addStretch()
        header.addWidget(subtitle_lbl)
        main_layout.addLayout(header)

        # ---------------------------------------------------------------------
        # KHỐI NGUỒN ÂM THANH & SÓNG ÂM (TOP SECTION)
        # ---------------------------------------------------------------------
        top_box = QGroupBox("1. THU ÂM TRỰC TIẾP HOẶC TẢI FILE AUDIO / VIDEO")
        top_layout = QVBoxLayout(top_box)

        rec_row = QHBoxLayout()
        rec_row.addWidget(QLabel("Microphone:"))
        self.cmb_mic = QComboBox()
        self.cmb_mic.setMinimumWidth(260)
        rec_row.addWidget(self.cmb_mic)

        self.btn_record = QPushButton("🔴 Bắt đầu Thu Âm")
        self.btn_record.setStyleSheet("background-color: #be123c; border-color: #f43f5e; padding: 8px 16px;")
        self.btn_record.clicked.connect(self._toggle_recording)
        rec_row.addWidget(self.btn_record)

        self.lbl_rec_time = QLabel("00:00.0")
        self.lbl_rec_time.setStyleSheet("font-size: 15px; font-weight: bold; color: #f43f5e; min-width: 65px;")
        rec_row.addWidget(self.lbl_rec_time)

        self.bar_mic_level = QProgressBar()
        self.bar_mic_level.setRange(0, 100)
        self.bar_mic_level.setValue(0)
        self.bar_mic_level.setTextVisible(False)
        self.bar_mic_level.setFixedWidth(130)
        self.bar_mic_level.setFixedHeight(14)
        rec_row.addWidget(self.bar_mic_level)

        rec_row.addSpacing(12)
        self.btn_open_file = QPushButton("📂 Mở File Audio / Video (.mp4, .wav, .mp3)")
        self.btn_open_file.setStyleSheet("background-color: #0369a1; border-color: #38bdf8;")
        self.btn_open_file.clicked.connect(self._open_media_file)
        rec_row.addWidget(self.btn_open_file)

        self.btn_goto_tts = QPushButton("📝 Chuyển Văn Bản Thành Giọng Nói (TTS)")
        self.btn_goto_tts.setStyleSheet("background-color: #0d9488; border-color: #2dd4bf; padding: 8px 14px;")
        self.btn_goto_tts.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        rec_row.addWidget(self.btn_goto_tts)

        self.btn_demo_voice = QPushButton("✨ Tạo Giọng Mẫu")
        self.btn_demo_voice.clicked.connect(self._generate_demo_sample)
        rec_row.addWidget(self.btn_demo_voice)

        rec_row.addStretch()
        top_layout.addLayout(rec_row)

        # Hai biểu đồ sóng âm song song (Gốc vs Đã xử lý)
        wave_layout = QGridLayout()
        self.wave_orig = WaveformWidget("BẢN THU GỐC (ORIGINAL)", accent_color="#64748b")
        self.wave_proc = WaveformWidget("BẢN ĐÃ CẢI THIỆN & ĐỔI GIỌNG (STUDIO PROCESSED)", accent_color="#22c55e")
        wave_layout.addWidget(self.wave_orig, 0, 0)
        wave_layout.addWidget(self.wave_proc, 0, 1)

        play_orig_row = QHBoxLayout()
        self.btn_play_orig = QPushButton("▶️ Nghe Giọng Gốc")
        self.btn_play_orig.setToolTip("Nhấn để Phát / Tạm dừng (Pause) / Tiếp tục nghe bản gốc")
        self.btn_play_orig.clicked.connect(lambda: self._play_target(self.orig_audio, track_id="orig"))
        self.btn_stop_audio = QPushButton("⏹️ Dừng Phát")
        self.btn_stop_audio.setToolTip("Dừng hẳn và đưa về đầu bài (00:00)")
        self.btn_stop_audio.clicked.connect(self._stop_all_playback)
        self.lbl_file_info = QLabel("Nguồn hiện tại: Chưa nạp file")
        self.lbl_file_info.setStyleSheet("color: #94a3b8;")
        play_orig_row.addWidget(self.btn_play_orig)
        play_orig_row.addWidget(self.btn_stop_audio)
        play_orig_row.addWidget(self.lbl_file_info)
        play_orig_row.addStretch()
        wave_layout.addLayout(play_orig_row, 1, 0)

        play_proc_row = QHBoxLayout()
        self.btn_play_proc = QPushButton("🔊 Nghe Bản Đã Xử Lý")
        self.btn_play_proc.setToolTip("Nhấn để Phát / Tạm dừng (Pause) / Tiếp tục nghe tiếp từ vị trí đang dừng")
        self.btn_play_proc.setStyleSheet("background-color: #15803d; border-color: #22c55e;")
        self.btn_play_proc.clicked.connect(lambda: self._play_target(self.proc_audio, track_id="proc"))
        self.btn_stop_proc = QPushButton("⏹️")
        self.btn_stop_proc.setToolTip("Dừng phát và đưa về đầu bài (00:00)")
        self.btn_stop_proc.setStyleSheet("padding: 6px 10px;")
        self.btn_stop_proc.clicked.connect(self._stop_all_playback)
        self.btn_use_proc_as_orig = QPushButton("🔄 Đặt Làm Bản Gốc")
        self.btn_use_proc_as_orig.clicked.connect(self._promote_processed_to_orig)
        play_proc_row.addWidget(self.btn_play_proc)
        play_proc_row.addWidget(self.btn_stop_proc)
        play_proc_row.addWidget(self.btn_use_proc_as_orig)

        play_proc_row.addSpacing(8)
        lbl_qvol_title = QLabel("🔊 Âm lượng:")
        lbl_qvol_title.setStyleSheet("font-weight: bold; color: #22c55e;")
        play_proc_row.addWidget(lbl_qvol_title)

        self.sld_quick_vol = QSlider(Qt.Orientation.Horizontal)
        self.sld_quick_vol.setRange(-200, 200)  # -20.0 dB -> +20.0 dB
        self.sld_quick_vol.setValue(0)
        self.sld_quick_vol.setMinimumWidth(120)
        self.sld_quick_vol.valueChanged.connect(self._on_volume_slider_changed)
        play_proc_row.addWidget(self.sld_quick_vol, stretch=1)

        self.lbl_quick_vol = QLabel("+0.0 dB (100%)")
        self.lbl_quick_vol.setStyleSheet("color: #22c55e; font-weight: bold; min-width: 105px;")
        play_proc_row.addWidget(self.lbl_quick_vol)

        self.btn_reset_vol = QPushButton("100%")
        self.btn_reset_vol.setToolTip("Đặt lại âm lượng mặc định (0 dB / 100%)")
        self.btn_reset_vol.setStyleSheet("padding: 4px 8px;")
        self.btn_reset_vol.clicked.connect(lambda: self.sld_quick_vol.setValue(0))
        play_proc_row.addWidget(self.btn_reset_vol)

        wave_layout.addLayout(play_proc_row, 1, 1)

        top_layout.addLayout(wave_layout)
        main_layout.addWidget(top_box)

        # ---------------------------------------------------------------------
        # KHỐI ĐIỀU KHIỂN CHÍNH (TABS)
        # ---------------------------------------------------------------------
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_voice_changer(), "🎭 1. Đổi Giọng Chân Thật & Studio EQ (25 Giọng)")
        self.tabs.addTab(self._build_tab_ai_tts(), "📝 2. Chuyển Văn Bản Thành Giọng Nói (AI Text-to-Speech)")
        self.tabs.addTab(self._build_tab_noise_clean(), "🧹 3. Khử Tạp Âm, Tách Nhạc & Cắt Khoảng Lặng")
        self.tabs.addTab(self._build_tab_elevenlabs_sts(), "☁️ 4. Đổi Giọng Thu Âm AI Cloud (ElevenLabs STS)")
        main_layout.addWidget(self.tabs, stretch=1)

        # ---------------------------------------------------------------------
        # THANH TÁC VỤ & XUẤT FILE / GHÉP VIDEO (BOTTOM BAR)
        # ---------------------------------------------------------------------
        bottom_frame = QFrame()
        bottom_frame.setStyleSheet("background-color: #111827; border: 1px solid #1e293b; border-radius: 10px; padding: 6px;")
        bottom_layout = QHBoxLayout(bottom_frame)

        self.btn_process = QPushButton("⚡ ÁP DỤNG LỌC TẠP ÂM & ĐỔI GIỌNG NGAY")
        self.btn_process.setStyleSheet(
            "background-color: #2563eb; border-color: #60a5fa; font-size: 14px; font-weight: bold; padding: 11px 22px;"
        )
        self.btn_process.clicked.connect(self._run_processing)
        bottom_layout.addWidget(self.btn_process)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumWidth(260)
        bottom_layout.addWidget(self.progress_bar, stretch=1)

        self.btn_export_audio = QPushButton("💾 Xuất File Audio (.WAV / .MP3)")
        self.btn_export_audio.setStyleSheet("background-color: #0f766e; border-color: #14b8a6; padding: 10px 16px;")
        self.btn_export_audio.clicked.connect(self._export_audio)
        bottom_layout.addWidget(self.btn_export_audio)

        self.btn_mux_video = QPushButton("🎬 Ghép Giọng Đã Xử Lý Vào Video (.MP4)")
        self.btn_mux_video.setStyleSheet("background-color: #6d28d9; border-color: #a855f7; padding: 10px 16px;")
        self.btn_mux_video.clicked.connect(self._export_muxed_video)
        bottom_layout.addWidget(self.btn_mux_video)

        main_layout.addWidget(bottom_frame)

    # =========================================================================
    # XÂY DỰNG CÁC TAB CHI TIẾT
    # =========================================================================
    def _build_tab_voice_changer(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)

        # Cột trái: Chọn Nhóm Giọng, Preset Giọng TD-PSOLA & Đổi Giọng AI Speech-to-Speech
        left_box = QGroupBox("CHỌN LOẠI GIỌNG (25 PRESET TD-PSOLA) & ĐỔI GIỌNG AI")
        left_layout = QVBoxLayout(left_box)
        left_layout.setSpacing(5)

        cat_row = QHBoxLayout()
        cat_row.addWidget(QLabel("📂 Lọc theo nhóm giọng:"))
        self.cmb_category = QComboBox()
        for c_id, c_name in VOICE_CATEGORIES:
            self.cmb_category.addItem(c_name, userData=c_id)
        self.cmb_category.currentIndexChanged.connect(self._on_category_changed)
        cat_row.addWidget(self.cmb_category, stretch=1)
        left_layout.addLayout(cat_row)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("🎭 Chọn kiểu giọng mẫu:"))
        self.cmb_preset = QComboBox()
        for key, info in VOICE_PRESETS.items():
            self.cmb_preset.addItem(info["name"], userData=key)
        self.cmb_preset.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.cmb_preset, stretch=1)
        left_layout.addLayout(preset_row)

        self.lbl_preset_desc = QLabel(VOICE_PRESETS["studio_natural"]["desc"])
        self.lbl_preset_desc.setWordWrap(True)
        self.lbl_preset_desc.setStyleSheet("color: #38bdf8; background-color: #0f172a; padding: 6px; border-radius: 6px;")
        left_layout.addWidget(self.lbl_preset_desc)

        self.sld_pitch, self.lbl_pitch = self._create_slider_row(
            left_layout,
            "Cao độ dây thanh (Pitch Shift TD-PSOLA - Không vang):",
            -120, 120, 0,
            fmt_fn=lambda v: f"{v / 10.0:+.1f} bán âm",
        )
        self.sld_formant, self.lbl_formant = self._create_slider_row(
            left_layout,
            "Kích thước Thanh quản (Formant - Đổi cấu trúc Nam/Nữ):",
            -60, 60, 0,
            fmt_fn=lambda v: f"{v / 10.0:+.1f} ({'Trầm/Nam' if v < -3 else ('Thanh/Nữ' if v > 3 else 'Tự nhiên')})",
        )
        self.sld_pitch_range, self.lbl_pitch_range = self._create_slider_row(
            left_layout,
            "Độ biểu cảm / Biến thiên ngữ điệu (Intonation Range):",
            15, 150, 100,
            fmt_fn=lambda v: f"{v}% ({'Trầm tĩnh/Đều' if v < 85 else ('Sôi nổi/Biểu cảm' if v > 112 else 'Tự nhiên')})",
        )
        self.sld_rate, self.lbl_rate = self._create_slider_row(
            left_layout,
            "Tốc độ nói (Speaking Rate):",
            75, 135, 100,
            fmt_fn=lambda v: f"{v / 100.0:.2f}x",
        )

        # Khối Đổi Giọng AI Speech-to-Speech Miễn Phí (Khử vang 100% & Đổi giọng khác biệt hoàn toàn)
        ai_sts_frame = QFrame()
        ai_sts_frame.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #0284c7; border-radius: 8px; padding: 6px;"
        )
        ai_sts_layout = QVBoxLayout(ai_sts_frame)
        ai_sts_layout.setContentsMargins(6, 6, 6, 6)
        ai_sts_layout.setSpacing(4)

        lbl_ai_sts = QLabel(
            "🪄 <b>ĐỔI BẢN THU SANG GIỌNG AI ĐỌC LẠI (KHỬ VANG 100% & RÕ TỪNG CHỮ):</b>"
        )
        lbl_ai_sts.setStyleSheet("color: #fbbf24; border: none;")
        ai_sts_layout.addWidget(lbl_ai_sts)

        sts_row = QHBoxLayout()
        self.cmb_quick_ai_voice = QComboBox()
        for vid, vname in EDGE_TTS_VOICES:
            self.cmb_quick_ai_voice.addItem(vname, userData=vid)
        sts_row.addWidget(self.cmb_quick_ai_voice, stretch=1)

        self.btn_quick_ai_sts = QPushButton("🪄 Đổi Bản Thu Sang Giọng AI Này")
        self.btn_quick_ai_sts.setStyleSheet(
            "background-color: #0d9488; border-color: #2dd4bf; font-weight: bold; padding: 6px 12px;"
        )
        self.btn_quick_ai_sts.clicked.connect(self._run_ai_neural_sts)
        sts_row.addWidget(self.btn_quick_ai_sts)
        ai_sts_layout.addLayout(sts_row)
        left_layout.addWidget(ai_sts_frame)

        left_layout.addStretch()
        layout.addWidget(left_box, stretch=1)

        # Cột phải: Khử Vang Phòng, Độ Rõ Lời & Bàn trộn Spotify Pedalboard
        right_box = QGroupBox("KHỬ VANG PHÒNG (DE-REVERB), LÀM RÕ LỜI & BÀN TRỘN STUDIO")
        right_layout = QVBoxLayout(right_box)
        right_layout.setSpacing(5)

        self.sld_dereverb, _ = self._create_slider_row(
            right_layout,
            "🏠 Khử tiếng vang phòng & cắt âm ồm hộp (De-Reverb):",
            0, 100, 50,
            lambda v: f"{v}% (Khô & Sát Mic)",
        )
        self.sld_clarity, _ = self._create_slider_row(
            right_layout,
            "🎯 Độ rõ lời & sắc nét phụ âm (Articulation Clarity):",
            0, 100, 50,
            lambda v: f"{v}% (Rõ chữ)",
        )
        self.sld_bass, _ = self._create_slider_row(
            right_layout, "Độ ấm trầm gọn (Bass Warmth 135Hz):", -60, 80, 10, lambda v: f"{v / 10.0:+.1f} dB"
        )
        self.sld_mid, _ = self._create_slider_row(
            right_layout, "Độ nổi lời nói (Mid Presence 2.8kHz):", -60, 80, 25, lambda v: f"{v / 10.0:+.1f} dB"
        )
        self.sld_treble, _ = self._create_slider_row(
            right_layout, "Độ sáng / Chi tiết (Treble Air 8kHz):", -60, 80, 18, lambda v: f"{v / 10.0:+.1f} dB"
        )
        self.sld_comp, _ = self._create_slider_row(
            right_layout, "Nén đều âm lượng từng câu (Studio Compressor):", 0, 100, 60, lambda v: f"{v}%"
        )
        self.sld_deesser, _ = self._create_slider_row(
            right_layout, "Giảm chói âm xuýt (De-esser s, x, ch):", 0, 100, 30, lambda v: f"{v}%"
        )
        self.sld_reverb, _ = self._create_slider_row(
            right_layout, "Độ vang nhân tạo (Mặc định 0% = Tắt vang hoàn toàn):", 0, 40, 0, lambda v: f"{v}%"
        )

        fx_row = QHBoxLayout()
        fx_row.addWidget(QLabel("✨ Hiệu ứng đặc biệt:"))
        self.cmb_special_fx = QComboBox()
        self.cmb_special_fx.addItem("Không (Giọng khô tự nhiên chuẩn Studio)", userData="none")
        self.cmb_special_fx.addItem("📻 Loa Phát Thanh Cổ Điển / Bộ Đàm (Radio)", userData="radio")
        self.cmb_special_fx.addItem("📢 Loa Phóng Thanh (Megaphone)", userData="megaphone")
        self.cmb_special_fx.addItem("🤖 Robot / Trí Tuệ Nhân Tạo (Sci-Fi Vocoder)", userData="robot")
        self.cmb_special_fx.addItem("🏛️ Không gian Đại Sảnh Điện Ảnh (Dark Hall)", userData="hall")
        fx_row.addWidget(self.cmb_special_fx, stretch=1)
        right_layout.addLayout(fx_row)

        self.chk_normalize = QCheckBox("Chuẩn hóa âm lượng chuẩn YouTube / TikTok / CapCut (-14 LUFS / chống vỡ tiếng)")
        self.chk_normalize.setChecked(True)
        right_layout.addWidget(self.chk_normalize)

        # Thanh trượt Tăng / Giảm Âm Lượng Tổng (Ẩn hoặc đặt gọn để đồng bộ)
        self.sld_master_vol, self.lbl_master_vol = self._create_slider_row(
            right_layout,
            "🔊 Tăng / Giảm Âm Lượng Đầu Ra (Master Volume):",
            -200, 200, 0,
            fmt_fn=lambda v: f"{v / 10.0:+.1f} dB ({int(round((10.0 ** (v / 200.0)) * 100))}%)",
        )
        self.sld_master_vol.valueChanged.connect(self._on_master_vol_tab_changed)

        right_layout.addStretch()
        layout.addWidget(right_box, stretch=1)
        return tab

    def _build_tab_noise_clean(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        # Khối 1: Lọc tạp âm & Tách nhạc nền
        box_nr = QGroupBox("KHỬ TẠP ÂM AI & TÁCH LỜI KHỎI NHẠC NỀN")
        nr_layout = QVBoxLayout(box_nr)

        self.chk_nr = QCheckBox("Bật khử tạp âm nền thông minh (Spectral Gating AI)")
        self.chk_nr.setChecked(True)
        nr_layout.addWidget(self.chk_nr)

        self.sld_nr_strength, _ = self._create_slider_row(
            nr_layout, "Cường độ lọc tạp âm (Noise Suppression):", 0, 100, 90, lambda v: f"{v}%"
        )

        self.chk_non_stationary = QCheckBox("Chế độ lọc tạp âm biến thiên mạnh (Chỉ bật khi thu ngoài đường nhiều tiếng xe cộ)")
        self.chk_non_stationary.setChecked(False)
        nr_layout.addWidget(self.chk_non_stationary)

        self.chk_hum = QCheckBox("Tự dò & cắt tiếng ù điện 50/60 Hz và các hài (Hum Removal)")
        self.chk_hum.setChecked(True)
        nr_layout.addWidget(self.chk_hum)

        self.sld_gap_db, _ = self._create_slider_row(
            nr_layout, "Hạ nền ồn trong khoảng nghỉ giữa câu (Gap Attenuation):", 0, 24, 18, lambda v: f"-{v} dB"
        )

        self.chk_ai_denoise = QCheckBox("Khử nhiễu bằng AI RNNoise (thử nghiệm — hợp tiếng gõ phím/người nói xa, có thể ăn lời nếu phòng vang)")
        self.chk_ai_denoise.setChecked(False)
        nr_layout.addWidget(self.chk_ai_denoise)

        self.sld_highpass, _ = self._create_slider_row(
            nr_layout, "Cắt tiếng ù trầm & rung micro (Low-Cut / Highpass):", 20, 180, 80, lambda v: f"{v} Hz"
        )

        nr_layout.addSpacing(10)
        self.chk_vocal_iso = QCheckBox("🎵 Tách lời nói khỏi nhạc nền (Vocal Isolation — Dùng khi lấy tiếng từ Video có sẵn nhạc)")
        self.chk_vocal_iso.setStyleSheet("font-weight: bold; color: #fbbf24;")
        nr_layout.addWidget(self.chk_vocal_iso)

        nr_layout.addStretch()
        layout.addWidget(box_nr, stretch=1)

        # Khối 2: Tự động cắt bỏ khoảng lặng / ngập ngừng
        box_trim = QGroupBox("TỰ ĐỘNG CẮT KHOẢNG LẶNG & NGẬP NGỪNG KHI THU ÂM")
        trim_layout = QVBoxLayout(box_trim)

        self.chk_trim_silence = QCheckBox("✂️ Tự động cắt bỏ các đoạn im lặng / nghỉ quá lâu giữa các câu")
        self.chk_trim_silence.setStyleSheet("font-weight: bold; color: #38bdf8;")
        trim_layout.addWidget(self.chk_trim_silence)

        self.sld_silence_db, _ = self._create_slider_row(
            trim_layout, "Ngưỡng nhận diện im lặng (Silence Threshold):", 20, 60, 38, lambda v: f"-{v} dB"
        )
        self.sld_silence_pad, _ = self._create_slider_row(
            trim_layout, "Khoảng nghỉ tự nhiên giữ lại giữa các câu:", 80, 800, 250, lambda v: f"{v} ms"
        )

        hint = QLabel(
            "💡 Mẹo làm Video nhanh:\n"
            "• Khi thu âm đọc kịch bản, nếu bạn lỡ đọc vấp hoặc dừng lại suy nghĩ vài giây, chỉ cần bật tính năng Cắt khoảng lặng — ứng dụng sẽ tự động nối mượt các câu nói lại với nhau.\n"
            "• Nếu file Video gốc bị dính nhạc nền nhỏ, hãy bật 'Tách lời nói khỏi nhạc nền'."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #94a3b8; background-color: #0f172a; padding: 10px; border-radius: 8px;")
        trim_layout.addWidget(hint)
        trim_layout.addStretch()
        layout.addWidget(box_trim, stretch=1)
        return tab

    def _build_tab_ai_tts(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(16)

        # Cột trái: Soạn thảo kịch bản & Tiện ích văn bản
        script_box = QGroupBox("SOẠN THẢO KỊCH BẢN LỒNG TIẾNG VIDEO (CHUYỂN VĂN BẢN THÀNH GIỌNG NÓI)")
        script_layout = QVBoxLayout(script_box)

        tool_row = QHBoxLayout()
        btn_load_txt = QPushButton("📄 Mở File .TXT / Phụ đề .SRT")
        btn_load_txt.setStyleSheet("background-color: #0369a1; border-color: #38bdf8; padding: 6px 12px;")
        btn_load_txt.clicked.connect(self._load_tts_text_file)
        tool_row.addWidget(btn_load_txt)

        self.cmb_tts_sample = QComboBox()
        self.cmb_tts_sample.addItem("💡 Chèn kịch bản mẫu...", userData="")
        self.cmb_tts_sample.addItem("🎬 Mẫu 1: Review Phim / Tóm Tắt Phim", userData="movie")
        self.cmb_tts_sample.addItem("🛒 Mẫu 2: Quảng Cáo / Giới Thiệu Sản Phẩm TikTok", userData="ad")
        self.cmb_tts_sample.addItem("📺 Mẫu 3: Bản Tin Thời Sự / Công Nghệ", userData="news")
        self.cmb_tts_sample.addItem("🌙 Mẫu 4: Podcast Tâm Sự / Chữa Lành Đêm Khuya", userData="podcast")
        self.cmb_tts_sample.currentIndexChanged.connect(self._insert_tts_template)
        tool_row.addWidget(self.cmb_tts_sample, stretch=1)

        btn_clear_txt = QPushButton("🧹 Xóa trắng")
        btn_clear_txt.setStyleSheet("padding: 6px 10px;")
        btn_clear_txt.clicked.connect(lambda: self.txt_tts.clear())
        tool_row.addWidget(btn_clear_txt)
        script_layout.addLayout(tool_row)

        self.txt_tts = QTextEdit()
        self.txt_tts.setPlaceholderText(
            "Nhập hoặc dán nội dung kịch bản video của bạn vào đây để chuyển thành giọng nói...\n"
            "Mẹo: Dùng dấu phẩy (,) để giọng đọc nghỉ nhịp ngắn, dấu chấm (.) hoặc xuống dòng để nghỉ nhịp dài tự nhiên."
        )
        self.txt_tts.setPlainText(
            "Xin chào các bạn! Chào mừng đã quay trở lại với kênh của mình. "
            "Đây là tính năng chuyển văn bản thành giọng nói trí tuệ nhân tạo tích hợp trong VoiceStudio AI Pro. "
            "Bạn chỉ cần nhập kịch bản, chọn giọng đọc yêu thích và ghép trực tiếp vào video chỉ trong vài giây!"
        )
        self.txt_tts.textChanged.connect(self._update_tts_stats)
        script_layout.addWidget(self.txt_tts, stretch=1)

        self.lbl_tts_stats = QLabel("Số từ: 48 từ   |   Số ký tự: 245   |   Thời lượng đọc ước tính: ~16.0 giây")
        self.lbl_tts_stats.setStyleSheet("color: #38bdf8; font-weight: 600;")
        script_layout.addWidget(self.lbl_tts_stats)

        layout.addWidget(script_box, stretch=3)

        # Cột phải: Chọn giọng đọc, Tốc độ, Cao độ & Hậu kỳ Studio
        cfg_box = QGroupBox("TÙY CHỈNH GIỌNG ĐỌC AI & HẬU KỲ PHÒNG THU")
        cfg_layout = QVBoxLayout(cfg_box)

        cfg_layout.addWidget(QLabel("🗣️ Chọn giọng đọc & phong cách AI Neural (27 Giọng):"))
        self.cmb_tts_voice = QComboBox()
        for vid, vname in EDGE_TTS_VOICES:
            self.cmb_tts_voice.addItem(vname, userData=vid)
        self.cmb_tts_voice.currentIndexChanged.connect(self._on_tts_voice_changed)
        cfg_layout.addWidget(self.cmb_tts_voice)

        self.sld_tts_rate, _ = self._create_slider_row(
            cfg_layout, "⏩ Tốc độ đọc (Speaking Rate):", -40, 40, 0, lambda v: f"{v:+d}%"
        )
        self.sld_tts_pitch, _ = self._create_slider_row(
            cfg_layout, "🎵 Độ trầm / bổng giọng AI (Pitch):", -25, 25, 0, lambda v: f"{v:+d} Hz"
        )

        cfg_layout.addSpacing(6)
        self.chk_tts_auto_studio = QCheckBox(
            "✨ Tự động làm ấm & qua Bộ lọc Studio EQ (Tab 1) sau khi tạo giọng"
        )
        self.chk_tts_auto_studio.setChecked(False)
        self.chk_tts_auto_studio.setStyleSheet("color: #fbbf24; font-weight: bold;")
        cfg_layout.addWidget(self.chk_tts_auto_studio)

        self.chk_tts_autoplay = QCheckBox("🔊 Tự động phát nghe thử ngay sau khi chuyển văn bản xong")
        self.chk_tts_autoplay.setChecked(True)
        cfg_layout.addWidget(self.chk_tts_autoplay)

        cfg_layout.addSpacing(8)
        self.btn_gen_tts = QPushButton("🎙️ CHUYỂN VĂN BẢN THÀNH GIỌNG NÓI NGAY")
        self.btn_gen_tts.setStyleSheet(
            "background-color: #0284c7; border-color: #38bdf8; font-size: 14px; font-weight: bold; padding: 12px;"
        )
        self.btn_gen_tts.clicked.connect(lambda: self._run_edge_tts(save_after=False))
        cfg_layout.addWidget(self.btn_gen_tts)

        self.btn_tts_export_direct = QPushButton("💾 Đọc Văn Bản & Lưu Luôn Ra File (.MP3 / .WAV)")
        self.btn_tts_export_direct.setStyleSheet(
            "background-color: #0f766e; border-color: #14b8a6; padding: 10px;"
        )
        self.btn_tts_export_direct.clicked.connect(lambda: self._run_edge_tts(save_after=True))
        cfg_layout.addWidget(self.btn_tts_export_direct)

        cfg_layout.addStretch()
        layout.addWidget(cfg_box, stretch=2)
        return tab

    def _build_tab_elevenlabs_sts(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        sts_box = QGroupBox("CHUYỂN ĐỔI GIỌNG THU ÂM SIÊU THỰC (ELEVENLABS SPEECH-TO-SPEECH API)")
        sts_layout = QVBoxLayout(sts_box)

        sts_desc = QLabel(
            "Tính năng nâng cao dành cho người dùng có tài khoản ElevenLabs:\n"
            "• Biến đổi trực tiếp bản thu âm giọng thật của bạn sang bất kỳ giọng AI nào trên ElevenLabs.\n"
            "• Giữ nguyên 100% cảm xúc, cách nhấn nhá, lên xuống giọng và nhịp nghỉ của bạn."
        )
        sts_desc.setWordWrap(True)
        sts_desc.setStyleSheet("color: #94a3b8; background-color: #0f172a; padding: 10px; border-radius: 8px;")
        sts_layout.addWidget(sts_desc)

        sts_layout.addWidget(QLabel("ElevenLabs API Key:"))
        self.inp_xi_key = QLineEdit()
        self.inp_xi_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.inp_xi_key.setPlaceholderText("Nhập xi-api-key của bạn (nếu dùng ElevenLabs Cloud)...")
        sts_layout.addWidget(self.inp_xi_key)

        sts_layout.addWidget(QLabel("Chọn nhanh 16 Giọng AI ElevenLabs nổi tiếng nhất:"))
        self.cmb_xi_preset = QComboBox()
        for xid, xname in ELEVENLABS_VOICES:
            self.cmb_xi_preset.addItem(xname, userData=xid)
        self.cmb_xi_preset.currentIndexChanged.connect(
            lambda: self.inp_xi_voice.setText(self.cmb_xi_preset.currentData() or "")
        )
        sts_layout.addWidget(self.cmb_xi_preset)

        sts_layout.addWidget(QLabel("Hoặc tự nhập Voice ID mục tiêu:"))
        self.inp_xi_voice = QLineEdit("JBFqnCBsd6RMkjVDRZzb")
        sts_layout.addWidget(self.inp_xi_voice)

        self.btn_run_sts = QPushButton("✨ Chuyển Đổi Giọng Thu Âm Bằng ElevenLabs STS")
        self.btn_run_sts.setStyleSheet("background-color: #7c3aed; border-color: #a855f7; padding: 11px;")
        self.btn_run_sts.clicked.connect(self._run_elevenlabs_sts)
        sts_layout.addWidget(self.btn_run_sts)

        sts_layout.addStretch()
        layout.addWidget(sts_box)
        return tab

    def _create_slider_row(self, parent_layout: QVBoxLayout, label_text: str, min_v: int, max_v: int, init_v: int, fmt_fn):
        row_top = QHBoxLayout()
        lbl_title = QLabel(label_text)
        lbl_val = QLabel(fmt_fn(init_v))
        lbl_val.setStyleSheet("color: #38bdf8; font-weight: bold;")
        row_top.addWidget(lbl_title)
        row_top.addStretch()
        row_top.addWidget(lbl_val)
        parent_layout.addLayout(row_top)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(init_v)
        slider.valueChanged.connect(lambda v: lbl_val.setText(fmt_fn(v)))
        parent_layout.addWidget(slider)
        return slider, lbl_val

    # =========================================================================
    # SỰ KIỆN & LOGIC ĐIỀU KHIỂN
    # =========================================================================
    def _populate_devices(self):
        self.cmb_mic.clear()
        devs = self.engine.get_input_devices()
        if not devs:
            self.cmb_mic.addItem("Mặc định hệ thống (Default Microphone)", userData=None)
        else:
            for idx, name in devs:
                self.cmb_mic.addItem(name, userData=idx)

    def _on_category_changed(self):
        cat_id = self.cmb_category.currentData()
        self.cmb_preset.blockSignals(True)
        self.cmb_preset.clear()
        for key, info in VOICE_PRESETS.items():
            if cat_id == "all" or info.get("category") == cat_id:
                self.cmb_preset.addItem(info["name"], userData=key)
        self.cmb_preset.blockSignals(False)
        if self.cmb_preset.count() > 0:
            self._on_preset_changed()

    def _on_preset_changed(self):
        preset_key = self.cmb_preset.currentData()
        info = VOICE_PRESETS.get(preset_key)
        if not info:
            return
        self.lbl_preset_desc.setText(info["desc"])
        self.sld_pitch.setValue(int(info["pitch_semitones"] * 10))
        self.sld_formant.setValue(int(info["formant_shift"] * 10))
        self.sld_pitch_range.setValue(int(info.get("pitch_range_factor", 1.0) * 100))
        self.sld_rate.setValue(int(info.get("speaking_rate", 1.0) * 100))
        self.sld_bass.setValue(int(info["bass_gain_db"] * 10))
        self.sld_mid.setValue(int(info["mid_gain_db"] * 10))
        self.sld_treble.setValue(int(info["treble_gain_db"] * 10))
        self.sld_comp.setValue(int(info["compressor_amount"] * 100))
        self.sld_deesser.setValue(int(info["deesser_amount"] * 100))
        self.sld_reverb.setValue(int(info["reverb_amount"] * 100))

        fx = info.get("special_fx", "none")
        idx = self.cmb_special_fx.findData(fx)
        if idx >= 0:
            self.cmb_special_fx.setCurrentIndex(idx)

    def _on_tts_voice_changed(self):
        raw = self.cmb_tts_voice.currentData() or ""
        parts = raw.split("|")
        if len(parts) == 3:
            try:
                self.sld_tts_rate.setValue(int(parts[1]))
                self.sld_tts_pitch.setValue(int(parts[2]))
            except Exception:
                pass

    def _collect_settings(self) -> VoiceSettings:
        return VoiceSettings(
            noise_reduction_enabled=self.chk_nr.isChecked(),
            noise_reduction_strength=self.sld_nr_strength.value() / 100.0,
            stationary_noise=not self.chk_non_stationary.isChecked(),
            ai_denoise=self.chk_ai_denoise.isChecked(),
            hum_removal=self.chk_hum.isChecked(),
            gap_attenuation_db=float(self.sld_gap_db.value()),
            dereverb_amount=self.sld_dereverb.value() / 100.0,
            remove_Background_music=self.chk_vocal_iso.isChecked(),
            auto_trim_silence=self.chk_trim_silence.isChecked(),
            silence_threshold_db=-float(self.sld_silence_db.value()),
            silence_min_pause_ms=self.sld_silence_pad.value(),
            highpass_hz=float(self.sld_highpass.value()),
            deesser_amount=self.sld_deesser.value() / 100.0,
            clarity_boost=self.sld_clarity.value() / 100.0,
            bass_gain_db=self.sld_bass.value() / 10.0,
            mid_gain_db=self.sld_mid.value() / 10.0,
            treble_gain_db=self.sld_treble.value() / 10.0,
            compressor_amount=self.sld_comp.value() / 100.0,
            reverb_amount=self.sld_reverb.value() / 100.0,
            normalize_lufs=self.chk_normalize.isChecked(),
            voice_preset=self.cmb_preset.currentData() or "studio_natural",
            pitch_semitones=self.sld_pitch.value() / 10.0,
            formant_shift=self.sld_formant.value() / 10.0,
            pitch_range_factor=self.sld_pitch_range.value() / 100.0,
            speaking_rate=self.sld_rate.value() / 100.0,
            special_fx=self.cmb_special_fx.currentData() or "none",
        )

    def _toggle_recording(self):
        if not self.engine.is_recording:
            try:
                dev_idx = self.cmb_mic.currentData()
                self.engine.start_recording(
                    device_index=dev_idx,
                    level_callback=lambda lvl: self.sig_mic_level.emit(lvl),
                )
                self.record_start_time = time.time()
                self.rec_timer.start(100)
                self.btn_record.setText("⏹️ Dừng Thu Âm")
                self.btn_record.setStyleSheet("background-color: #eab308; color: #000; font-weight: bold; padding: 8px 16px;")
                self.progress_bar.setFormat("Đang thu âm trực tiếp từ Microphone...")
            except Exception as e:
                QMessageBox.warning(self, "Lỗi Microphone", f"Không thể mở Microphone:\n{e}")
        else:
            self.rec_timer.stop()
            self.bar_mic_level.setValue(0)
            audio, sr = self.engine.stop_recording()
            self.btn_record.setText("🔴 Bắt đầu Thu Âm")
            self.btn_record.setStyleSheet("background-color: #be123c; border-color: #f43f5e; padding: 8px 16px;")
            self._set_original_audio(audio, sr, "Bản thu âm Microphone trực tiếp", video_path=None)
            # Tự động xử lý ngay sau khi thu âm xong
            self._run_processing()

    def _update_record_timer(self):
        elapsed = time.time() - self.record_start_time
        mins = int(elapsed // 60)
        secs = elapsed % 60
        self.lbl_rec_time.setText(f"{mins:02d}:{secs:04.1f}")

    def _on_mic_level(self, level: float):
        self.bar_mic_level.setValue(int(level * 100))

    def _open_media_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn File Âm Thanh hoặc Video",
            "",
            "Media Files (*.wav *.mp3 *.m4a *.flac *.ogg *.mp4 *.mov *.mkv *.avi *.webm);;All Files (*.*)",
        )
        if not path:
            return
        try:
            self.progress_bar.setValue(20)
            self.progress_bar.setFormat("Đang đọc dữ liệu âm thanh từ file...")
            QApplication.processEvents()
            audio, sr, is_video = self.engine.load_media_file(path)
            vpath = path if is_video else None
            label = f"{Path(path).name} {'(🎬 File Video)' if is_video else '(🎵 File Audio)'}"
            self._set_original_audio(audio, sr, label, video_path=vpath)
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Đã tải file thành công! Nhấn 'Áp dụng Lọc tạp âm & Đổi giọng' để xử lý.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi đọc file", f"Không thể đọc file media:\n{e}")

    def _generate_demo_sample(self):
        """
        Tạo giọng nói tiếng Việt mẫu (bằng Edge-TTS kèm giả lập tiếng ồn nền nhẹ)
        để người dùng có thể thử nghiệm ngay lập tức bộ lọc tạp âm và đổi giọng!
        """
        self.progress_bar.setValue(25)
        self.progress_bar.setFormat("Đang tạo giọng mẫu thử nghiệm (kèm tạp âm giả lập)...")
        self.btn_demo_voice.setEnabled(False)

        def _on_done(audio: np.ndarray, sr: int, _msg: str):
            self.btn_demo_voice.setEnabled(True)
            # Pha thêm chút tiếng ồn nền giả lập + khoảng lặng để người dùng thấy rõ hiệu quả lọc
            rng = np.random.default_rng(123)
            noise = rng.normal(0, 0.012, size=len(audio)).astype(np.float32)
            hum = (0.015 * np.sin(2 * np.pi * 60.0 * np.arange(len(audio)) / sr)).astype(np.float32)
            noisy_demo = np.clip(audio + noise + hum, -0.98, 0.98)
            self._set_original_audio(noisy_demo, sr, "Giọng mẫu thử nghiệm (Có tiếng ồn nền & ù 60Hz)", video_path=None)
            self._run_processing()

        def _on_fail(err: str):
            self.btn_demo_voice.setEnabled(True)
            # Nếu không có mạng để gọi Edge-TTS, tạo sóng tổng hợp giọng nói mô phỏng
            sr = self.engine.target_sr
            t = np.linspace(0, 4.0, int(sr * 4.0), dtype=np.float32)
            f0 = 135.0 + 15.0 * np.sin(2 * np.pi * 1.5 * t)
            phase = 2 * np.pi * np.cumsum(f0) / sr
            synth = np.zeros_like(t)
            for h in range(1, 18):
                synth += (1.0 / (h ** 1.15)) * np.sin(h * phase)
            synth *= 0.25 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.2 * t))
            noise = np.random.default_rng(7).normal(0, 0.015, size=len(t)).astype(np.float32)
            self._set_original_audio(synth + noise, sr, "Tín hiệu giọng mẫu Offline", video_path=None)
            self._run_processing()

        self._ai_worker = AIWorker(
            "edge_tts",
            self.engine,
            {
                "text": "Xin chào! Đây là bản thu âm thử nghiệm có lẫn tiếng ồn nền. Hãy nghe sự khác biệt trước và sau khi lọc tạp âm và đổi giọng.",
                "voice_id": "vi-VN-NamMinhNeural",
                "rate_pct": 0,
                "pitch_hz": 0,
            },
        )
        self._ai_worker.finished_ok.connect(_on_done)
        self._ai_worker.failed.connect(_on_fail)
        self._ai_worker.start()

    def _on_volume_slider_changed(self, val: int):
        db = val / 10.0
        pct = int(round((10.0 ** (db / 20.0)) * 100))
        self.lbl_quick_vol.setText(f"{db:+.1f} dB ({pct}%)")
        if hasattr(self, "sld_master_vol") and self.sld_master_vol.value() != val:
            self.sld_master_vol.blockSignals(True)
            self.sld_master_vol.setValue(val)
            self.lbl_master_vol.setText(f"{db:+.1f} dB ({pct}%)")
            self.sld_master_vol.blockSignals(False)
        self._apply_live_volume_gain()

    def _on_master_vol_tab_changed(self, val: int):
        if hasattr(self, "sld_quick_vol") and self.sld_quick_vol.value() != val:
            self.sld_quick_vol.setValue(val)

    def _apply_live_volume_gain(self):
        base = self.base_proc_audio if self.base_proc_audio is not None else self.orig_audio
        if base is None:
            return
        gain_db = self.sld_quick_vol.value() / 10.0
        self.proc_audio = self.engine.apply_volume_gain(base, gain_db)
        self.wave_proc.set_audio(self.proc_audio, self.current_sr)
        self.engine.update_playing_buffer(self.proc_audio, track_id="proc")

    def _set_original_audio(self, audio: np.ndarray, sr: int, source_name: str, video_path: Optional[str]):
        self._stop_all_playback()
        self.orig_audio = audio
        self.current_sr = sr
        self.loaded_video_path = video_path
        self.lbl_file_info.setText(f"Nguồn: {source_name}")
        self.wave_orig.set_audio(audio, sr)

    def _promote_processed_to_orig(self):
        if self.proc_audio is None:
            return
        self._stop_all_playback()
        self.orig_audio = self.proc_audio.copy()
        self.base_proc_audio = self.proc_audio.copy()
        self.sld_quick_vol.setValue(0)
        self.wave_orig.set_audio(self.orig_audio, self.current_sr)
        self.lbl_file_info.setText("Nguồn: Bản đã xử lý (Đặt làm gốc)")

    @staticmethod
    def _fmt_mmss(sec: float) -> str:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"

    def _play_target(self, audio: Optional[np.ndarray], track_id: str = "proc"):
        if audio is None or len(audio) == 0:
            QMessageBox.information(self, "Chưa có âm thanh", "Hãy thu âm, mở file hoặc nhấn xử lý trước khi nghe thử.")
            return

        state = self.engine.toggle_play_pause(
            audio,
            self.current_sr,
            track_id=track_id,
            on_finish=lambda: self.sig_play_finished.emit(),
        )
        if state in ("playing", "paused"):
            if not self.play_ui_timer.isActive():
                self.play_ui_timer.start(150)
        self._update_playback_buttons_ui()

    def _stop_all_playback(self):
        self.engine.stop_playback()
        self.play_ui_timer.stop()
        self._reset_play_buttons_appearance()

    def _on_playback_finished_ui(self):
        if not self.engine.is_playing:
            self.play_ui_timer.stop()
            self._reset_play_buttons_appearance()

    def _reset_play_buttons_appearance(self):
        self.btn_play_orig.setText("▶️ Nghe Giọng Gốc")
        self.btn_play_orig.setStyleSheet("")
        self.btn_play_proc.setText("🔊 Nghe Bản Đã Xử Lý")
        self.btn_play_proc.setStyleSheet("background-color: #15803d; border-color: #22c55e;")

    def _update_playback_buttons_ui(self):
        if not self.engine.is_playing:
            self.play_ui_timer.stop()
            self._reset_play_buttons_appearance()
            return

        cur_s, tot_s = self.engine.get_playback_time()
        time_str = f"{self._fmt_mmss(cur_s)} / {self._fmt_mmss(tot_s)}"
        tid = self.engine.playing_track_id
        paused = self.engine.is_paused

        if tid == "proc":
            self.btn_play_orig.setText("▶️ Nghe Giọng Gốc")
            self.btn_play_orig.setStyleSheet("")
            if paused:
                self.btn_play_proc.setText(f"▶️ Tiếp Tục Nghe ({time_str})")
                self.btn_play_proc.setStyleSheet("background-color: #0284c7; border-color: #38bdf8; font-weight: bold;")
            else:
                self.btn_play_proc.setText(f"⏸️ Tạm Dừng ({time_str})")
                self.btn_play_proc.setStyleSheet("background-color: #d97706; border-color: #fbbf24; font-weight: bold;")
        elif tid == "orig":
            self.btn_play_proc.setText("🔊 Nghe Bản Đã Xử Lý")
            self.btn_play_proc.setStyleSheet("background-color: #15803d; border-color: #22c55e;")
            if paused:
                self.btn_play_orig.setText(f"▶️ Tiếp Tục Gốc ({time_str})")
                self.btn_play_orig.setStyleSheet("background-color: #0284c7; border-color: #38bdf8; font-weight: bold;")
            else:
                self.btn_play_orig.setText(f"⏸️ Tạm Dừng Gốc ({time_str})")
                self.btn_play_orig.setStyleSheet("background-color: #d97706; border-color: #fbbf24; font-weight: bold;")

    def _run_processing(self):
        if self.orig_audio is None or len(self.orig_audio) == 0:
            QMessageBox.information(self, "Chưa có âm thanh gốc", "Vui lòng Thu âm, Mở file Audio/Video hoặc nhấn 'Tạo Giọng Mẫu Thử Nghiệm' trước!")
            return

        self.btn_process.setEnabled(False)
        settings = self._collect_settings()
        settings.master_gain_db = 0.0  # Áp dụng riêng tức thời ở bước hiển thị để kéo thanh trượt mượt mà
        self._worker = ProcessWorker(self.engine, self.orig_audio, self.current_sr, settings)
        self._worker.progress.connect(self._on_process_progress)
        self._worker.finished_ok.connect(self._on_process_done)
        self._worker.failed.connect(self._on_process_failed)
        self._worker.start()

    def _on_process_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{pct}% — {msg}")

    def _on_process_done(self, out_audio: np.ndarray):
        self.btn_process.setEnabled(True)
        self.base_proc_audio = out_audio
        self._apply_live_volume_gain()
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("✅ Hoàn tất! Nhấn 'Nghe Bản Đã Xử Lý' để kiểm tra chất lượng.")

    def _on_process_failed(self, err: str):
        self.btn_process.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Lỗi khi xử lý âm thanh")
        QMessageBox.critical(self, "Lỗi xử lý", f"Đã xảy ra lỗi:\n{err}")

    def _load_tts_text_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn File Kịch Bản Văn Bản hoặc Phụ Đề",
            "",
            "Text & Subtitle Files (*.txt *.srt *.md);;All Files (*.*)",
        )
        if not path:
            return
        try:
            content = Path(path).read_text(encoding="utf-8", errors="ignore")
            if Path(path).suffix.lower() == ".srt":
                # Tự động lọc bỏ số thứ tự và mốc thời gian SRT để chỉ lấy lời thoại
                lines = []
                for line in content.splitlines():
                    s = line.strip()
                    if not s or s.isdigit() or "-->" in s:
                        continue
                    lines.append(s)
                content = " ".join(lines)
            self.txt_tts.setPlainText(content)
        except Exception as e:
            QMessageBox.warning(self, "Lỗi đọc file văn bản", f"Không thể đọc file:\n{e}")

    def _insert_tts_template(self):
        key = self.cmb_tts_sample.currentData()
        if not key:
            return
        templates = {
            "movie": (
                "Trong một thế giới nơi thời gian có thể mua bán bằng chính tuổi thọ của con người, "
                "nhân vật chính bất ngờ nhận được hơn một thế kỷ từ một người lạ mặt. "
                "Nhưng cũng chính từ giây phút đó, anh trở thành mục tiêu truy đuổi gắt gao nhất của tổ chức kiểm soát thời gian."
            ),
            "ad": (
                "Bạn đang tìm kiếm một giải pháp thu âm và lồng tiếng video chuyên nghiệp ngay tại nhà? "
                "Đừng bỏ qua công cụ xử lý giọng nói bằng trí tuệ nhân tạo thế hệ mới! "
                "Lọc sạch tạp âm chỉ với một cú nhấp chuột, biến đổi hàng chục giọng đọc tự nhiên, giúp video của bạn lên xu hướng ngay hôm nay!"
            ),
            "news": (
                "Bản tin công nghệ hôm nay xin gửi tới quý vị những cập nhật mới nhất về trí tuệ nhân tạo trong lĩnh vực sản xuất truyền thông. "
                "Các thuật toán xử lý tín hiệu số kết hợp mô hình giọng nói thần kinh đang giúp rút ngắn tới tám mươi phần trăm thời gian hậu kỳ âm thanh."
            ),
            "podcast": (
                "Có những buổi chiều cuối tuần, khi thành phố bắt đầu lên đèn, chúng ta chỉ muốn tìm một góc nhỏ bình yên để lắng nghe chính mình. "
                "Cảm ơn bạn đã ở đây, cùng mình đi qua những câu chuyện nhỏ nhẹ nhàng trong tập podcast tối nay."
            ),
        }
        if key in templates:
            self.txt_tts.setPlainText(templates[key])
        self.cmb_tts_sample.setCurrentIndex(0)

    def _update_tts_stats(self):
        text = self.txt_tts.toPlainText().strip()
        words = len(text.split()) if text else 0
        chars = len(text)
        est_sec = words / 3.0  # Trung bình tiếng Việt đọc ~180 từ/phút (3 từ/giây)
        self.lbl_tts_stats.setText(
            f"Số từ: {words} từ   |   Số ký tự: {chars}   |   Thời lượng đọc ước tính: ~{est_sec:.1f} giây"
        )

    def _run_edge_tts(self, save_after: bool = False):
        text = self.txt_tts.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Thiếu nội dung", "Vui lòng nhập kịch bản văn bản cần đọc.")
            return
        voice_id = self.cmb_tts_voice.currentData()
        rate_pct = self.sld_tts_rate.value()
        pitch_hz = self.sld_tts_pitch.value()

        self.btn_gen_tts.setEnabled(False)
        self.btn_tts_export_direct.setEnabled(False)
        self.progress_bar.setValue(35)
        self.progress_bar.setFormat("Đang chuyển văn bản thành giọng nói AI tự nhiên...")

        def _done(audio: np.ndarray, sr: int, msg: str):
            self.btn_gen_tts.setEnabled(True)
            self.btn_tts_export_direct.setEnabled(True)
            self._set_original_audio(audio, sr, f"Văn bản -> Giọng nói AI ({voice_id.split('|')[0]})", video_path=self.loaded_video_path)

            if self.chk_tts_auto_studio.isChecked():
                # Áp dụng thêm bộ lọc Studio EQ & đổi giọng ở Tab 1
                settings = self._collect_settings()
                settings.noise_reduction_enabled = False  # Giọng TTS đã sạch sẵn
                settings.master_gain_db = 0.0
                processed = self.engine.process_voice(audio, sr, settings)
                self.base_proc_audio = processed
            else:
                self.base_proc_audio = audio.copy()

            self._apply_live_volume_gain()
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("✅ Đã chuyển văn bản thành giọng nói thành công!")

            if self.chk_tts_autoplay.isChecked() and not save_after:
                self._play_target(self.proc_audio)
            if save_after:
                self._export_audio()

        def _fail(err: str):
            self.btn_gen_tts.setEnabled(True)
            self.btn_tts_export_direct.setEnabled(True)
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "Lỗi tạo giọng AI", f"Không thể tạo giọng đọc AI:\n{err}")

        self._ai_worker = AIWorker(
            "edge_tts",
            self.engine,
            {"text": text, "voice_id": voice_id, "rate_pct": rate_pct, "pitch_hz": pitch_hz},
        )
        self._ai_worker.finished_ok.connect(_done)
        self._ai_worker.failed.connect(_fail)
        self._ai_worker.start()

    def _run_ai_neural_sts(self):
        if self.orig_audio is None or len(self.orig_audio) == 0:
            QMessageBox.warning(
                self,
                "Chưa có giọng thu âm",
                "Vui lòng Thu âm, Mở file Audio/Video hoặc Tạo Giọng Mẫu trước khi thay đổi sang Giọng AI Neural!",
            )
            return

        voice_id = self.cmb_quick_ai_voice.currentData()
        self.btn_quick_ai_sts.setEnabled(False)
        self.progress_bar.setValue(35)
        self.progress_bar.setFormat("🤖 Đang nhận diện lời thoại & tái tạo bằng Giọng AI Neural siêu thực...")

        def _done(audio: np.ndarray, sr: int, transcript: str):
            self.btn_quick_ai_sts.setEnabled(True)
            self.current_sr = sr
            self.base_proc_audio = audio.copy()
            if transcript and hasattr(self, "txt_tts"):
                self.txt_tts.setPlainText(transcript)
            self._apply_live_volume_gain()
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat(f"✅ Đã tái tạo Giọng AI tròn vành rõ chữ: \"{transcript[:65]}\"")
            self._play_target(self.proc_audio)

        def _fail(err: str):
            self.btn_quick_ai_sts.setEnabled(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Không nhận diện được lời nói hoặc lỗi kết nối AI")
            QMessageBox.warning(self, "Thông báo AI Speech-to-Speech", err)

        self._ai_worker = AIWorker(
            "ai_neural_sts",
            self.engine,
            {"audio": self.orig_audio, "sr": self.current_sr, "voice_id": voice_id},
        )
        self._ai_worker.finished_ok.connect(_done)
        self._ai_worker.failed.connect(_fail)
        self._ai_worker.start()

    def _run_elevenlabs_sts(self):
        if self.orig_audio is None:
            QMessageBox.warning(self, "Chưa có giọng thu âm", "Vui lòng thu âm hoặc mở file giọng nói trước khi chuyển đổi bằng ElevenLabs.")
            return
        api_key = self.inp_xi_key.text().strip()
        voice_id = self.inp_xi_voice.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Thiếu API Key", "Vui lòng nhập ElevenLabs API Key của bạn.")
            return

        self.btn_run_sts.setEnabled(False)
        self.progress_bar.setValue(40)
        self.progress_bar.setFormat("Đang gửi giọng thu âm lên ElevenLabs Speech-to-Speech AI...")

        def _done(audio: np.ndarray, sr: int, msg: str):
            self.btn_run_sts.setEnabled(True)
            self.base_proc_audio = audio.copy()
            self._apply_live_volume_gain()
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat(msg)

        def _fail(err: str):
            self.btn_run_sts.setEnabled(True)
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "Lỗi ElevenLabs STS", err)

        source_for_sts = self.proc_audio if self.proc_audio is not None else self.orig_audio
        self._ai_worker = AIWorker(
            "elevenlabs",
            self.engine,
            {"audio": source_for_sts, "sr": self.current_sr, "api_key": api_key, "voice_id": voice_id},
        )
        self._ai_worker.finished_ok.connect(_done)
        self._ai_worker.failed.connect(_fail)
        self._ai_worker.start()

    # =========================================================================
    # XUẤT FILE AUDIO & GHÉP VÀO VIDEO
    # =========================================================================
    def _export_audio(self):
        target = self.proc_audio if self.proc_audio is not None else self.orig_audio
        if target is None:
            QMessageBox.warning(self, "Chưa có âm thanh", "Không có dữ liệu âm thanh để xuất.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu File Âm Thanh Chất Lượng Cao",
            "giong_da_xu_ly_studio.wav",
            "WAV Lossless 24-bit (*.wav);;MP3 320kbps (*.mp3)",
        )
        if not out_path:
            return
        try:
            self.engine.save_audio_file(target, self.current_sr, out_path)
            QMessageBox.information(self, "Xuất file thành công", f"Đã lưu file âm thanh tại:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi lưu file", f"Không thể xuất file:\n{e}")

    def _export_muxed_video(self):
        target = self.proc_audio if self.proc_audio is not None else self.orig_audio
        if target is None:
            QMessageBox.warning(self, "Chưa có âm thanh", "Vui lòng thu âm hoặc xử lý giọng nói trước khi ghép vào video.")
            return

        video_in = self.loaded_video_path
        if not video_in or not os.path.exists(video_in):
            video_in, _ = QFileDialog.getOpenFileName(
                self,
                "Chọn File Video Gốc Cần Thay Thế Giọng Nói",
                "",
                "Video Files (*.mp4 *.mov *.mkv *.avi *.webm)",
            )
            if not video_in:
                return

        default_out = str(Path(video_in).with_name(f"{Path(video_in).stem}_VoiceStudio.mp4"))
        video_out, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu Video Đã Ghép Giọng Mới",
            default_out,
            "MP4 Video (*.mp4);;MOV Video (*.mov);;MKV Video (*.mkv)",
        )
        if not video_out:
            return

        try:
            self.progress_bar.setValue(50)
            self.progress_bar.setFormat("Đang ghép âm thanh đã xử lý vào Video gốc...")
            QApplication.processEvents()
            self.engine.mux_audio_into_video(video_in, target, self.current_sr, video_out)
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("✅ Đã xuất Video hoàn chỉnh thành công!")
            QMessageBox.information(
                self,
                "Ghép Video Thành Công",
                f"Đã ghép giọng nói mới vào video (giữ nguyên 100% chất lượng hình ảnh gốc):\n{video_out}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Lỗi ghép Video", f"Không thể ghép âm thanh vào video:\n{e}")


def main():
    app = QApplication(sys.argv)
    win = VoiceStudioApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
