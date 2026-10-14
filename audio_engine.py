import os
import sys
import time
import wave
import shutil
import tempfile
import threading
import subprocess
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Dict

import numpy as np
import soundfile as sf
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d, maximum_filter1d, minimum_filter1d, uniform_filter
import librosa
import noisereduce as nr

try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    HAS_PARSELMOUTH = True
except Exception:
    HAS_PARSELMOUTH = False

try:
    from pedalboard import (
        Pedalboard,
        NoiseGate,
        HighpassFilter,
        LowpassFilter,
        LowShelfFilter,
        HighShelfFilter,
        PeakFilter,
        Compressor,
        Limiter,
        Reverb,
        Chorus,
        Clipping,
    )
    HAS_PEDALBOARD = True
except Exception:
    HAS_PEDALBOARD = False

try:
    import pyloudnorm as pyln
    HAS_PYLN = True
except Exception:
    HAS_PYLN = False

try:
    from pyrnnoise import RNNoise
    HAS_RNNOISE = True
except Exception:
    HAS_RNNOISE = False

try:
    import speech_recognition as sr_lib
    HAS_SR = True
except Exception:
    HAS_SR = False

try:
    import sounddevice as sd
    HAS_SD = True
except Exception:
    HAS_SD = False

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = shutil.which("ffmpeg")

try:
    import edge_tts
    HAS_EDGE_TTS = True
except Exception:
    HAS_EDGE_TTS = False

import requests


@dataclass
class VoiceSettings:
    # 1. Làm sạch, Khử tạp âm & Khử vang phòng (Noise, De-Reverb & Silence)
    noise_reduction_enabled: bool = True
    noise_reduction_strength: float = 0.90      # 0.0 -> 1.0 (đo trên bộ thử có đáp án: 0.9 sạch nền, rõ lời hơn rõ rệt so với 0.75)
    stationary_noise: bool = True               # True = 1 lượt (nhiễu nền ổn định); False = thêm lượt lọc nhiễu biến thiên
    dereverb_amount: float = 0.50               # Khử tiếng vang phòng / dội tường (0.0 -> 1.0)
    clarity_boost: float = 0.50                 # Tăng độ rõ lời / phụ âm tiếng Việt (0.0 -> 1.0)
    ai_denoise: bool = False                    # Khử nhiễu bằng mạng nơ-ron RNNoise (chất lượng cao, cần pyrnnoise)
    hum_removal: bool = True                    # Tự dò & cắt tiếng ù điện 50/60 Hz và các hài
    gap_attenuation_db: float = 18.0            # Hạ nền ồn còn sót trong khoảng nghỉ (dB, 0 = tắt)
    target_lufs: float = -14.0                  # Độ lớn đích khi bật chuẩn hóa (LUFS)
    remove_Background_music: bool = False       # Tách lời nói khỏi nhạc nền
    auto_trim_silence: bool = False             # Tự động cắt khoảng lặng/ngập ngừng
    silence_threshold_db: float = -38.0         # Ngưỡng dB coi là im lặng
    silence_min_pause_ms: int = 250             # Giữ lại khoảng nghỉ tự nhiên (ms)

    # 2. Cải thiện giọng Studio (Voice Enhancement)
    highpass_hz: float = 90.0                   # Lọc ù nền trầm (<90Hz giúp hết ồm)
    deesser_amount: float = 0.30                # Giảm âm xuýt (s, x, ch)
    bass_gain_db: float = 1.0                   # Độ ấm (140 Hz) - Không đẩy quá cao để tránh ồm/vang
    mid_gain_db: float = 2.5                    # Độ rõ lời (3.2 kHz)
    treble_gain_db: float = 1.8                 # Độ sáng / Chi tiết (7.5 kHz)
    compressor_amount: float = 0.60             # Nén đều âm lượng (0.0 -> 1.0)
    reverb_amount: float = 0.0                  # Mặc định = 0.0 (Tắt vang hoàn toàn để giọng khô & nét!)
    normalize_lufs: bool = True                 # Chuẩn hóa âm lượng chuẩn Video (-14 dB / -1 dBTP)
    master_gain_db: float = 0.0                 # Tăng/giảm âm lượng tổng (-20.0 dB -> +20.0 dB)

    # 3. Thay đổi giọng nói bằng TD-PSOLA (Praat Vocal Tract & Pitch)
    voice_preset: str = "studio_natural"        # Mã preset
    pitch_semitones: float = 0.0                # Cao độ (-12 -> +12 semitones)
    formant_shift: float = 0.0                  # Âm sắc thanh quản (-6 -> +6): âm = thanh quản rộng/Nam, dương = thanh quản nhỏ/Nữ
    pitch_range_factor: float = 1.0             # Độ biến thiên ngữ điệu (0.2 = trầm tĩnh/đều, 1.0 = tự nhiên, 1.35 = giàu cảm xúc)
    speaking_rate: float = 1.0                  # Tốc độ nói (0.75 -> 1.35)
    special_fx: str = "none"                    # Hiệu ứng đặc biệt: none, radio, robot, hall, megaphone


VOICE_CATEGORIES = [
    ("all", "🌟 Tất Cả Các Loại Giọng (25 Giọng)"),
    ("studio", "🎙️ Nhóm 1: Chuẩn Phòng Thu Khô & Rõ Lời (5 Giọng)"),
    ("male", "🧔 Nhóm 2: Giọng Nam Chân Thật Đa Phong Cách (5 Giọng)"),
    ("female", "👩 Nhóm 3: Giọng Nữ Chân Thật Đa Phong Cách (5 Giọng)"),
    ("broadcast", "🎬 Nhóm 4: Điện Ảnh, Truyền Hình & Kể Chuyện (5 Giọng)"),
    ("character", "🎭 Nhóm 5: Nhân Vật, Ẩn Danh & Hiệu Ứng Đặc Biệt (5 Giọng)"),
]


# Tất cả các Preset chuẩn đều đặt reverb_amount = 0.0 để giọng KHÔ, SÁT MIC, RÕ TỪNG CHỮ
# và thiết lập chênh lệch Pitch + Formant + Pitch Range rõ rệt bằng thuật toán Praat TD-PSOLA!
VOICE_PRESETS: Dict[str, dict] = {
    # ================= NHÓM 1: PHÒNG THU KHÔ & RÕ LỜI =================
    "studio_natural": {
        "category": "studio",
        "name": "🎙️ [Phòng Thu] Giọng Gốc Khử Vang & Rõ Lời Nhất (Khuyên dùng)",
        "desc": "Giữ nguyên giọng thật của bạn nhưng triệt tiêu tiếng vang phòng, cắt âm ồm đục và đẩy độ rõ từng chữ lên mức tối đa.",
        "pitch_semitones": 0.0,
        "formant_shift": 0.0,
        "pitch_range_factor": 1.0,
        "speaking_rate": 1.0,
        "bass_gain_db": 1.0,
        "mid_gain_db": 2.5,
        "treble_gain_db": 1.8,
        "compressor_amount": 0.60,
        "deesser_amount": 0.30,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "podcast_pro": {
        "category": "studio",
        "name": "🎧 [Phòng Thu] Podcast Shure SM7B (Ấm, Khô & Sát Mic)",
        "desc": "Mô phỏng chất âm micro thu âm chuyên nghiệp: giọng khô hoàn toàn không vang, âm trầm chắc gọn, lời nói nổi sát tai.",
        "pitch_semitones": -0.8,
        "formant_shift": -0.8,
        "pitch_range_factor": 0.95,
        "speaking_rate": 1.0,
        "bass_gain_db": 3.2,
        "mid_gain_db": 3.5,
        "treble_gain_db": 2.0,
        "compressor_amount": 0.72,
        "deesser_amount": 0.35,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "crispy_vlog": {
        "category": "studio",
        "name": "✨ [Phòng Thu] Vlog & TikTok Siêu Nét (Bắt Tai Trên Điện Thoại)",
        "desc": "Cắt sạch âm trầm ồm, đẩy mạnh dải tần rõ chữ (3kHz–6kHz) giúp giọng nghe cực kỳ sắc nét và nổi bật trên nhạc nền TikTok.",
        "pitch_semitones": 0.6,
        "formant_shift": 0.5,
        "pitch_range_factor": 1.15,
        "speaking_rate": 1.05,
        "bass_gain_db": -1.0,
        "mid_gain_db": 5.0,
        "treble_gain_db": 4.2,
        "compressor_amount": 0.70,
        "deesser_amount": 0.35,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "asmr_whisper": {
        "category": "studio",
        "name": "🌙 [Phòng Thu] Tâm Sự Đêm Khuya / Radio Ấm Áp",
        "desc": "Giọng trầm nhẹ, nhịp nói chậm rãi, êm dịu và gần gũi cho các video tâm sự, chữa lành, podcast đêm.",
        "pitch_semitones": -1.2,
        "formant_shift": -1.0,
        "pitch_range_factor": 0.85,
        "speaking_rate": 0.94,
        "bass_gain_db": 2.8,
        "mid_gain_db": 2.5,
        "treble_gain_db": 3.0,
        "compressor_amount": 0.75,
        "deesser_amount": 0.45,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "lecture_elearning": {
        "category": "studio",
        "name": "📚 [Phòng Thu] Giảng Bài / Thuyết Trình / Hướng Dẫn",
        "desc": "Tối ưu độ tách bạch của từng âm tiết, khử vang phòng tối đa để học viên nghe rõ từng từ mà không bị chói tai.",
        "pitch_semitones": 0.0,
        "formant_shift": 0.0,
        "pitch_range_factor": 1.05,
        "speaking_rate": 1.0,
        "bass_gain_db": 0.5,
        "mid_gain_db": 4.5,
        "treble_gain_db": 2.2,
        "compressor_amount": 0.65,
        "deesser_amount": 0.35,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },

    # ================= NHÓM 2: GIỌNG NAM ĐA PHONG CÁCH (TD-PSOLA) =================
    "warm_male_mc": {
        "category": "male",
        "name": "🧔 [Giọng Nam] Nam Trầm Ấm (MC / Review Phim)",
        "desc": "Mở rộng khoang thanh quản (-2.8) và hạ cao độ (-3.5 bán âm) bằng TD-PSOLA tạo giọng nam trầm ấm, khô nét, không vang.",
        "pitch_semitones": -3.5,
        "formant_shift": -2.8,
        "pitch_range_factor": 1.0,
        "speaking_rate": 1.0,
        "bass_gain_db": 2.8,
        "mid_gain_db": 3.8,
        "treble_gain_db": 2.2,
        "compressor_amount": 0.68,
        "deesser_amount": 0.30,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "deep_baritone": {
        "category": "male",
        "name": "🦁 [Giọng Nam] Nam Trầm Sâu Uy Lực (Trailer / Quảng Cáo)",
        "desc": "Giọng nam Baritone cực trầm (-5.2 bán âm, thanh quản rộng -4.0), đanh chắc và đầy uy lực cho TVC quảng cáo & Trailer.",
        "pitch_semitones": -5.2,
        "formant_shift": -4.0,
        "pitch_range_factor": 0.88,
        "speaking_rate": 0.96,
        "bass_gain_db": 3.8,
        "mid_gain_db": 4.2,
        "treble_gain_db": 2.5,
        "compressor_amount": 0.78,
        "deesser_amount": 0.30,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "young_male_genz": {
        "category": "male",
        "name": "🧑 [Giọng Nam] Nam Trẻ Trung Năng Động (GenZ / Gaming / Tech)",
        "desc": "Giọng nam trẻ trung, sáng rõ, ngữ điệu linh hoạt (1.2x) phù hợp kênh công nghệ, bình luận game, TikToker.",
        "pitch_semitones": -1.8,
        "formant_shift": -1.5,
        "pitch_range_factor": 1.20,
        "speaking_rate": 1.06,
        "bass_gain_db": 1.0,
        "mid_gain_db": 4.5,
        "treble_gain_db": 3.5,
        "compressor_amount": 0.65,
        "deesser_amount": 0.30,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "mature_narrator": {
        "category": "male",
        "name": "👴 [Giọng Nam] Nam Trung Niên Điềm Đạm (Lịch Sử / Quân Sự)",
        "desc": "Âm sắc nam trung niên chững chạc, ngữ điệu trầm tĩnh (0.78x) tạo sự uy nghiêm cho kênh lịch sử, hồ sơ vụ án.",
        "pitch_semitones": -4.2,
        "formant_shift": -3.4,
        "pitch_range_factor": 0.78,
        "speaking_rate": 0.95,
        "bass_gain_db": 3.0,
        "mid_gain_db": 4.0,
        "treble_gain_db": 1.8,
        "compressor_amount": 0.72,
        "deesser_amount": 0.30,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "old_storyteller": {
        "category": "male",
        "name": "🧙‍♂️ [Giọng Nam] Ông Lão Trầm Cổ Điển (Cổ Tích / Kiếm Hiệp)",
        "desc": "Giọng trầm già dặn (-6.0 bán âm), thanh quản mở rộng và nhịp kể thong thả phù hợp kể chuyện cổ tích, kiếm hiệp.",
        "pitch_semitones": -6.0,
        "formant_shift": -4.6,
        "pitch_range_factor": 0.82,
        "speaking_rate": 0.90,
        "bass_gain_db": 3.2,
        "mid_gain_db": 4.2,
        "treble_gain_db": 1.5,
        "compressor_amount": 0.75,
        "deesser_amount": 0.25,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },

    # ================= NHÓM 3: GIỌNG NỮ ĐA PHONG CÁCH (TD-PSOLA) =================
    "clear_female_vlog": {
        "category": "female",
        "name": "👩 [Giọng Nữ] Nữ Trong Trẻo Tự Nhiên (Vlog / TikTok)",
        "desc": "Thu nhỏ khoang thanh quản (+3.6) và nâng cao độ (+4.5 bán âm) bằng TD-PSOLA tạo giọng nữ sáng rõ, tự nhiên, không vang.",
        "pitch_semitones": 4.5,
        "formant_shift": 3.6,
        "pitch_range_factor": 1.12,
        "speaking_rate": 1.02,
        "bass_gain_db": -2.0,
        "mid_gain_db": 4.0,
        "treble_gain_db": 3.8,
        "compressor_amount": 0.62,
        "deesser_amount": 0.35,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "sweet_female_story": {
        "category": "female",
        "name": "🌸 [Giọng Nữ] Nữ Dịu Dàng Truyền Cảm (Radio Tình Cảm / Truyện)",
        "desc": "Giọng nữ nhẹ nhàng, ngọt ngào (+3.8 bán âm, thanh quản +3.0), nhịp đọc êm ái cho tản văn và tâm sự.",
        "pitch_semitones": 3.8,
        "formant_shift": 3.0,
        "pitch_range_factor": 1.05,
        "speaking_rate": 0.96,
        "bass_gain_db": -0.5,
        "mid_gain_db": 3.5,
        "treble_gain_db": 3.2,
        "compressor_amount": 0.65,
        "deesser_amount": 0.40,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "pro_female_anchor": {
        "category": "female",
        "name": "👩‍💼 [Giọng Nữ] Nữ Biên Tập Viên Sắc Sảo (Tin Tức / Tài Chính)",
        "desc": "Giọng nữ chững chạc (+3.0 bán âm), phát âm đanh gọn, cực kỳ rõ chữ cho bản tin kinh tế, điểm tin nhanh.",
        "pitch_semitones": 3.0,
        "formant_shift": 2.4,
        "pitch_range_factor": 0.95,
        "speaking_rate": 1.04,
        "bass_gain_db": -1.0,
        "mid_gain_db": 4.8,
        "treble_gain_db": 3.0,
        "compressor_amount": 0.72,
        "deesser_amount": 0.35,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "cute_anime_girl": {
        "category": "female",
        "name": "🎀 [Giọng Nữ] Nữ Cao Đáng Yêu (Anime / VTuber / Giải Trí)",
        "desc": "Giọng nữ cao trong trẻo (+6.5 bán âm, thanh quản +5.0), ngữ điệu biểu cảm sống động kiểu Anime / VTuber.",
        "pitch_semitones": 6.5,
        "formant_shift": 5.0,
        "pitch_range_factor": 1.30,
        "speaking_rate": 1.05,
        "bass_gain_db": -3.5,
        "mid_gain_db": 4.2,
        "treble_gain_db": 4.5,
        "compressor_amount": 0.65,
        "deesser_amount": 0.40,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "mature_lady": {
        "category": "female",
        "name": "👵 [Giọng Nữ] Nữ Trầm Từng Trải (Kể Chuyện Gia Đình / Đời Sống)",
        "desc": "Giọng nữ trầm đằm thắm (+2.0 bán âm, thanh quản +1.8), sâu lắng cho kênh kể chuyện đời sống, tâm lý xã hội.",
        "pitch_semitones": 2.0,
        "formant_shift": 1.8,
        "pitch_range_factor": 0.90,
        "speaking_rate": 0.95,
        "bass_gain_db": 1.0,
        "mid_gain_db": 3.8,
        "treble_gain_db": 2.2,
        "compressor_amount": 0.65,
        "deesser_amount": 0.35,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },

    # ================= NHÓM 4: ĐIỆN ẢNH, TRUYỀN HÌNH & KỂ CHUYỆN =================
    "deep_storyteller": {
        "category": "broadcast",
        "name": "🎬 [Điện Ảnh] Giọng Thuyết Minh Điện Ảnh (Review / Tóm Tắt Phim)",
        "desc": "Giọng trầm dày, đanh gọn, tốc độ lôi cuốn chuẩn các kênh Tóm Tắt Phim / Review Phim triệu view.",
        "pitch_semitones": -3.0,
        "formant_shift": -2.4,
        "pitch_range_factor": 1.08,
        "speaking_rate": 1.08,
        "bass_gain_db": 2.2,
        "mid_gain_db": 4.8,
        "treble_gain_db": 3.0,
        "compressor_amount": 0.78,
        "deesser_amount": 0.30,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "news_broadcaster": {
        "category": "broadcast",
        "name": "📺 [Truyền Hình] Phát Thanh Viên Thời Sự (Đanh, Rõ Từng Chữ)",
        "desc": "Tập trung tối đa vào dải tần trung cao (Mid-Presence +5dB), giọng đanh, phát âm sắc nét chuẩn đài truyền hình.",
        "pitch_semitones": -0.8,
        "formant_shift": -0.8,
        "pitch_range_factor": 0.88,
        "speaking_rate": 1.02,
        "bass_gain_db": 1.0,
        "mid_gain_db": 5.2,
        "treble_gain_db": 3.2,
        "compressor_amount": 0.75,
        "deesser_amount": 0.30,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "audiobook_warm": {
        "category": "broadcast",
        "name": "📖 [Sách Nói] Đọc Sách Audiobook (Ấm Áp, Khô Tĩnh Tuyệt Đối)",
        "desc": "Chuẩn âm thanh Audible: nền âm tĩnh lặng tuyệt đối, giọng rõ ràng, ấm áp vừa phải để nghe lâu hàng giờ.",
        "pitch_semitones": -1.5,
        "formant_shift": -1.2,
        "pitch_range_factor": 0.95,
        "speaking_rate": 0.98,
        "bass_gain_db": 2.0,
        "mid_gain_db": 3.6,
        "treble_gain_db": 2.0,
        "compressor_amount": 0.65,
        "deesser_amount": 0.40,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "horror_mystery": {
        "category": "broadcast",
        "name": "👻 [Kinh Dị] Kể Chuyện Ma / Trinh Thám Đêm Khuya",
        "desc": "Giọng trầm lạnh (-4.5 bán âm), ngữ điệu u ám (0.70x), nhịp chậm tạo cảm giác rùng rợn nhưng vẫn rõ từng lời.",
        "pitch_semitones": -4.5,
        "formant_shift": -3.5,
        "pitch_range_factor": 0.70,
        "speaking_rate": 0.91,
        "bass_gain_db": 3.2,
        "mid_gain_db": 4.0,
        "treble_gain_db": 2.5,
        "compressor_amount": 0.75,
        "deesser_amount": 0.30,
        "reverb_amount": 0.08,
        "special_fx": "none",
    },
    "sports_commentator": {
        "category": "broadcast",
        "name": "⚽ [Thể Thao] Bình Luận Viên Sôi Động (Nén Mạnh, Đầy Lửa)",
        "desc": "Ngữ điệu bùng nổ (1.35x), nén dải động cực mạnh và đẩy cao độ rõ lời để luôn nổi bật trên nền tiếng cổ vũ.",
        "pitch_semitones": 0.8,
        "formant_shift": 0.4,
        "pitch_range_factor": 1.35,
        "speaking_rate": 1.07,
        "bass_gain_db": 1.2,
        "mid_gain_db": 5.5,
        "treble_gain_db": 3.8,
        "compressor_amount": 0.85,
        "deesser_amount": 0.30,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },

    # ================= NHÓM 5: NHÂN VẬT, ẨN DANH & HIỆU ỨNG ĐẶC BIỆT =================
    "young_energetic": {
        "category": "character",
        "name": "👦 [Nhân Vật] Giọng Trẻ Em / Hoạt Hình Vui Nhộn",
        "desc": "Thu nhỏ mạnh khoang thanh quản (+5.5) và nâng cao độ (+7.5 bán âm) tạo giọng trẻ em / nhân vật hoạt hình rõ nét.",
        "pitch_semitones": 7.5,
        "formant_shift": 5.5,
        "pitch_range_factor": 1.35,
        "speaking_rate": 1.06,
        "bass_gain_db": -4.0,
        "mid_gain_db": 4.5,
        "treble_gain_db": 3.5,
        "compressor_amount": 0.68,
        "deesser_amount": 0.35,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "anonymous_witness": {
        "category": "character",
        "name": "🕵️ [Ẩn Danh] Nhân Chứng Ẩn Danh Phóng Sự (Bảo Mật Giọng)",
        "desc": "Biến đổi mạnh cấu trúc thanh quản (-5.0) và hạ trầm (-6.5 bán âm) để không ai nhận ra giọng gốc mà vẫn nghe cực rõ.",
        "pitch_semitones": -6.5,
        "formant_shift": -5.0,
        "pitch_range_factor": 0.65,
        "speaking_rate": 1.0,
        "bass_gain_db": 2.0,
        "mid_gain_db": 4.8,
        "treble_gain_db": 1.5,
        "compressor_amount": 0.78,
        "deesser_amount": 0.25,
        "reverb_amount": 0.0,
        "special_fx": "none",
    },
    "vintage_radio": {
        "category": "character",
        "name": "📻 [Hiệu Ứng] Loa Phát Thanh Cổ Điển / Bộ Đàm (Radio)",
        "desc": "Mô phỏng âm thanh qua loa đài Radio hoặc bộ đàm liên lạc quân sự (lọc dải 450Hz–3.4kHz & nén bão hòa).",
        "pitch_semitones": 0.0,
        "formant_shift": 0.0,
        "pitch_range_factor": 0.90,
        "speaking_rate": 1.0,
        "bass_gain_db": -6.0,
        "mid_gain_db": 6.0,
        "treble_gain_db": -5.0,
        "compressor_amount": 0.85,
        "deesser_amount": 0.20,
        "reverb_amount": 0.0,
        "special_fx": "radio",
    },
    "cyber_robot": {
        "category": "character",
        "name": "🤖 [Hiệu Ứng] Trí Tuệ Nhân Tạo AI / Robot (Giọng Phẳng Điện Tử)",
        "desc": "Đưa độ biến thiên ngữ điệu về mức phẳng (Monotone 0.15x) kết hợp hiệu ứng âm thanh điện tử Sci-Fi.",
        "pitch_semitones": -2.0,
        "formant_shift": -1.5,
        "pitch_range_factor": 0.15,
        "speaking_rate": 1.0,
        "bass_gain_db": 1.0,
        "mid_gain_db": 4.5,
        "treble_gain_db": 3.0,
        "compressor_amount": 0.80,
        "deesser_amount": 0.30,
        "reverb_amount": 0.0,
        "special_fx": "robot",
    },
    "giant_boss": {
        "category": "character",
        "name": "👹 [Nhân Vật] Khổng Lồ / Chúa Tể Điện Ảnh (Siêu Trầm Uy Áp)",
        "desc": "Hạ cực sâu cao độ (-8.5 bán âm) và mở rộng khoang thanh quản cực đại (-5.8) cho nhân vật Boss / Thần thoại.",
        "pitch_semitones": -8.5,
        "formant_shift": -5.8,
        "pitch_range_factor": 0.75,
        "speaking_rate": 0.88,
        "bass_gain_db": 4.5,
        "mid_gain_db": 4.5,
        "treble_gain_db": 2.0,
        "compressor_amount": 0.85,
        "deesser_amount": 0.25,
        "reverb_amount": 0.12,
        "special_fx": "hall",
    },
}


# Danh sách mở rộng 27 Giọng đọc & Phong cách AI Neural (Edge-TTS)
EDGE_TTS_VOICES = [
    # --- Tiếng Việt: Đa dạng phong cách đã tinh chỉnh sẵn ---
    ("vi-VN-NamMinhNeural|0|0", "🇻🇳 Tiếng Việt - Nam Minh (Nam chuẩn Tự Nhiên)"),
    ("vi-VN-NamMinhNeural|-6|-10", "🇻🇳 Tiếng Việt - Nam Minh (Nam Trầm Ấm - Kể Chuyện / Podcast)"),
    ("vi-VN-NamMinhNeural|+14|-4", "🇻🇳 Tiếng Việt - Nam Minh (Review Phim Nhanh & Cuốn Hút)"),
    ("vi-VN-NamMinhNeural|+6|+6", "🇻🇳 Tiếng Việt - Nam Minh (Nam Trẻ Trung - Vlog / TikTok)"),
    ("vi-VN-NamMinhNeural|-5|-18", "🇻🇳 Tiếng Việt - Nam Minh (Nam Trầm Sâu - Phim Tài Liệu / Lịch Sử)"),
    ("vi-VN-HoaiMyNeural|0|0", "🇻🇳 Tiếng Việt - Hoài My (Nữ chuẩn Tự Nhiên)"),
    ("vi-VN-HoaiMyNeural|-8|-6", "🇻🇳 Tiếng Việt - Hoài My (Nữ Dịu Dàng - Radio Tâm Sự / Đọc Truyện)"),
    ("vi-VN-HoaiMyNeural|+12|+5", "🇻🇳 Tiếng Việt - Hoài My (Nữ Sôi Nổi - Review / TikTok / Bán Hàng)"),
    ("vi-VN-HoaiMyNeural|+5|-3", "🇻🇳 Tiếng Việt - Hoài My (Nữ Biên Tập Viên - Tin Tức / Thời Sự)"),
    ("vi-VN-HoaiMyNeural|+10|+15", "🇻🇳 Tiếng Việt - Hoài My (Nữ Cao Đáng Yêu - Hoạt Hình / Giải Trí)"),

    # --- Đa ngôn ngữ (Đọc được cả Tiếng Việt có dấu & Tiếng Anh chuẩn Quốc tế) ---
    ("en-US-AndrewMultilingualNeural|0|0", "🌍 Đa Ngôn Ngữ - Andrew (Nam Mỹ Trầm Ấm Chuẩn Studio)"),
    ("en-US-AvaMultilingualNeural|0|0", "🌍 Đa Ngôn Ngữ - Ava (Nữ Mỹ Tự Nhiên Chuẩn Studio)"),
    ("en-US-BrianMultilingualNeural|0|0", "🌍 Đa Ngôn Ngữ - Brian (Nam Mỹ Trẻ Trung Năng Động)"),
    ("en-US-EmmaMultilingualNeural|0|0", "🌍 Đa Ngôn Ngữ - Emma (Nữ Mỹ Truyền Cảm, Ấm Áp)"),
    ("fr-FR-RemyMultilingualNeural|0|0", "🌍 Đa Ngôn Ngữ - Rémy (Nam Pháp Lịch Lãm)"),
    ("fr-FR-VivienneMultilingualNeural|0|0", "🌍 Đa Ngôn Ngữ - Vivienne (Nữ Pháp Sang Trọng)"),
    ("de-DE-FlorianMultilingualNeural|0|0", "🌍 Đa Ngôn Ngữ - Florian (Nam Đức Chững Chạc)"),
    ("de-DE-SeraphinaMultilingualNeural|0|0", "🌍 Đa Ngôn Ngữ - Seraphina (Nữ Đức Trong Trẻo)"),
    ("ko-KR-HyunsuMultilingualNeural|0|0", "🌍 Đa Ngôn Ngữ - Hyunsu (Nam Hàn Quốc Trầm Ấm)"),

    # --- Tiếng Anh & Châu Á chuyên biệt ---
    ("en-US-ChristopherNeural|0|0", "🇺🇸 Tiếng Anh US - Christopher (Nam Thuyết Minh Điện Ảnh)"),
    ("en-US-JennyNeural|0|0", "🇺🇸 Tiếng Anh US - Jenny (Nữ Trợ Lý AI Tự Nhiên)"),
    ("en-GB-RyanNeural|0|0", "🇬🇧 Tiếng Anh UK - Ryan (Nam Quý Ông Anh Quốc)"),
    ("en-GB-SoniaNeural|0|0", "🇬🇧 Tiếng Anh UK - Sonia (Nữ Phát Thanh BBC Anh Quốc)"),
    ("ja-JP-KeitaNeural|0|0", "🇯🇵 Tiếng Nhật - Keita (Nam Anime Nhật Bản)"),
    ("ja-JP-NanamiNeural|0|0", "🇯🇵 Tiếng Nhật - Nanami (Nữ Anime Nhật Bản)"),
    ("zh-CN-YunxiNeural|0|0", "🇨🇳 Tiếng Trung - Yunxi (Nam Review Phim Douyin)"),
    ("zh-CN-XiaoxiaoNeural|0|0", "🇨🇳 Tiếng Trung - Xiaoxiao (Nữ Ngọt Ngào Douyin)"),
]


ELEVENLABS_VOICES = [
    ("JBFqnCBsd6RMkjVDRZzb", "🧔 George — Nam Trầm Ấm Kể Chuyện (Multilingual v2)"),
    ("pNInz6obpgDQGcFmaJgB", "🦁 Adam — Nam Trầm Sâu Thuyết Minh Nổi Tiếng Nhất"),
    ("ErXwobaYiN019PkySvjV", "🧑 Antoni — Nam Trẻ Trung, Rõ Ràng, Tự Nhiên"),
    ("nPczCjzI2devNBz1zQrb", "🎬 Brian — Nam Trầm Điện Ảnh & Trailer"),
    ("N2lVS1w4EtoT3dr4eOWO", "🕵️ Callum — Nam Khàn Đặc Trưng Nhân Vật"),
    ("IKne3meq5aSn9XLyUdCD", "🎙️ Charlie — Nam Tự Nhiên Đời Thường"),
    ("onwK4e9ZLuTAKqWW03F9", "📺 Daniel — Nam Phát Thanh Viên Tin Tức Uy Tín"),
    ("TxGEqnHWrfWFTfGW9XjX", "👦 Josh — Nam Trầm Trẻ Trung (Vlog / Podcast)"),
    ("TX3LPaxmHKxFdv7VOQHJ", "⚡ Liam — Nam Năng Động, Hiện Đại"),
    ("21m00Tcm4TlvDq8ikWAM", "👩 Rachel — Nữ Chuẩn Studio Tự Nhiên Nổi Tiếng Nhất"),
    ("EXAVITQu4vr4xnSDxMaL", "🌸 Bella — Nữ Dịu Dàng, Mềm Mại, Kể Chuyện"),
    ("XB0fDUnXU5powFXDhCwa", "👩‍💼 Charlotte — Nữ Sang Trọng, Truyền Cảm"),
    ("AZnzlk1XvdvUeBnXmlld", "✨ Domi — Nữ Sôi Nổi, Mạnh Mẽ, Cá Tính"),
    ("MF3mGyEYCl7XYWbV9V6O", "🎀 Elli — Nữ Trẻ Trung, Trong Trẻo"),
    ("XrExE9yKIg1WjnnlVkGX", "📖 Matilda — Nữ Ấm Áp Đọc Sách Nói (Audiobook)"),
    ("EXAVITQu4vr4xnSDxMaL", "📰 Sarah — Nữ Điểm Tin Nhanh & Rõ Chữ"),
]


class AudioEngine:
    def __init__(self, target_sr: int = 44100):
        self.target_sr = target_sr
        self.is_recording = False
        self.is_playing = False
        self.is_paused = False
        self.playing_track_id: Optional[str] = None  # "orig" hoặc "proc"
        self._record_frames: List[np.ndarray] = []
        self._record_stream = None
        self._play_stream = None
        self._play_buffer: Optional[np.ndarray] = None
        self._play_sr: int = target_sr
        self._play_pos: int = 0
        self._on_play_finish: Optional[Callable[[], None]] = None
        self._play_lock = threading.Lock()

    # =========================================================================
    # 1. THIẾT BỊ THU ÂM & PHÁT ÂM THANH (HỖ TRỢ PAUSE / RESUME CHÍNH XÁC TỪNG GIÂY)
    # =========================================================================
    def get_input_devices(self) -> List[Tuple[int, str]]:
        if not HAS_SD:
            return []
        devices = []
        try:
            all_devs = sd.query_devices()
            seen_names = set()
            for idx, dev in enumerate(all_devs):
                if dev.get("max_input_channels", 0) > 0:
                    name = dev.get("name", f"Microphone {idx}").strip()
                    if name not in seen_names:
                        seen_names.add(name)
                        devices.append((idx, name))
        except Exception:
            pass
        return devices

    def start_recording(
        self,
        device_index: Optional[int] = None,
        level_callback: Optional[Callable[[float], None]] = None,
    ):
        if not HAS_SD:
            raise RuntimeError("Thư viện sounddevice chưa sẵn sàng hoặc máy không có Microphone.")
        if self.is_recording:
            return

        self.stop_playback()
        self._record_frames = []
        self.is_recording = True

        def _callback(indata, frames, time_info, status):
            if not self.is_recording:
                return
            mono = indata[:, 0].copy()
            self._record_frames.append(mono)
            if level_callback is not None:
                rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-9)
                db = max(0.0, min(1.0, (20 * np.log10(rms) + 55.0) / 55.0))
                level_callback(db)

        self._record_stream = sd.InputStream(
            samplerate=self.target_sr,
            device=device_index,
            channels=1,
            dtype="float32",
            blocksize=2048,
            latency="high",
            callback=_callback,
        )
        self._record_stream.start()

    def stop_recording(self) -> Tuple[np.ndarray, int]:
        self.is_recording = False
        if self._record_stream is not None:
            try:
                self._record_stream.stop()
                self._record_stream.close()
            except Exception:
                pass
            self._record_stream = None

        if not self._record_frames:
            return np.zeros(int(self.target_sr * 0.5), dtype=np.float32), self.target_sr

        audio = np.concatenate(self._record_frames, axis=0).astype(np.float32)
        return audio, self.target_sr

    def toggle_play_pause(
        self,
        audio: np.ndarray,
        sr: int,
        track_id: str = "proc",
        on_finish: Optional[Callable[[], None]] = None,
    ) -> str:
        """
        Chuyển đổi thông minh giữa Phát (Playing) <-> Tạm dừng (Paused) <-> Tiếp tục (Resumed).
        Trả về trạng thái mới: "playing", "paused", hoặc "stopped".
        """
        if not HAS_SD or audio is None or len(audio) == 0:
            if on_finish:
                on_finish()
            return "stopped"

        with self._play_lock:
            # Nếu đang phát hoặc tạm dừng cùng một track -> Toggle Pause / Resume
            if self.is_playing and self.playing_track_id == track_id and self._play_stream is not None:
                self.is_paused = not self.is_paused
                return "paused" if self.is_paused else "playing"

        # Nếu khác track hoặc chưa phát -> Bắt đầu phát mới từ đầu
        self.play_audio(audio, sr, on_finish=on_finish, track_id=track_id)
        return "playing"

    def update_playing_buffer(self, audio: np.ndarray, track_id: str = "proc"):
        """Cập nhật dữ liệu âm thanh đang phát theo thời gian thực (ví dụ khi kéo thanh Âm lượng)."""
        with self._play_lock:
            if self.is_playing and self.playing_track_id == track_id and audio is not None:
                new_buf = np.ascontiguousarray(np.clip(audio, -1.0, 1.0), dtype=np.float32)
                if self._play_buffer is not None and len(new_buf) == len(self._play_buffer):
                    self._play_buffer = new_buf

    def get_playback_time(self) -> Tuple[float, float]:
        """Trả về (thời_gian_hiện_tại_giây, tổng_thời_lượng_giây)."""
        with self._play_lock:
            if not self.is_playing or self._play_buffer is None or self._play_sr <= 0:
                return 0.0, 0.0
            cur = float(self._play_pos) / float(self._play_sr)
            tot = float(len(self._play_buffer)) / float(self._play_sr)
            return min(cur, tot), tot

    def play_audio(
        self,
        audio: np.ndarray,
        sr: int,
        on_finish: Optional[Callable[[], None]] = None,
        track_id: str = "proc",
    ):
        if not HAS_SD or audio is None or len(audio) == 0:
            if on_finish:
                on_finish()
            return

        self.stop_playback()

        with self._play_lock:
            self._play_buffer = np.ascontiguousarray(np.clip(audio, -1.0, 1.0), dtype=np.float32)
            self._play_sr = int(sr)
            self._play_pos = 0
            self.is_playing = True
            self.is_paused = False
            self.playing_track_id = track_id
            self._on_play_finish = on_finish

            def _out_callback(outdata, frames, time_info, status):
                if not self.is_playing or self._play_buffer is None:
                    outdata.fill(0.0)
                    raise sd.CallbackStop()
                if self.is_paused:
                    outdata.fill(0.0)
                    return

                buf = self._play_buffer
                rem = len(buf) - self._play_pos
                if rem <= 0:
                    outdata.fill(0.0)
                    raise sd.CallbackStop()

                n_copy = min(frames, rem)
                outdata[:n_copy, 0] = buf[self._play_pos : self._play_pos + n_copy]
                if n_copy < frames:
                    outdata[n_copy:, 0] = 0.0
                    self._play_pos += n_copy
                    raise sd.CallbackStop()
                self._play_pos += n_copy

            def _finished_callback():
                cb = None
                with self._play_lock:
                    self.is_playing = False
                    self.is_paused = False
                    self.playing_track_id = None
                    self._play_pos = 0
                    cb = self._on_play_finish
                if cb:
                    try:
                        cb()
                    except Exception:
                        pass

            try:
                self._play_stream = sd.OutputStream(
                    samplerate=self._play_sr,
                    channels=1,
                    dtype="float32",
                    blocksize=2048,
                    latency="high",
                    callback=_out_callback,
                    finished_callback=_finished_callback,
                )
                self._play_stream.start()
            except Exception:
                self.is_playing = False
                self.is_paused = False
                self.playing_track_id = None

    def stop_playback(self):
        stream = None
        cb = None
        with self._play_lock:
            stream = self._play_stream
            self._play_stream = None
            was_playing = self.is_playing
            self.is_playing = False
            self.is_paused = False
            self.playing_track_id = None
            self._play_pos = 0
            cb = self._on_play_finish if was_playing else None
            self._on_play_finish = None

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        if HAS_SD:
            try:
                sd.stop()
            except Exception:
                pass
        if cb:
            try:
                cb()
            except Exception:
                pass

    # =========================================================================
    # 2. ĐỌC / XUẤT FILE ÂM THANH & VIDEO (FFMPEG)
    # =========================================================================
    def load_media_file(self, file_path: str) -> Tuple[np.ndarray, int, bool]:
        ext = Path(file_path).suffix.lower()
        video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv"}
        is_video = ext in video_exts

        if not is_video and ext in {".wav", ".flac", ".ogg"}:
            data, sr = sf.read(file_path, dtype="float32", always_2d=True)
            mono = np.mean(data, axis=1)
            if sr != self.target_sr:
                mono = librosa.resample(mono, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            return mono.astype(np.float32), sr, False

        if FFMPEG_PATH and os.path.exists(FFMPEG_PATH):
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            try:
                cmd = [
                    FFMPEG_PATH,
                    "-y",
                    "-i", file_path,
                    "-vn",
                    "-ac", "1",
                    "-ar", str(self.target_sr),
                    "-acodec", "pcm_s16le",
                    tmp_wav,
                ]
                creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
                data, sr = sf.read(tmp_wav, dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                return data.astype(np.float32), sr, is_video
            finally:
                if os.path.exists(tmp_wav):
                    try:
                        os.remove(tmp_wav)
                    except Exception:
                        pass

        mono, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        return mono.astype(np.float32), sr, is_video

    def save_audio_file(self, audio: np.ndarray, sr: int, out_path: str):
        ext = Path(out_path).suffix.lower()
        audio_clipped = np.clip(audio, -1.0, 1.0)
        if ext == ".mp3" and FFMPEG_PATH:
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            try:
                sf.write(tmp_wav, audio_clipped, sr, subtype="PCM_16")
                cmd = [
                    FFMPEG_PATH, "-y",
                    "-i", tmp_wav,
                    "-codec:a", "libmp3lame",
                    "-b:a", "320k",
                    out_path,
                ]
                creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
            finally:
                if os.path.exists(tmp_wav):
                    try:
                        os.remove(tmp_wav)
                    except Exception:
                        pass
        else:
            sf.write(out_path, audio_clipped, sr, subtype="PCM_24")

    def mux_audio_into_video(self, video_path: str, audio: np.ndarray, sr: int, out_video_path: str):
        if not FFMPEG_PATH:
            raise RuntimeError("Không tìm thấy công cụ FFmpeg để ghép video.")

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            sf.write(tmp_wav, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
            cmd = [
                FFMPEG_PATH,
                "-y",
                "-i", video_path,
                "-i", tmp_wav,
                "-c:v", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:a", "aac",
                "-b:a", "256k",
                "-shortest",
                out_video_path,
            ]
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
        finally:
            if os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass

    # =========================================================================
    # 3. CHUỖI XỬ LÝ ÂM THANH & ĐỔI GIỌNG CHÂN THẬT (TD-PSOLA + SPOTIFY PEDALBOARD)
    # =========================================================================
    def process_voice(
        self,
        audio: np.ndarray,
        sr: int,
        settings: VoiceSettings,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> np.ndarray:
        if audio is None or len(audio) == 0:
            return audio

        def report(pct: int, msg: str):
            if progress_cb:
                progress_cb(pct, msg)

        y = audio.astype(np.float32).copy()
        y = y - np.mean(y)

        # Bước 0: Cắt tiếng ù điện (50/60 Hz + hài) và ù trầm TRƯỚC khi khử nhiễu,
        # để bộ khử nhiễu không phải "học" các vạch ù và chừa sức xử lý cho nhiễu khác.
        if settings.hum_removal:
            report(5, "Đang dò & cắt tiếng ù điện...")
            y = self._remove_hum(y, sr)
        if settings.highpass_hz > 25.0:
            sos_hp = signal.butter(4, float(settings.highpass_hz), btype="highpass", fs=sr, output="sos")
            y = signal.sosfiltfilt(sos_hp, y).astype(np.float32)

        # Cân bằng mức tín hiệu đầu vào (Input Gain Staging) để bộ lọc làm việc ở vùng chuẩn.
        y = self._pre_normalize_input(y, sr)

        # Bước 1: Tách lời nói khỏi nhạc nền (nếu bật)
        if settings.remove_Background_music:
            report(12, "Đang tách giọng nói khỏi nhạc nền (Vocal Isolation)...")
            y = self._isolate_vocals(y, sr)

        # Bước 2: Khử tạp âm nền
        if settings.noise_reduction_enabled and settings.noise_reduction_strength > 0.01:
            strength = float(settings.noise_reduction_strength)
            if settings.ai_denoise and HAS_RNNOISE:
                report(25, "Đang khử tạp âm bằng AI (RNNoise)...")
                y = self._denoise_rnnoise(y, sr, strength)
                # Một lượt phổ nhẹ để dọn hiss còn sót mà RNNoise bỏ qua
                y = self._reduce_noise_crisp(y, sr, prop_decrease=strength * 0.5, stationary=True)
            else:
                report(25, "Đang khử tạp âm nền (2 lượt: ổn định + biến thiên)...")
                y = self._reduce_noise_crisp(
                    y,
                    sr,
                    prop_decrease=strength,
                    stationary=settings.stationary_noise,
                )
            # Hạ nền ồn còn sót trong các khoảng nghỉ giữa câu (có giữ trước/sau từ để không mất chữ)
            if settings.gap_attenuation_db > 0.5:
                y = self._gap_expander(y, sr, depth_db=settings.gap_attenuation_db * (0.5 + 0.5 * strength))

        # Bước 3: Khử vang phòng (suy giảm đuôi vang muộn + cắt cộng hưởng phòng)
        if settings.dereverb_amount > 0.02:
            report(40, "Đang khử vang phòng & làm trong giọng...")
            y = self._remove_room_reverb(y, sr, amount=settings.dereverb_amount)

        # Bước 4: Tự động cắt bỏ khoảng lặng / ngập ngừng (chỉ khi người dùng chủ động bật)
        if settings.auto_trim_silence:
            report(52, "Đang cắt bỏ các đoạn im lặng dài...")
            y = self._trim_silence(
                y,
                sr,
                top_db=abs(settings.silence_threshold_db),
                pad_ms=settings.silence_min_pause_ms,
            )

        # Bước 5: Đổi giọng bằng TD-PSOLA (Praat). Giọng gốc (pitch=0, formant=0, rate=1.0) giữ nguyên sóng âm.
        if (
            abs(settings.pitch_semitones) > 0.05
            or abs(settings.formant_shift) > 0.05
            or abs(settings.pitch_range_factor - 1.0) > 0.03
            or abs(settings.speaking_rate - 1.0) > 0.02
        ):
            report(68, "Đang biến đổi thanh quản & cao độ bằng thuật toán TD-PSOLA...")
            y = self._shift_voice_psola(
                y,
                sr,
                pitch_semitones=settings.pitch_semitones,
                formant_semitones=settings.formant_shift,
                pitch_range_factor=settings.pitch_range_factor,
                speaking_rate=settings.speaking_rate,
            )

        # Bước 6: EQ làm rõ lời + nén + de-esser (chưa limiter; limiter đặt ở cuối chuỗi)
        report(84, "Đang cân bằng âm sắc EQ & tăng độ rõ lời chuẩn Phòng thu...")
        y = self._apply_studio_pedalboard(y, sr, settings)

        # Bước 7: Hiệu ứng đặc biệt (Radio / Robot / Hall) nếu người dùng chủ động chọn
        if settings.special_fx and settings.special_fx != "none":
            report(92, f"Đang áp dụng hiệu ứng âm học: {settings.special_fx}...")
            y = self._apply_special_fx(y, sr, fx_type=settings.special_fx)

        # Bước 8: Chuẩn hóa độ lớn (LUFS thật) rồi limiter đỉnh cuối cùng
        if settings.normalize_lufs:
            y = self._normalize_loudness(y, sr, target_lufs=settings.target_lufs)
        else:
            y = self._final_limiter(y, sr)

        # Bước 9: Tăng / Giảm âm lượng đầu ra thủ công
        if abs(settings.master_gain_db) > 0.05:
            y = self.apply_volume_gain(y, settings.master_gain_db)

        report(100, "Hoàn tất xử lý âm thanh!")
        return y.astype(np.float32)

    # =========================================================================
    # CÁC THUẬT TOÁN KHỬ TẠP ÂM MƯỢT, CHỐNG VANG & TD-PSOLA VOICE CHANGER
    # =========================================================================
    def _pre_normalize_input(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Đưa mức âm lượng giọng nói từ Microphone về vùng hoạt động chuẩn (-18 dBFS)
        trước khi xử lý để không bao giờ bị mất chữ khi người dùng nói nhỏ.
        """
        try:
            frame_len = max(256, int(0.04 * sr))
            n_frames = len(y) // frame_len
            if n_frames >= 3:
                frames = y[: n_frames * frame_len].reshape(n_frames, frame_len)
                frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
                # Chỉ tính trên các khung có tiếng nói (trên phân vị 35%)
                thresh = np.percentile(frame_rms, 35)
                active = frame_rms[frame_rms >= thresh]
                active_rms = float(np.mean(active)) if len(active) > 0 else float(np.sqrt(np.mean(y ** 2)))
            else:
                active_rms = float(np.sqrt(np.mean(y ** 2)))

            if active_rms < 1e-6:
                return y

            target_rms = 10.0 ** (-18.0 / 20.0)  # ~0.1258
            gain = float(np.clip(target_rms / active_rms, 0.4, 20.0))
            out = y * gain
            # Bảo vệ đỉnh không vượt quá 0.95
            peak = float(np.max(np.abs(out)))
            if peak > 0.95:
                out = out * (0.95 / peak)
            return out.astype(np.float32)
        except Exception:
            return y

    def _remove_hum(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Tự dò và cắt tiếng ù điện lưới (50 Hz hoặc 60 Hz) cùng các hài của nó bằng notch hẹp pha-không.
        Chỉ phân tích trên các khung yên tĩnh nhất để giọng nói (F0 nam ~100 Hz) không bị nhận nhầm là ù.
        """
        try:
            fl = int(0.05 * sr)
            n = len(y) // fl
            if n < 8:
                return y
            frames = y[: n * fl].reshape(n, fl)
            rms = np.sqrt(np.mean(frames ** 2, axis=1))
            quiet = frames[np.argsort(rms)[: max(6, n // 4)]].reshape(-1)
            nper = int(min(len(quiet), 2 * sr))
            f, pxx = signal.welch(quiet, fs=sr, nperseg=nper, window="hann")

            def line_db(fc: float) -> float:
                peak = pxx[(f > fc - 1.0) & (f < fc + 1.0)]
                side = pxx[((f > fc - 20) & (f < fc - 6)) | ((f > fc + 6) & (f < fc + 20))]
                if len(peak) == 0 or len(side) == 0:
                    return 0.0
                return float(10.0 * np.log10(peak.max() / (np.median(side) + 1e-20) + 1e-20))

            best_hits = []
            for base in (50.0, 60.0):
                hits = [(base * k, line_db(base * k)) for k in range(1, 9) if base * k < 600.0]
                hits = [(fc, db) for fc, db in hits if db > 6.0]
                # Cần ít nhất 2 hài rõ rệt mới coi là ù điện (loại trừ trường hợp 1 đỉnh ngẫu nhiên)
                if len(hits) >= 2 and sum(db for _, db in hits) > sum(db for _, db in best_hits):
                    best_hits = hits
            if not best_hits:
                return y

            out = y.astype(np.float64)
            for fc, _ in best_hits:
                b, a = signal.iirnotch(fc, Q=30.0, fs=sr)
                out = signal.sosfiltfilt(signal.tf2sos(b, a), out)
            return out.astype(np.float32)
        except Exception as exc:
            print(f"[AudioEngine] Dò ù điện lỗi, bỏ qua: {exc}", file=sys.stderr)
            return y

    def _pick_noise_clip(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Chọn mẫu nhiễu nền từ các đoạn yên tĩnh LIÊN TỤC dài >= 300 ms (khoảng nghỉ thật giữa câu).
        Trả về None nếu bản thu gần như không có đoạn nghỉ nào (nói liên tục) -> không dám "học" nhiễu từ lời nói.
        """
        fl = max(256, int(0.05 * sr))
        n = len(y) // fl
        if n < 12:
            return None
        frames = y[: n * fl].reshape(n, fl)
        rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9
        db = 20.0 * np.log10(rms)
        floor = float(np.percentile(db, 8))
        ceil = float(np.percentile(db, 90))
        if ceil - floor < 12.0:
            return None  # không đủ chênh lệch lời/nền để tin vào đoạn nghỉ
        quiet = db < floor + 0.18 * (ceil - floor) + 3.0

        runs, start = [], None
        for i, q in enumerate(quiet):
            if q and start is None:
                start = i
            elif not q and start is not None:
                if i - start >= 6:
                    runs.append((start, i))
                start = None
        if start is not None and n - start >= 6:
            runs.append((start, n))
        if not runs:
            return None

        # Ưu tiên các đoạn yên tĩnh nhất, lấy tối đa ~4 giây, bỏ 1 khung ở mỗi đầu để tránh đuôi/đầu từ
        runs.sort(key=lambda r: float(np.mean(db[r[0]:r[1]])))
        chunks, total = [], 0
        for s, e in runs:
            seg = frames[s + 1 : e - 1].reshape(-1)
            chunks.append(seg)
            total += len(seg)
            if total >= 4 * sr:
                break
        return np.concatenate(chunks).astype(np.float32) if chunks else None

    def _reduce_noise_crisp(self, y: np.ndarray, sr: int, prop_decrease: float, stationary: bool) -> np.ndarray:
        """
        Khử tạp âm 2 lượt bằng cổng phổ (noisereduce):
        - Lượt 1 (ổn định): trừ nền quạt/điều hòa/hiss theo mẫu nhiễu lấy từ khoảng nghỉ thật.
        - Lượt 2 (biến thiên): khi người dùng bật chế độ biến thiên, hoặc không tìm được khoảng nghỉ đáng tin
          để lấy mẫu, chạy thêm bộ lọc thích nghi theo thời gian để bắt nhiễu thay đổi (xe cộ, tiếng gõ phím...).
        Giữ time_mask_smooth để cổng đóng/mở mượt, tránh nhiễu "nhạc" và cụt đuôi chữ.
        """
        try:
            strength = float(np.clip(prop_decrease, 0.0, 1.0))
            noise_clip = self._pick_noise_clip(y, sr)

            out = y
            if noise_clip is not None:
                out = nr.reduce_noise(
                    y=out,
                    sr=sr,
                    y_noise=noise_clip,
                    prop_decrease=float(np.clip(strength * 0.95, 0.0, 0.95)),
                    stationary=True,
                    n_fft=2048,
                    hop_length=512,
                    n_std_thresh_stationary=1.5,
                    freq_mask_smooth_hz=300,
                    time_mask_smooth_ms=60,
                )

            # Lượt 2: nhiễu biến thiên. Cũng là phương án thay thế khi không có mẫu nhiễu đáng tin.
            if (not stationary) or noise_clip is None:
                out = nr.reduce_noise(
                    y=out,
                    sr=sr,
                    stationary=False,
                    prop_decrease=float(np.clip(strength * (0.8 if noise_clip is None else 0.55), 0.0, 0.85)),
                    n_fft=2048,
                    hop_length=512,
                    time_constant_s=1.5,
                    freq_mask_smooth_hz=300,
                    time_mask_smooth_ms=60,
                    thresh_n_mult_nonstationary=2.0,
                    sigmoid_slope_nonstationary=8,
                )
            return np.asarray(out, dtype=np.float32)
        except Exception as exc:
            print(f"[AudioEngine] Khử tạp âm lỗi, giữ nguyên tín hiệu: {exc}", file=sys.stderr)
            return y

    def _denoise_rnnoise(self, y: np.ndarray, sr: int, strength: float) -> np.ndarray:
        """
        Khử nhiễu bằng mạng nơ-ron RNNoise (48 kHz). Trộn khô/ướt theo cường độ để người dùng vẫn chỉnh được.
        """
        try:
            if sr == 48000:
                y48 = y
            elif sr == 44100:
                y48 = signal.resample_poly(y, 160, 147)
            else:
                y48 = librosa.resample(y, orig_sr=sr, target_sr=48000)

            pcm = (np.clip(y48, -1.0, 1.0) * 32767.0).astype(np.int16)
            den = RNNoise(48000)
            frames_out = [fr for _, fr in den.denoise_chunk(pcm[None, :], partial=True)]
            if not frames_out:
                return y
            wet48 = np.concatenate([np.asarray(fr)[0] for fr in frames_out]).astype(np.float32) / 32768.0
            if len(wet48) < len(y48):
                wet48 = np.pad(wet48, (0, len(y48) - len(wet48)))
            wet48 = wet48[: len(y48)]

            if sr == 48000:
                wet = wet48
            elif sr == 44100:
                wet = signal.resample_poly(wet48, 147, 160)
            else:
                wet = librosa.resample(wet48, orig_sr=48000, target_sr=sr)
            if len(wet) < len(y):
                wet = np.pad(wet, (0, len(y) - len(wet)))
            wet = wet[: len(y)].astype(np.float32)

            # Giữ lại tối thiểu 20% tín hiệu gốc: RNNoise full-wet dễ "ăn" vào lời khi phòng có vang
            mix = float(np.clip(0.3 + 0.5 * strength, 0.0, 0.8))
            return (mix * wet + (1.0 - mix) * y).astype(np.float32)
        except Exception as exc:
            print(f"[AudioEngine] RNNoise lỗi, bỏ qua bước khử nhiễu AI: {exc}", file=sys.stderr)
            return y

    def _gap_expander(self, y: np.ndarray, sr: int, depth_db: float = 12.0) -> np.ndarray:
        """
        Hạ nền ồn còn sót lại trong các khoảng nghỉ giữa câu (downward expander có "giữ" và làm mượt):
        - Cổng chỉ đóng khi cả vùng ±60 ms xung quanh đều dưới ngưỡng -> không cắt đầu/đuôi từ.
        - Mức hạ tối đa = depth_db, chuyển mượt 40 ms -> không nghe thấy "bật/tắt".
        """
        try:
            hop = max(1, int(0.005 * sr))
            n = len(y) // hop
            if n < 20 or depth_db < 0.5:
                return y
            env = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9
            env_db = 20.0 * np.log10(env)
            env_db = uniform_filter1d(env_db, size=3, mode="nearest")
            floor = float(np.percentile(env_db, 10))
            ceil = float(np.percentile(env_db, 90))
            if ceil - floor < 15.0:
                return y  # chênh lệch lời/nền quá nhỏ -> không đủ cơ sở phân biệt khoảng nghỉ
            thr = floor + 0.30 * (ceil - floor)

            held = maximum_filter1d(env_db, size=25, mode="nearest")  # ±60 ms lookahead + hold
            open_amt = np.clip((held - (thr - 4.0)) / 8.0, 0.0, 1.0)
            gain_db = -float(depth_db) * (1.0 - open_amt)
            gain_db = uniform_filter1d(gain_db, size=9, mode="nearest")

            centers = (np.arange(n) + 0.5) * hop
            gain = 10.0 ** (np.interp(np.arange(len(y)), centers, gain_db) / 20.0)
            return (y * gain).astype(np.float32)
        except Exception as exc:
            print(f"[AudioEngine] Gap expander lỗi, bỏ qua: {exc}", file=sys.stderr)
            return y

    def _suppress_late_reverb(self, y: np.ndarray, sr: int, amount: float, rt60: float = 0.5) -> np.ndarray:
        """
        Khử đuôi vang muộn bằng mô hình suy giảm mũ (Lebart/Habets): công suất vang muộn tại thời điểm t được
        ước lượng từ công suất ~50 ms trước đó nhân hệ số suy giảm theo RT60, rồi trừ phổ có sàn tối thiểu.
        """
        n_fft = 1024 if sr <= 48000 else 2048
        hop = n_fft // 4
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        P = np.abs(S) ** 2
        Ps = uniform_filter1d(P, size=3, axis=1, mode="nearest")

        hop_s = hop / float(sr)
        d_frames = max(1, int(round(0.050 / hop_s)))
        delta = 3.0 * np.log(10.0) / rt60
        decay = float(np.exp(-2.0 * delta * d_frames * hop_s))

        late = np.zeros_like(P)
        late[:, d_frames:] = decay * Ps[:, :-d_frames]

        alpha = 1.0 + 1.5 * amount
        g_min = 10.0 ** (-(5.0 + 9.0 * amount) / 20.0)
        gain = np.sqrt(np.clip(1.0 - alpha * late / (P + 1e-12), g_min ** 2, 1.0))
        gain = uniform_filter(gain, size=(3, 3), mode="nearest")  # làm mượt phổ -> bớt "nhiễu nhạc"
        out = librosa.istft(S * gain, hop_length=hop, length=len(y))
        return out.astype(np.float32)

    def _remove_room_reverb(self, y: np.ndarray, sr: int, amount: float = 0.65) -> np.ndarray:
        """
        Khử vang phòng:
        1. Suy giảm đuôi vang muộn trong miền phổ (có sàn tối thiểu, làm mượt).
        2. Hạ nhẹ các dải cộng hưởng phòng kín (260 Hz, 380 Hz) bằng Pedalboard PeakFilter.
        """
        try:
            amt = float(np.clip(amount, 0.0, 1.0))
            out = y.copy()

            if amt > 0.05:
                out = self._suppress_late_reverb(out, sr, amt)

            if HAS_PEDALBOARD and amt > 0.05:
                de_room_board = Pedalboard([
                    PeakFilter(cutoff_frequency_hz=260.0, gain_db=float(-2.0 * amt), q=1.1),
                    PeakFilter(cutoff_frequency_hz=380.0, gain_db=float(-1.8 * amt), q=1.2),
                ])
                out = de_room_board(out.astype(np.float32), sr)
            return out.astype(np.float32)
        except Exception as exc:
            print(f"[AudioEngine] Khử vang lỗi, giữ nguyên tín hiệu: {exc}", file=sys.stderr)
            return y

    def _shift_voice_psola(
        self,
        y: np.ndarray,
        sr: int,
        pitch_semitones: float,
        formant_semitones: float,
        pitch_range_factor: float = 1.0,
        speaking_rate: float = 1.0,
    ) -> np.ndarray:
        """
        Đổi giọng nói chân thật bằng thuật toán TD-PSOLA (Praat 'Change gender'):
        - Hoạt động trực tiếp theo chu kỳ đóng mở dây thanh trên miền thời gian -> KHÔNG BỊ VANG / NHÒE PHA!
        - Thay đổi đồng thời:
          1. Formant Shift Ratio (Chiều dài thanh quản Nam / Nữ / Trẻ em / Ông lão)
          2. New Pitch Median (Tần số cơ bản F0 mới)
          3. Pitch Range Factor (Độ biểu cảm / ngữ điệu câu nói)
          4. Duration Factor (Tốc độ nói)
        """
        if HAS_PARSELMOUTH:
            try:
                snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=float(sr))
                pitch_obj = praat_call(snd, "To Pitch", 0.0, 75.0, 520.0)
                median_f0 = praat_call(pitch_obj, "Get quantile", 0.0, 0.0, 0.5, "Hertz")

                if median_f0 is None or np.isnan(median_f0) or median_f0 <= 0:
                    median_f0 = 150.0

                # Hệ số co giãn thanh quản (Formant Ratio)
                formant_ratio = float(np.clip(2.0 ** (formant_semitones / 11.0), 0.68, 1.50))

                # Cao độ dây thanh mục tiêu (Hz)
                target_median_f0 = float(np.clip(median_f0 * (2.0 ** (pitch_semitones / 12.0)), 65.0, 500.0))

                # Trong Praat Change gender, duration_factor < 1.0 là đọc nhanh hơn, > 1.0 là đọc chậm hơn
                duration_factor = float(np.clip(1.0 / max(0.5, speaking_rate), 0.65, 1.45))
                p_range = float(np.clip(pitch_range_factor, 0.10, 1.60))

                changed_snd = praat_call(
                    snd,
                    "Change gender",
                    75.0,               # Pitch floor (Hz) - Tránh bắt nhầm tạp âm trầm làm rung giọng
                    520.0,              # Pitch ceiling (Hz)
                    formant_ratio,      # Formant shift ratio (Vocal tract scaling)
                    target_median_f0,   # New pitch median (Hz)
                    p_range,            # Pitch range factor (Intonation variation)
                    duration_factor,    # Duration factor (Speaking speed)
                )
                out = changed_snd.values[0].astype(np.float32)
                return out
            except Exception:
                pass

        # Fallback nếu không có parselmouth
        out = y.copy()
        if abs(pitch_semitones) > 0.05:
            out = librosa.effects.pitch_shift(out, sr=sr, n_steps=pitch_semitones)
        if abs(speaking_rate - 1.0) > 0.02:
            out = librosa.effects.time_stretch(out, rate=speaking_rate)
        return out.astype(np.float32)

    def _apply_studio_pedalboard(self, y: np.ndarray, sr: int, settings: VoiceSettings) -> np.ndarray:
        """
        Sử dụng thư viện C++ Pedalboard của Spotify để xử lý giọng nói chuẩn Phòng thu phát thanh:
        - Loại bỏ hoàn toàn cổng cắt cứng (Hard NoiseGate) để giọng đọc liền mạch, không bao giờ bị mất âm cuối.
        - Anti-Mud Filter (310 Hz Cut): Triệt tiêu âm ồm/đục như nói trong hộp kín.
        - Articulation & Clarity Boost (2.8 kHz & 4.5 kHz): Làm nổi bật phụ âm và độ rõ lời.
        - Smooth Broadcast Compressor & Limiter: Giữ âm lượng từng chữ đều đặn, êm ái.
        """
        clarity_db = float(settings.clarity_boost * 3.5)  # 0 -> +3.5 dB extra articulation boost

        if HAS_PEDALBOARD:
            try:
                plugins = []
                # 1. (Highpass đã áp dụng ở đầu chuỗi xử lý, trước khi khử nhiễu)

                # 2. Cắt nhẹ dải tần gây ồm / đục phòng kín (310 Hz)
                mud_cut_db = -1.5 - (settings.dereverb_amount * 2.0)
                plugins.append(PeakFilter(cutoff_frequency_hz=310.0, gain_db=float(mud_cut_db), q=1.1))

                # 3. Độ ấm trầm gọn (135 Hz)
                if abs(settings.bass_gain_db) > 0.1:
                    plugins.append(PeakFilter(cutoff_frequency_hz=135.0, gain_db=float(settings.bass_gain_db), q=0.85))

                # 4. Độ rõ lời nói chính (Mid Presence 2800 Hz + Articulation 4500 Hz)
                total_mid_db = float(settings.mid_gain_db * 0.8 + clarity_db * 0.55)
                if abs(total_mid_db) > 0.1:
                    plugins.append(PeakFilter(cutoff_frequency_hz=2800.0, gain_db=total_mid_db, q=0.85))

                if clarity_db > 0.2:
                    plugins.append(PeakFilter(cutoff_frequency_hz=4500.0, gain_db=clarity_db * 0.65, q=0.95))

                # 5. Độ sáng / Chi tiết (High Shelf 7800 Hz)
                if abs(settings.treble_gain_db) > 0.1:
                    plugins.append(
                        HighShelfFilter(cutoff_frequency_hz=7800.0, gain_db=float(settings.treble_gain_db * 0.85))
                    )

                # 6. Nén đều giọng êm mượt (Smooth Broadcast Compressor - Release 160ms chống nhấp nhô âm lượng)
                if settings.compressor_amount > 0.02:
                    comp_thresh = -16.0 - (settings.compressor_amount * 6.0)
                    comp_ratio = 1.6 + (settings.compressor_amount * 1.6)
                    plugins.append(
                        Compressor(
                            threshold_db=float(comp_thresh),
                            ratio=float(comp_ratio),
                            attack_ms=15.0,
                            release_ms=160.0,
                        )
                    )

                # 7. Chỉ thêm Reverb khi người dùng chủ động kéo thanh Reverb > 0%
                if settings.reverb_amount > 0.01:
                    plugins.append(
                        Reverb(
                            room_size=0.25 if settings.special_fx != "hall" else 0.70,
                            damping=0.75,
                            wet_level=float(settings.reverb_amount * 0.55),
                            dry_level=1.0,
                            width=0.5,
                        )
                    )

                # Limiter đặt ở cuối chuỗi (sau chuẩn hóa độ lớn), không đặt ở đây
                board = Pedalboard(plugins)
                out = board(y.astype(np.float32), sr)

                # De-esser giảm chói âm 's', 'x' (đặt sau EQ vì EQ có thể đẩy dải 4-8 kHz lên)
                if settings.deesser_amount > 0.05 and sr > 18000:
                    out = self._apply_deesser(out, sr, settings.deesser_amount)
                return out.astype(np.float32)
            except Exception as exc:
                print(f"[AudioEngine] Pedalboard lỗi, bỏ qua EQ/nén: {exc}", file=sys.stderr)

        return y.astype(np.float32)

    def _apply_deesser(self, y: np.ndarray, sr: int, deesser_amount: float) -> np.ndarray:
        # sosfiltfilt (pha-không): dải xuýt trừ ra khỏi tín hiệu gốc nên không được lệch pha so với nó
        sos_sib = signal.butter(2, [5800, min(9800, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
        sibilance = signal.sosfiltfilt(sos_sib, y)
        env = np.abs(sibilance)
        win = max(1, int(0.010 * sr))
        env_smooth = uniform_filter1d(env, size=win, mode="nearest")
        thresh = np.percentile(env_smooth, 90) + 1e-6
        reduction = np.where(
            env_smooth > thresh,
            1.0 - deesser_amount * 0.45 * np.tanh((env_smooth - thresh) / thresh),
            1.0,
        )
        reduction = uniform_filter1d(reduction, size=max(1, int(0.008 * sr)), mode="nearest")  # tránh méo điều biên
        return (y - sibilance * (1.0 - reduction)).astype(np.float32)

    def _isolate_vocals(self, y: np.ndarray, sr: int) -> np.ndarray:
        try:
            S_full, phase = librosa.magphase(librosa.stft(y, n_fft=2048, hop_length=512))
            S_filter = librosa.decompose.nn_filter(
                S_full,
                aggregate=np.median,
                metric="cosine",
                width=int(librosa.time_to_frames(1.5, sr=sr, hop_length=512)),
            )
            S_filter = np.minimum(S_full, S_filter)
            margin_v = 3.5
            power = 2
            mask_v = librosa.util.softmask(
                S_full - S_filter, margin_v * S_filter, power=power
            )
            S_foreground = mask_v * S_full
            y_fg = librosa.istft(S_foreground * phase, hop_length=512, length=len(y))
            sos = signal.butter(3, [85, min(11500, sr * 0.45)], btype="bandpass", fs=sr, output="sos")
            return signal.sosfilt(sos, y_fg).astype(np.float32)
        except Exception:
            return y

    def _trim_silence(self, y: np.ndarray, sr: int, top_db: float = 38.0, pad_ms: int = 250) -> np.ndarray:
        intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
        if len(intervals) == 0:
            return y

        pad_samples = int((pad_ms / 1000.0) * sr)
        fade_samples = min(int(0.015 * sr), pad_samples // 2 + 1)

        segments = []
        for start, end in intervals:
            s = max(0, start - pad_samples // 2)
            e = min(len(y), end + pad_samples // 2)
            seg = y[s:e].copy()
            if len(seg) > fade_samples * 2 and fade_samples > 1:
                fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                seg[:fade_samples] *= fade_in
                seg[-fade_samples:] *= fade_out
            segments.append(seg)

        return np.concatenate(segments, axis=0).astype(np.float32)

    def _apply_special_fx(self, y: np.ndarray, sr: int, fx_type: str = "none") -> np.ndarray:
        if fx_type == "radio":
            sos = signal.butter(4, [420, min(3400, sr * 0.42)], btype="bandpass", fs=sr, output="sos")
            filtered = signal.sosfilt(sos, y).astype(np.float32)
            return np.tanh(filtered * 2.5).astype(np.float32)
        elif fx_type == "robot":
            t = np.arange(len(y), dtype=np.float32) / float(sr)
            carrier = np.sin(2.0 * np.pi * 65.0 * t)
            ring = 0.60 * y + 0.40 * (y * carrier)
            return ring.astype(np.float32)
        elif fx_type == "megaphone":
            sos = signal.butter(4, [600, min(4200, sr * 0.42)], btype="bandpass", fs=sr, output="sos")
            filtered = signal.sosfilt(sos, y).astype(np.float32)
            return np.clip(np.tanh(filtered * 3.2), -0.95, 0.95).astype(np.float32)
        return y

    def _final_limiter(
        self,
        y: np.ndarray,
        sr: int,
        peak_limit_db: float = -1.0,
        release_ms: float = 120.0,
    ) -> np.ndarray:
        """
        Limiter đỉnh cuối chuỗi (lookahead ~5 ms + release mượt), chỉ GIẢM gain khi vượt trần, không bao giờ tăng.
        (Không dùng Pedalboard Limiter vì bản đó tự khuếch đại tín hiệu nhỏ và ép đỉnh về 0 dBFS.)
        """
        ceiling = float(10.0 ** (peak_limit_db / 20.0))
        x = y.astype(np.float32)
        need = np.minimum(1.0, ceiling / (np.abs(x) + 1e-12))
        if need.min() >= 1.0:
            return x

        blk = max(1, int(sr // 1000))                       # khối 1 ms
        nb = -(-len(x) // blk)
        padded = np.ones(nb * blk, dtype=np.float32)
        padded[: len(x)] = need
        g = padded.reshape(nb, blk).min(axis=1)
        g = minimum_filter1d(g, size=11, mode="nearest")     # nhìn trước/giữ ±5 ms

        rel = float(np.exp(-1.0 / max(1.0, release_ms)))     # release theo từng khối 1 ms
        out = np.empty_like(g)
        prev = 1.0
        for i in range(nb):
            gi = g[i]
            prev = gi if gi < prev else min(gi, prev + (1.0 - prev) * (1.0 - rel))
            out[i] = prev
        out = uniform_filter1d(out, size=5, mode="nearest")  # làm mượt bậc gain, vẫn <= g tại các đỉnh

        centers = (np.arange(nb) + 0.5) * blk
        gain = np.interp(np.arange(len(x)), centers, out).astype(np.float32)
        return np.clip(x * gain, -ceiling, ceiling).astype(np.float32)

    def _normalize_loudness(
        self,
        y: np.ndarray,
        sr: int,
        target_lufs: float = -14.0,
        peak_limit_db: float = -1.0,
    ) -> np.ndarray:
        """
        Chuẩn hóa độ lớn theo LUFS thật (ITU-R BS.1770 qua pyloudnorm) rồi limiter đỉnh ở cuối.
        Không có pyloudnorm thì dùng RMS của các đoạn có tiếng nói làm phương án thay thế.
        """
        gain_db = None
        if HAS_PYLN and len(y) >= int(0.5 * sr):
            try:
                loud = float(pyln.Meter(sr).integrated_loudness(y.astype(np.float64)))
                if np.isfinite(loud):
                    gain_db = float(np.clip(target_lufs - loud, -12.0, 24.0))
            except Exception as exc:
                print(f"[AudioEngine] Đo LUFS lỗi, dùng RMS: {exc}", file=sys.stderr)

        if gain_db is None:
            frame_len = 2048
            if len(y) >= frame_len * 2:
                frames = y[: (len(y) // frame_len) * frame_len].reshape(-1, frame_len)
                f_rms = np.sqrt(np.mean(frames ** 2, axis=1))
                active = f_rms[f_rms >= np.percentile(f_rms, 35)]
                rms = float(np.mean(active)) + 1e-8 if len(active) > 0 else float(np.sqrt(np.mean(y ** 2))) + 1e-8
            else:
                rms = float(np.sqrt(np.mean(y ** 2))) + 1e-8
            gain_db = float(np.clip((target_lufs - 2.0) - 20.0 * np.log10(rms), -12.0, 20.0))

        y_norm = (y * (10.0 ** (gain_db / 20.0))).astype(np.float32)
        limited = self._final_limiter(y_norm, sr, peak_limit_db)

        # Limiter làm độ lớn tụt nhẹ so với đích -> đo lại một lần và bù (tối đa +3 dB) rồi giới hạn lại
        if HAS_PYLN and len(limited) >= int(0.5 * sr):
            try:
                loud2 = float(pyln.Meter(sr).integrated_loudness(limited.astype(np.float64)))
                shortfall = target_lufs - loud2
                if np.isfinite(loud2) and shortfall > 0.3:
                    boost = float(10.0 ** (min(shortfall, 3.0) / 20.0))
                    limited = self._final_limiter((y_norm * boost).astype(np.float32), sr, peak_limit_db)
            except Exception:
                pass
        return limited

    def apply_volume_gain(self, y: np.ndarray, gain_db: float) -> np.ndarray:
        if y is None or len(y) == 0 or abs(gain_db) < 0.01:
            return y
        factor = float(10.0 ** (gain_db / 20.0))
        scaled = y.astype(np.float32) * factor
        threshold = 0.92
        abs_s = np.abs(scaled)
        mask = abs_s > threshold
        if np.any(mask):
            sign = np.sign(scaled[mask])
            over = abs_s[mask] - threshold
            scaled[mask] = sign * (threshold + (1.0 - threshold) * np.tanh(over / (1.0 - threshold + 1e-6)))
        return np.clip(scaled, -0.995, 0.995).astype(np.float32)

    # =========================================================================
    # 4. AI VOICE CONVERSION (SPEECH-TO-SPEECH MIỄN PHÍ & ELEVENLABS) + EDGE-TTS
    # =========================================================================
    def recognize_speech_from_audio(self, audio: np.ndarray, sr: int, language: str = "vi-VN") -> str:
        """
        Nhận diện lời nói Tiếng Việt từ bản thu âm (Speech-to-Text) để phục vụ tính năng
        Đổi Giọng AI Speech-to-Speech (Tái tạo lại giọng đọc mới sạch & không vang 100%).
        Không dùng Spectral Gate trước khi nhận diện để tránh làm đứt âm tiết khi gửi lên AI.
        """
        if not HAS_SR:
            raise RuntimeError("Chưa cài đặt thư viện SpeechRecognition.")

        clean_y = audio.astype(np.float32) - np.mean(audio)
        sos = signal.butter(2, 80.0, btype="highpass", fs=sr, output="sos")
        clean_y = signal.sosfilt(sos, clean_y).astype(np.float32)
        clean_y = self._pre_normalize_input(clean_y, sr)

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            sf.write(tmp_wav, np.clip(clean_y, -1.0, 1.0), sr, subtype="PCM_16")
            recognizer = sr_lib.Recognizer()
            with sr_lib.AudioFile(tmp_wav) as source:
                audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language)
            return text.strip()
        finally:
            if os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass

    def convert_voice_ai_neural_sts(
        self,
        audio: np.ndarray,
        sr: int,
        voice_id: str = "vi-VN-NamMinhNeural|0|0",
        rate_pct: int = 0,
        pitch_hz: int = 0,
    ) -> Tuple[np.ndarray, int, str]:
        """
        Đổi giọng AI Speech-to-Speech Miễn Phí:
        Tự động tách lời thoại từ bản thu âm gốc -> Đọc lại bằng giọng AI Neural chuẩn phòng thu
        (Giúp khử 100% tiếng vang phòng, rõ từng chữ tuyệt đối và đổi sang giọng Nam/Nữ mới hoàn toàn!).
        """
        recognized_text = self.recognize_speech_from_audio(audio, sr, language="vi-VN")
        if not recognized_text:
            raise ValueError("Không nhận diện được lời nói rõ ràng trong bản thu âm. Hãy nói to và rõ hơn!")

        out_audio, out_sr = self.generate_edge_tts(
            text=recognized_text,
            voice_id=voice_id,
            rate_pct=rate_pct,
            pitch_hz=pitch_hz,
        )
        return out_audio, out_sr, recognized_text

    def generate_edge_tts(
        self,
        text: str,
        voice_id: str = "vi-VN-NamMinhNeural",
        rate_pct: int = 0,
        pitch_hz: int = 0,
    ) -> Tuple[np.ndarray, int]:
        if not HAS_EDGE_TTS:
            raise RuntimeError("Chưa cài đặt thư viện edge-tts.")
        if not text.strip():
            raise ValueError("Vui lòng nhập nội dung kịch bản cần đọc.")

        actual_voice = voice_id.split("|")[0] if "|" in voice_id else voice_id
        rate_str = f"{rate_pct:+d}%"
        pitch_str = f"{pitch_hz:+d}Hz"
        tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name

        async def _run():
            communicate = edge_tts.Communicate(text=text, voice=actual_voice, rate=rate_str, pitch=pitch_str)
            await communicate.save(tmp_mp3)

        try:
            asyncio.run(_run())
            audio, sr, _ = self.load_media_file(tmp_mp3)
            return audio, sr
        finally:
            if os.path.exists(tmp_mp3):
                try:
                    os.remove(tmp_mp3)
                except Exception:
                    pass

    def convert_voice_elevenlabs_sts(
        self,
        audio: np.ndarray,
        sr: int,
        api_key: str,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
        model_id: str = "eleven_multilingual_sts_v2",
    ) -> Tuple[np.ndarray, int]:
        if not api_key.strip():
            raise ValueError("Vui lòng nhập ElevenLabs API Key.")

        tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        tmp_out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        try:
            sf.write(tmp_in, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")
            url = f"https://api.elevenlabs.io/v1/speech-to-speech/{voice_id.strip()}"
            headers = {"xi-api-key": api_key.strip()}
            data = {
                "model_id": model_id,
                "voice_settings": '{"stability": 0.55, "similarity_boost": 0.8, "style": 0.2, "use_speaker_boost": true}',
            }
            with open(tmp_in, "rb") as f:
                files = {"audio": ("input.wav", f, "audio/wav")}
                resp = requests.post(url, headers=headers, data=data, files=files, timeout=90)

            if resp.status_code != 200:
                raise RuntimeError(f"Lỗi ElevenLabs API ({resp.status_code}): {resp.text[:250]}")

            with open(tmp_out, "wb") as f:
                f.write(resp.content)

            out_audio, out_sr, _ = self.load_media_file(tmp_out)
            return out_audio, out_sr
        finally:
            for p in (tmp_in, tmp_out):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    def generate_demo_voice(self, duration_sec: float = 4.0) -> Tuple[np.ndarray, int]:
        """
        Tạo giọng mẫu Tiếng Việt thực tế (hoặc âm thanh harmonic giả lập nếu ngoại tuyến)
        để người dùng thử ngay 25 mẫu đổi giọng TD-PSOLA và bộ lọc khử vang mà không cần cắm mic.
        """
        if HAS_EDGE_TTS:
            try:
                sample_text = (
                    "Xin chào các bạn, đây là bản thu âm thử nghiệm bộ xử lý giọng nói chống vang "
                    "và biến đổi thanh quản thế hệ mới."
                )
                audio, sr = self.generate_edge_tts(sample_text, voice_id="vi-VN-NamMinhNeural", rate_pct=0, pitch_hz=0)
                # Thêm một chút tạp âm nền nhỏ để người dùng thấy rõ hiệu quả Lọc Tạp Âm & Đổi Giọng
                rng = np.random.default_rng(42)
                noise = rng.normal(0.0, 0.008, len(audio)).astype(np.float32)
                return (audio + noise).astype(np.float32), sr
            except Exception:
                pass

        sr = self.target_sr
        n = int(duration_sec * sr)
        t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
        f0 = 135.0 + 20.0 * np.sin(2.0 * np.pi * 1.4 * t)
        phase = 2.0 * np.pi * np.cumsum(f0) / sr
        voice = np.zeros_like(t)
        for k in range(1, 16):
            freq_k = k * 135.0
            f1 = np.exp(-0.5 * ((freq_k - 520.0) / 140.0) ** 2)
            f2 = 0.7 * np.exp(-0.5 * ((freq_k - 1450.0) / 240.0) ** 2)
            amp = (0.25 + f1 + f2) / (k ** 0.85)
            voice += amp * np.sin(k * phase)
        syllable_env = np.clip(np.sin(2.0 * np.pi * 2.8 * t) * 1.8 + 0.4, 0.0, 1.0)
        voice = voice * syllable_env
        voice = voice / (np.max(np.abs(voice)) + 1e-6) * 0.65
        return voice.astype(np.float32), sr

