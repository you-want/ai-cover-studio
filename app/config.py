"""全局配置：所有路径与开关均可通过环境变量覆盖（前缀 VC_）。"""
import os
from pathlib import Path

APP_NAME = "声替 CoverMe"
VERSION = "1.1.0"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("VC_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLES_DIR = DATA_DIR / "samples"
JOBS_DIR = DATA_DIR / "jobs"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "studio.db"
MODELS_DIR = Path(os.getenv("VC_MODELS_DIR", BASE_DIR / "models"))
WEB_DIR = BASE_DIR / "app" / "web"

HOST = os.getenv("VC_HOST", "127.0.0.1")
PORT = int(os.getenv("VC_PORT", "8765"))

TARGET_SR = 44100
MAX_UPLOAD_MB = int(os.getenv("VC_MAX_UPLOAD_MB", "150"))
MAX_SONG_MINUTES = float(os.getenv("VC_MAX_SONG_MINUTES", "12"))
MAX_CONCURRENT_JOBS = int(os.getenv("VC_MAX_CONCURRENT_JOBS", "1"))

DEFAULT_ENGINE = os.getenv("VC_DEFAULT_ENGINE", "auto")       # auto = 按优先级选可用引擎
DEFAULT_SEPARATOR = os.getenv("VC_DEFAULT_SEPARATOR", "auto")  # auto = demucs 优先，中心声道兜底
# 计算设备覆盖（auto = cuda→mps→cpu 自动探测；可显式设 cpu/mps/cuda，
# 调试或某引擎在 MPS 上不稳时回退用，如 VC_DEVICE=cpu）
DEVICE = os.getenv("VC_DEVICE", "auto").lower()
DEMUCS_MODEL = os.getenv("VC_DEMUCS_MODEL", "htdemucs")

# 分离子进程超时（秒）：CPU 上一首 12 分钟的歌约需 20 分钟，超时按失败处理走兜底
DEMUCS_TIMEOUT = int(os.getenv("VC_DEMUCS_TIMEOUT", "3600"))

# 分离结果缓存：同一首歌 + 同一模型只分离一次，调参重跑时秒级出结果
SEPARATION_CACHE = os.getenv("VC_SEPARATION_CACHE", "1") != "0"
CACHE_MAX_GB = float(os.getenv("VC_CACHE_MAX_GB", "5"))

# 人声去混响（UVR-DeEcho-DeReverb）：AI 翻唱流水线公认的关键步骤，
# Demucs 分出的人声带房间混响，转换前先"烘干"能显著减少转换毛刺
DEREVERB_ENABLED = os.getenv("VC_DEREVEB", "1") != "0"
AUDIO_SEPARATOR_DIR = MODELS_DIR / "audio-separator"

# 频谱包络匹配：把转换人声分频段能量对齐到原唱人声（治发闷的根治方案）
SPECTRAL_MATCH = os.getenv("VC_SPECTRAL_MATCH", "1") != "0"

# Seed-VC 扩散步数（CPU 建议 30；越大质量略好但线性变慢）
SEEDVC_DIFFUSION_STEPS = int(os.getenv("VC_SEEDVC_STEPS", "30"))
# Seed-VC 歌唱模式（SVC，f0-condition）：权重就绪时优先使用，关闭则强制 VC 模式
SEEDVC_SVC = os.getenv("VC_SEEDVC_SVC", "1") != "0"
# SVC 硬上限（小时）：整曲多分块在算力被挤占时可能超过 4h（实测教训）；
# 真正的卡死由"30 分钟无输出"看门狗负责，这里只防天文数字级别的失控
SEEDVC_CAP_H = float(os.getenv("VC_SEEDVC_CAP_H", "8"))

# 云端付费 API 引擎（可选；auto 模式永不自动使用，仅用户明确选择时计费）
CLOUD_PROVIDER = os.getenv("VC_CLOUD_PROVIDER", "")            # replicate / elevenlabs
CLOUD_API_KEY = os.getenv("VC_CLOUD_API_KEY", "")
CLOUD_VOICE_ID = os.getenv("VC_CLOUD_VOICE_ID", "")            # elevenlabs Voice ID 或 replicate 模型 slug
CLOUD_EXTRA_JSON = os.getenv("VC_CLOUD_EXTRA_JSON", "")        # replicate 额外 input 字段（JSON）
CLOUD_INPUT_KEY = os.getenv("VC_CLOUD_INPUT_KEY", "audio")     # replicate 模型的音频输入字段名

# 国内网络下 huggingface.co 常不可达，模型下载默认走 hf-mirror 镜像
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
# 镜像不支持 Xet 协议，禁用后回退普通 HTTP 下载
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
OPENVOICE_DIR = Path(os.getenv("VC_OPENVOICE_DIR", MODELS_DIR / "openvoice"))
SEEDVC_DIR = Path(os.getenv("VC_SEEDVC_DIR", MODELS_DIR / "seedvc"))

# demucs/torch 模型缓存统一放在项目内，便于管理与清理
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
# Intel Mac 上 torch 仍可能报告 MPS 可用，但部分算子（如 FFT）未实现，允许自动回退 CPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SUPPORTED_AUDIO_EXT = {
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".flac",
    ".webm", ".aiff", ".aif", ".opus", ".wma",
}
SUPPORTED_SF_EXT = {".wav", ".aiff", ".aif", ".flac", ".ogg", ".oga"}

for _d in (DATA_DIR, UPLOAD_DIR, SAMPLES_DIR, JOBS_DIR, CACHE_DIR, MODELS_DIR, AUDIO_SEPARATOR_DIR):
    _d.mkdir(parents=True, exist_ok=True)
