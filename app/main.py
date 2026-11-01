"""声替 CoverMe Web 服务入口：FastAPI + 静态前端。"""
from __future__ import annotations

import logging
import re
import secrets
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .audio.io import decode_to_wav, find_ffmpeg, load_wav, wav_duration
from .separation import iter_separators, separator_status
from .store import Store
from .tasks import Pipeline
from .voice import engine_status

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("studio")

store = Store(config.DB_PATH)
pipeline = Pipeline(store)

_ID_RE = re.compile(r"^[a-f0-9]{6,32}$")


def _new_id() -> str:
    return secrets.token_hex(8)


def _safe_upload_name(filename: str | None) -> str:
    name = Path(filename or "").name
    return re.sub(r'[\\/:*?"<>|\r\n]+', "_", name)[:120]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    pipeline.start()
    # 后台预热引擎可用性（首次会 import torch/openvoice 等重库）
    threading.Thread(target=lambda: [engine_status(), separator_status()], daemon=True).start()
    yield
    pipeline.stop()


app = FastAPI(title=config.APP_NAME, version=config.VERSION, lifespan=lifespan)


# ---------------------------------------------------------------- 系统信息

@app.get("/api/health")
def health():
    return {"ok": True, "app": config.APP_NAME, "version": config.VERSION,
            "ffmpeg": bool(find_ffmpeg()), "data_dir": str(config.DATA_DIR)}


@app.get("/api/engines")
def engines():
    return {"engines": engine_status(), "separators": separator_status()}


# ---------------------------------------------------------------- 声音样本

def _save_upload(file: UploadFile, dest: Path, cap_bytes: int) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with dest.open("wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > cap_bytes:
                    raise HTTPException(413, f"文件超过大小上限 {cap_bytes // 1024 // 1024}MB")
                f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    return total


@app.get("/api/voice-samples")
def list_samples():
    return store.list_samples()


@app.post("/api/voice-samples")
async def upload_sample(file: UploadFile = File(...)):
    ext = Path(file.filename or "ref.wav").suffix.lower() or ".wav"
    if ext not in config.SUPPORTED_AUDIO_EXT:
        raise HTTPException(400, f"不支持的音频格式: {ext}")
    sid = _new_id()
    raw = config.UPLOAD_DIR / "voice" / f"{sid}{ext}"
    _save_upload(file, raw, config.MAX_UPLOAD_MB * 1024 * 1024)
    try:
        ref = config.SAMPLES_DIR / sid / "ref.wav"
        decode_to_wav(raw, ref, config.TARGET_SR)
        dur = wav_duration(ref)
    except Exception as e:
        raw.unlink(missing_ok=True)
        shutil.rmtree(config.SAMPLES_DIR / sid, ignore_errors=True)
        raise HTTPException(400, f"音频解码失败: {e}")
    if dur < 5:
        raw.unlink(missing_ok=True)
        shutil.rmtree(config.SAMPLES_DIR / sid, ignore_errors=True)
        raise HTTPException(400, f"录音太短（{dur:.1f} 秒），请录制 10~30 秒清晰的人声")
    # 内容有效性：静音/几乎无声的样本会提取出随机音色向量，必须拒绝
    import numpy as np
    data, _ = load_wav(ref)
    peak = float(np.abs(data).max()) if data.size else 0.0
    if peak < 0.01:
        raw.unlink(missing_ok=True)
        shutil.rmtree(config.SAMPLES_DIR / sid, ignore_errors=True)
        raise HTTPException(400, "录音内容是静音——请检查麦克风权限/设备后重录，"
                                 "或改用「上传音频文件」（10~30 秒清晰人声）")
    name = Path(_safe_upload_name(file.filename)).stem or "我的声音"
    store.add_sample(sid, name, ref, dur)
    log.info("声音样本上传: %s (%s, %.1fs)", sid, name, dur)
    return {"id": sid, "name": name, "duration": round(dur, 1)}


@app.delete("/api/voice-samples/{sid}")
def delete_sample(sid: str):
    if not _ID_RE.match(sid):
        raise HTTPException(400, "id 不合法")
    sample = store.get_sample(sid)
    if not sample:
        raise HTTPException(404, "样本不存在")
    shutil.rmtree(config.SAMPLES_DIR / sid, ignore_errors=True)
    store.delete_sample(sid)
    return {"ok": True}


@app.get("/api/voice-samples/{sid}/preview")
def sample_preview(sid: str):
    """试听声音样本：返回解码后的 ref.wav。"""
    if not _ID_RE.match(sid):
        raise HTTPException(400, "id 不合法")
    p = config.SAMPLES_DIR / sid / "ref.wav"
    if not p.exists():
        raise HTTPException(404, "样本不存在")
    return FileResponse(p, media_type="audio/wav")


# ---------------------------------------------------------------- 翻唱任务

@app.post("/api/analyze-song")
async def analyze_song_endpoint(file: UploadFile = File(...)):
    """智能匹配：分析歌曲（分离+音域/亮度），自动推荐音色与设置。约 1~2 分钟。"""
    ext = Path(file.filename or "song.mp3").suffix.lower() or ".mp3"
    if ext not in config.SUPPORTED_AUDIO_EXT:
        raise HTTPException(400, f"不支持的音频格式: {ext}")
    tmp = config.DATA_DIR / "tmp" / f"analyze_upload_{secrets.token_hex(4)}{ext}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    _save_upload(file, tmp, config.MAX_UPLOAD_MB * 1024 * 1024)
    try:
        from .analysis import analyze_song
        result = analyze_song(tmp, store.list_samples())
        log.info("智能匹配: %s → %s", Path(file.filename or "").name, result.get("voice_name"))
        return result
    except Exception as e:
        raise HTTPException(500, f"分析失败: {e}")
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/api/covers")
async def create_cover(
    song: UploadFile = File(...),
    voice_sample_id: str = Form(...),
    engine: str = Form("auto"),
    separator: str = Form("auto"),
    vocal_gain_db: float = Form(0.0),
    instrumental_gain_db: float = Form(0.0),
    pitch_shift_semitones: float = Form(0.0),
    openvoice_tau: float = Form(0.3),
    auto_balance: bool = Form(True),
    denoise_gate: bool = Form(True),
    title: str = Form(""),
):
    if not _ID_RE.match(voice_sample_id):
        raise HTTPException(400, "voice_sample_id 不合法")
    sample = store.get_sample(voice_sample_id)
    if not sample:
        raise HTTPException(404, "声音样本不存在，请先录制或上传")
    ext = Path(song.filename or "song.wav").suffix.lower() or ".wav"
    if ext not in config.SUPPORTED_AUDIO_EXT:
        raise HTTPException(400, f"不支持的音频格式: {ext}")
    jid = _new_id()
    song_path = config.UPLOAD_DIR / "song" / f"{jid}{ext}"
    _save_upload(song, song_path, config.MAX_UPLOAD_MB * 1024 * 1024)
    title = (title or "").strip() or Path(_safe_upload_name(song.filename)).stem or "未命名翻唱"
    store.create_job(jid, title=title[:80], song_name=_safe_upload_name(song.filename),
                     song_path=song_path, engine=engine, separator=separator,
                     voice_sample_id=voice_sample_id,
                     options={"vocal_gain_db": vocal_gain_db,
                              "instrumental_gain_db": instrumental_gain_db,
                              "pitch_shift_semitones": pitch_shift_semitones,
                              "tau": max(0.05, min(0.9, float(openvoice_tau))),
                              "auto_balance": auto_balance,
                              "denoise_gate": denoise_gate})
    pipeline.enqueue(jid)
    log.info("创建翻唱任务: %s (%s, engine=%s, separator=%s)", jid, title, engine, separator)
    return {"id": jid, "status": "queued", "title": title[:80]}


def _job_or_404(jid: str) -> dict:
    if not _ID_RE.match(jid):
        raise HTTPException(400, "id 不合法")
    job = store.get_job(jid)
    if not job:
        raise HTTPException(404, "任务不存在")
    return job


@app.get("/api/covers")
def list_covers():
    return store.list_jobs()


@app.get("/api/covers/{jid}")
def get_cover(jid: str):
    job = _job_or_404(jid)
    jd = config.JOBS_DIR / jid
    job["has_result"] = (jd / "cover.mp3").exists() or (jd / "cover.wav").exists()
    # 存活反馈：worker 线程是否在执行 + 最近一次进度心跳的时间
    job["worker_alive"] = pipeline.is_running(jid) if job["status"] == "processing" else None
    return job


def _download_name(jid: str, ext: str) -> str:
    job = store.get_job(jid) or {}
    t = re.sub(r'[\\/:*?"<>|\s]+', "_", job.get("title") or "cover")[:60] or "cover"
    return f"{t}-声替翻唱.{ext}"


@app.get("/api/covers/{jid}/result")
def cover_result(jid: str):
    _job_or_404(jid)
    jd = config.JOBS_DIR / jid
    for name, media in (("cover.mp3", "audio/mpeg"), ("cover.wav", "audio/wav")):
        p = jd / name
        if p.exists():
            return FileResponse(p, media_type=media, filename=_download_name(jid, Path(name).suffix.lstrip(".")))
    raise HTTPException(404, "结果尚未生成")


@app.get("/api/covers/{jid}/stems/{kind}")
def cover_stem(jid: str, kind: str):
    _job_or_404(jid)
    mapping = {
        "song": "song.wav",
        "vocals": "stems/vocals.wav",
        "instrumental": "stems/no_vocals.wav",
        "converted": "converted_vocals.wav",
    }
    if kind not in mapping:
        raise HTTPException(404, "未知音轨，可选: " + "/".join(mapping))
    p = config.JOBS_DIR / jid / mapping[kind]
    if not p.exists():
        raise HTTPException(404, "音轨不存在")
    return FileResponse(p, media_type="audio/wav")


@app.post("/api/covers/{jid}/cancel")
def cancel_cover(jid: str):
    """取消排队中/进行中的任务。进行中的任务会在下一个阶段边界停止。"""
    job = _job_or_404(jid)
    if job["status"] not in ("queued", "processing"):
        raise HTTPException(409, "任务已结束，无需取消")
    action = pipeline.cancel(jid)
    return {"ok": True, "status": "cancelled" if action == "cancelled" else "cancelling"}


@app.delete("/api/covers/{jid}")
def delete_cover(jid: str):
    job = _job_or_404(jid)
    if job["status"] == "processing":
        raise HTTPException(409, "任务进行中，请先取消再删除")
    if job["status"] == "queued":
        # 排队中的任务可能永远不会被派发（如服务重启后），允许直接删除；
        # 同时设置取消事件，防止竞态下派发后继续执行
        pipeline.cancel(jid)
    shutil.rmtree(config.JOBS_DIR / jid, ignore_errors=True)
    Path(job.get("song_path") or "").unlink(missing_ok=True)
    store.delete_job(jid)
    return {"ok": True}


# ---------------------------------------------------------------- 静态前端

# 本地开发工具，静态资源一律不缓存，保证前端代码改动刷新即生效
# （不依赖 StaticFiles 版本的 cache_control 参数，直接包装 asgi 响应头）
_static_app = StaticFiles(directory=config.WEB_DIR)


async def _assets_no_cache(scope, receive, send):
    if scope["type"] != "http":
        await _static_app(scope, receive, send)
        return

    async def _send(message, **kwargs):
        if message["type"] == "http.response.start":
            # 直接追加到 header 列表（不转 dict），保留可能存在的重复 header
            headers = list(message.get("headers") or [])
            headers.append((b"cache-control", b"no-cache"))
            message = {**message, "headers": headers}
        await send(message, **kwargs)

    await _static_app(scope, receive, _send)


app.mount("/assets", _assets_no_cache, name="assets")


@app.get("/")
def index():
    return FileResponse(config.WEB_DIR / "index.html",
                        headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
