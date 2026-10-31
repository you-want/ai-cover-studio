"""云端付费 API 声音转换引擎（可选，按量计费）。

设计原则：
  - auto 模式永不选择本引擎——只有用户在界面上明确选中才产生付费调用；
  - API Key 只从环境变量读取，绝不入库、绝不上传；
  - 转换结果进入与本地引擎完全相同的混音链（人声总线/电平对齐/母带）。

支持的 provider（通过 VC_CLOUD_PROVIDER 切换）：
  replicate   Replicate 通用音频模型（自托管开源 SVC 的云端版，按秒计费）
  elevenlabs  ElevenLabs Voice Changer（语音级变声，音质高，订阅制）
"""
from __future__ import annotations

import base64
from pathlib import Path

import requests

from .. import config
from ..audio.dsp import peak_normalize, pitch_shift, resample, to_stereo
from ..audio.io import load_wav, save_wav
from .base import BaseVoiceEngine, ProgressCB, VoiceProfile

_TIMEOUT_S = 900


def _replicate_convert(wav_path: Path, profile: VoiceProfile,
                       progress: ProgressCB) -> Path:
    """Replicate 通用适配：上传音频（data URI）→ 建预测 → 轮询 → 下载输出。"""
    key = config.CLOUD_API_KEY
    slug = config.CLOUD_VOICE_ID
    if not slug:
        raise RuntimeError("缺少 VC_CLOUD_VOICE_ID（Replicate 模型 slug，如 owner/name[:version]）")
    b64 = base64.b64encode(wav_path.read_bytes()).decode()
    payload = {"input": {config.CLOUD_INPUT_KEY or "audio": f"data:audio/wav;base64,{b64}"}}
    if config.CLOUD_EXTRA_JSON:
        import json
        try:
            payload["input"].update(json.loads(config.CLOUD_EXTRA_JSON))
        except Exception as e:
            raise RuntimeError(f"VC_CLOUD_EXTRA_JSON 不是合法 JSON: {e}")
    headers = {"Authorization": f"Token {key}", "Content-Type": "application/json"}
    if ":" in slug:  # 指定版本
        name, version = slug.split(":", 1)
        r = requests.post(f"https://api.replicate.com/v1/predictions",
                          json={"version": version, "input": payload["input"]},
                          headers=headers, timeout=60)
    else:
        r = requests.post(f"https://api.replicate.com/v1/models/{slug}/predictions",
                          json={"input": payload["input"]}, headers=headers, timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Replicate 创建预测失败 {r.status_code}: {r.text[:300]}")
    pred = r.json()
    progress(0.3, "云端推理排队中")
    poll_url = pred.get("urls", {}).get("get")
    while True:
        pr = requests.get(poll_url, headers=headers, timeout=60)
        pr.raise_for_status()
        pred = pr.json()
        st = pred.get("status")
        if st == "succeeded":
            break
        if st in ("failed", "canceled"):
            raise RuntimeError(f"Replicate 推理失败: {str(pred.get('error'))[:300]}")
        progress(min(0.8, 0.3 + 0.05 * (1 if st == "processing" else 0)), f"云端推理 {st}")
        import time
        time.sleep(3)
    outputs = pred.get("output") or []
    url = outputs[0] if isinstance(outputs, list) else outputs
    if not url:
        raise RuntimeError("Replicate 未返回输出")
    progress(0.9, "下载云端结果")
    out = Path(config.DATA_DIR / "uploads" / "_cloud_out.wav")
    ar = requests.get(url, timeout=300)
    ar.raise_for_status()
    out.write_bytes(ar.content)
    return out


def _elevenlabs_convert(wav_path: Path, profile: VoiceProfile,
                        progress: ProgressCB) -> Path:
    """ElevenLabs Voice Changer：multipart 上传 → 返回转换后音频。"""
    voice_id = config.CLOUD_VOICE_ID
    if not voice_id:
        raise RuntimeError("缺少 VC_CLOUD_VOICE_ID（ElevenLabs Voice ID）")
    progress(0.4, "云端转换中")
    r = requests.post(
        f"https://api.elevenlabs.io/v1/voice-changer/{voice_id}",
        headers={"xi-api-key": config.CLOUD_API_KEY},
        files={"audio": ("vocal.wav", wav_path.read_bytes(), "audio/wav")},
        data={"remove_background_noise": "true"},
        timeout=_TIMEOUT_S)
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs 转换失败 {r.status_code}: {r.text[:300]}")
    out = Path(config.DATA_DIR / "uploads" / "_cloud_out.wav")
    out.write_bytes(r.content)
    return out


_ADAPTERS = {"replicate": _replicate_convert, "elevenlabs": _elevenlabs_convert}


class CloudEngine(BaseVoiceEngine):
    name = "cloud"
    label = "云端 API · 按量付费"
    kind = "api"
    description = ("调用付费云端 API 做声音转换（结果进入相同的混音链）。"
                   "需要配置环境变量：VC_CLOUD_PROVIDER（replicate/elevenlabs）、"
                   "VC_CLOUD_API_KEY、VC_CLOUD_VOICE_ID。auto 模式永不自动使用本引擎，"
                   "仅在你明确选中时产生按量计费调用。")

    def _configured(self) -> tuple[bool, str]:
        if not config.CLOUD_PROVIDER:
            return False, "未配置：设置 VC_CLOUD_PROVIDER / VC_CLOUD_API_KEY / VC_CLOUD_VOICE_ID（见 README）"
        if config.CLOUD_PROVIDER not in _ADAPTERS:
            return False, f"未知 provider: {config.CLOUD_PROVIDER}（可选 replicate/elevenlabs）"
        if not config.CLOUD_API_KEY:
            return False, "缺少 VC_CLOUD_API_KEY"
        if not config.CLOUD_VOICE_ID:
            return False, "缺少 VC_CLOUD_VOICE_ID"
        return True, ""

    def available(self) -> tuple[bool, str]:
        ok, why = self._configured()
        if not ok:
            return False, why
        return True, f"已配置 · {config.CLOUD_PROVIDER}（按量计费）"

    def build_profile(self, ref_wav: Path, cache_dir: Path,
                      progress: ProgressCB = lambda p, m: None) -> VoiceProfile:
        progress(0.5, "准备云端参数")
        profile = VoiceProfile(sample_id=Path(ref_wav).parent.name, ref_wav=Path(ref_wav))
        profile.extra["ref_wav"] = str(Path(ref_wav).resolve())
        progress(1.0, "声音分析完成")
        return profile

    def convert(self, source_vocal: Path, profile: VoiceProfile, out_wav: Path,
                pitch_shift_semitones: float = 0.0,
                progress: ProgressCB = lambda p, m: None) -> dict:
        ok, why = self._configured()
        if not ok:
            raise RuntimeError(why)
        progress(0.15, f"调用云端 API（{config.CLOUD_PROVIDER}，按量计费）")
        raw = _ADAPTERS[config.CLOUD_PROVIDER](Path(source_vocal).resolve(), profile, progress)
        progress(0.85, "后处理")
        data, sr = load_wav(raw)
        data = to_stereo(data)
        if sr != config.TARGET_SR:
            data = resample(data, sr, config.TARGET_SR)
        if abs(pitch_shift_semitones) > 1e-3:
            data = pitch_shift(data, config.TARGET_SR, pitch_shift_semitones)
        data = peak_normalize(data, 0.9)
        save_wav(out_wav, data, config.TARGET_SR)
        raw.unlink(missing_ok=True)
        progress(1.0, "转换完成")
        return {"engine": f"已使用云端 API（{config.CLOUD_PROVIDER}，按量计费）"}
