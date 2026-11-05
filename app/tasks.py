"""翻唱流水线与后台任务执行。

流水线：预处理 → 声音分析 → 轨道分离 → 声音转换 → 混音 → 导出
任何一步失败时按策略回退（高质量组件失败 → 轻量兜底组件），并在任务日志中说明。
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from . import config
from .audio.dsp import fade, hf_energy_ratio, resample, to_stereo
from .audio.io import decode_to_wav, export_mp3, load_wav, save_wav, wav_duration
from .audio.mix import (auto_vocal_gain, control_vocal_peaks, gate_to_source,
                        master, match_spectral_envelope, mix_tracks, vocal_bus)
from .separation import SeparationResult, resolve_separator
from .separation import cache as sep_cache
from .separation.center_extract import CenterExtractSeparator
from .voice import PitchFitVoiceEngine, resolve_engine

log = logging.getLogger("pipeline")

STAGES = ["预处理", "声音分析", "轨道分离", "声音转换", "混音", "导出"]
RANGES = ((0.00, 0.10), (0.10, 0.20), (0.20, 0.55), (0.55, 0.85), (0.85, 0.95), (0.95, 1.00))


class JobCancelled(Exception):
    """任务被用户取消（在阶段边界检查点抛出）。"""


def _short_reason(exc: BaseException, limit: int = 240) -> str:
    """把异常压成单行短句：任务日志里不该出现多行 traceback。"""
    text = " ".join(f"{exc}".split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _stage_progress(progress, idx: int):
    """把阶段内 0~1 进度映射为全局进度。"""
    lo, hi = RANGES[idx]

    def cb(p: float, _msg: str = ""):
        p = max(0.0, min(1.0, float(p)))
        progress(STAGES[idx], lo + p * (hi - lo))

    return cb


def _execute(*, song_path: Path, ref_wav: Path, engine_name: str, separator_name: str,
             options: dict, job_dir: Path, progress, notes: list[str],
             should_cancel: Callable[[], bool] = lambda: False) -> dict:
    """执行完整翻唱流水线（阻塞）。progress(stage_name, global_0_1)。

    should_cancel 在各阶段边界被轮询，返回 True 时抛出 JobCancelled。
    注意：分离/转换子进程运行中无法中断，取消会等当前阶段结束才生效。
    """
    options = options or {}

    def checkpoint(idx: int):
        if should_cancel():
            raise JobCancelled(f"任务在「{STAGES[idx]}」阶段被取消")

    vocal_gain = float(options.get("vocal_gain_db", 0.0))
    inst_gain = float(options.get("instrumental_gain_db", 0.0))
    manual_pitch = float(options.get("pitch_shift_semitones", 0.0))
    auto_balance = bool(options.get("auto_balance", True))
    denoise_gate = bool(options.get("denoise_gate", True))
    tau = options.get("tau")
    job_dir = Path(job_dir)

    # 1) 预处理：统一解码为 44.1kHz 双声道 wav
    checkpoint(0)
    progress(STAGES[0], 0.3)
    song_wav = job_dir / "song.wav"
    decode_to_wav(song_path, song_wav, config.TARGET_SR)
    dur = wav_duration(song_wav)
    if dur > config.MAX_SONG_MINUTES * 60:
        raise RuntimeError(f"歌曲时长 {dur / 60:.1f} 分钟，超过上限 {config.MAX_SONG_MINUTES:g} 分钟")
    progress(STAGES[0], 1.0)

    # 2) 声音分析：构建用户声音档案
    checkpoint(1)
    engine = resolve_engine(engine_name)
    profile = engine.build_profile(ref_wav, job_dir / "profile",
                                   progress=_stage_progress(progress, 1))
    if tau is not None:
        profile.extra["tau"] = max(0.05, min(0.9, float(tau)))

    # 3) 轨道分离（同歌同模型命中缓存则直接复用；demucs 失败自动回退中心声道）
    checkpoint(2)
    separator, sep_note = resolve_separator(separator_name)
    if sep_note:
        notes.append(sep_note)
    stems = job_dir / "stems"
    model_key = separator.name + (f"-{separator.model}" if hasattr(separator, "model") else "")
    cacheable = config.SEPARATION_CACHE and separator.kind == "model"
    cached = sep_cache.get_cached(song_path, model_key) if cacheable else None
    if cached:
        v, i = sep_cache.restore_into(cached[0], cached[1], stems)
        sep_res = SeparationResult(vocals=v, instrumental=i)
        notes.append("命中分离缓存（同一首歌 + 同一模型），跳过分离开销")
        progress(STAGES[2], 1.0)
    else:
        try:
            sep_res = separator.separate(song_wav, stems, progress=_stage_progress(progress, 2))
            if cacheable:
                sep_cache.store(sep_res.vocals, sep_res.instrumental, song_path, model_key,
                                meta={"separator": separator.name})
        except Exception as e:
            if separator.name != "center":
                notes.append(f"分离器 {separator.name} 失败，已自动回退到中心声道提取：{_short_reason(e)}")
                log.warning("分离器 %s 失败，回退: %s", separator.name, e)
                sep_res = CenterExtractSeparator().separate(song_wav, stems,
                                                            progress=_stage_progress(progress, 2))
            else:
                raise

    # 3b) 人声去混响：分离人声自带房间混响，混响尾巴会让转换产生拖影毛刺，
    #     转换前先"烘干"（Jammable/kits.ai 同款流水线的关键步骤）
    if config.DEREVERB_ENABLED:
        progress(STAGES[2], 0.95)
        try:
            from .separation import dereverb as dereverb_mod
            ok, why = dereverb_mod.available()
            if ok:
                _cb2 = _stage_progress(progress, 2)
                dry = dereverb_mod.dereverb(
                    sep_res.vocals, stems,
                    progress=lambda p, m="": _cb2(0.9 + 0.08 * p))
                sep_res = SeparationResult(vocals=dry, instrumental=sep_res.instrumental)
                notes.append("人声去混响：已对分离人声做 DeEcho-DeReverb 烘干（转换源更干净）")
        except Exception as e:
            notes.append(f"人声去混响失败（跳过，不影响生成）：{_short_reason(e)}")

    # 4) 声音转换（引擎失败自动回退演示引擎）
    checkpoint(3)
    converted = job_dir / "converted_vocals.wav"
    engine_notes: dict = {}
    try:
        engine_notes = engine.convert(sep_res.vocals, profile, converted,
                                      pitch_shift_semitones=manual_pitch,
                                      progress=_stage_progress(progress, 3)) or {}
    except Exception as e:
        if engine.name != "pitchfit":
            notes.append(f"声音引擎 {engine.name} 转换失败，已自动回退到演示引擎：{_short_reason(e)}")
            log.warning("引擎 %s 失败，回退: %s", engine.name, e)
            engine = PitchFitVoiceEngine()
            profile = engine.build_profile(ref_wav, job_dir / "profile",
                                           progress=_stage_progress(progress, 1))
            engine_notes = engine.convert(sep_res.vocals, profile, converted,
                                          pitch_shift_semitones=manual_pitch,
                                          progress=_stage_progress(progress, 3)) or {}
        else:
            raise
    for k, v in engine_notes.items():
        notes.append(v)

    # 5) 混音 + 母带
    checkpoint(4)
    progress(STAGES[4], 0.1)
    vocals, sr = load_wav(converted)
    inst, sr2 = load_wav(sep_res.instrumental)
    src_vocals, sr3 = load_wav(sep_res.vocals)
    if sr2 != sr:
        inst = resample(inst, sr2, sr)
    if sr3 != sr:
        src_vocals = resample(src_vocals, sr3, sr)
    vocals = to_stereo(vocals)
    inst = to_stereo(inst)
    src_vocals = to_stereo(src_vocals)

    # 5-pre) 转换质量体检：转换人声高频能量远低于原唱是"输出偏闷"的典型特征
    try:
        r_src = hf_energy_ratio(src_vocals, sr)
        r_conv = hf_energy_ratio(vocals, sr)
        if r_src is not None and r_conv is not None and r_src > 0.08 and r_conv < r_src * 0.2:
            notes.append(f"转换人声高频明显缺失（>5kHz 能量占比 {r_conv * 100:.1f}%，"
                         f"原唱 {r_src * 100:.1f}%）——输出偏闷，建议更换引擎或调高 tau / 更换声音样本")
    except Exception:
        pass

    # 5a) 静音段噪声门：原唱人声不唱的地方把转换后人声压到静音，
    #     清掉句间的模型残响/噪声，避免伴奏里混进"底噪人声"
    if denoise_gate:
        progress(STAGES[4], 0.2)
        try:
            from .audio.mix import vocal_activity_mask
            active_pct = float(vocal_activity_mask(src_vocals, sr).mean() * 100)
            if active_pct < 5.0:
                notes.append(f"原唱人声活跃段仅 {active_pct:.0f}%，疑似分离异常——跳过静音段清理以防误伤人声")
            else:
                vocals, gated_pct = gate_to_source(vocals, src_vocals, sr)
                if gated_pct > 1.0:
                    notes.append(f"静音段清理：已抑制句间非演唱片段（约 {gated_pct:.0f}% 时长），"
                                 "去除转换残留噪声")
        except Exception as e:
            notes.append(f"静音段清理失败（跳过）：{e}")

    # 5a1) 频谱包络匹配：把转换人声的分频段能量对齐到原唱人声
    #      （治"发闷/低频浑浊"的根治方案，逐歌自动测算，替代手调静态 EQ）
    if config.SPECTRAL_MATCH:
        progress(STAGES[4], 0.15)
        try:
            vocals = match_spectral_envelope(vocals, src_vocals, sr)
            notes.append("频谱包络匹配：已将转换人声分频段能量对齐到原唱人声")
        except Exception as e:
            notes.append(f"频谱包络匹配失败（跳过）：{_short_reason(e)}")

    # 5a2) 人声总线：低切去浊 → 压缩增密 → 去刺 → 板式混响粘合。
    #      转换人声低频浑浊、动态松散、干得像卡拉OK，商业人声必经的"总线"处理
    progress(STAGES[4], 0.25)
    try:
        vocals = vocal_bus(vocals, sr)
        notes.append("人声总线：低切去浊 + 压缩增密 + 板式混响（人声融进伴奏）")
    except Exception as e:
        notes.append(f"人声总线处理失败（跳过）：{_short_reason(e)}")

    # 5b) 人声电平自动对齐：把转换后人声的响度对齐到原唱人声水平，
    #     修正不同引擎/不同歌曲下人声忽大忽小的问题（手动增益在其上叠加）
    if auto_balance:
        progress(STAGES[4], 0.3)
        try:
            g = auto_vocal_gain(src_vocals, vocals, sr)
            if g:
                vocals = vocals * (10 ** (g / 20.0))
                note = f"人声电平自动对齐：{g:+.1f} dB（转换后响度 → 原唱人声响度）"
                if abs(g) >= 8.9:
                    note += "——已达对齐上限，转换输出电平异常，请查看引擎回退日志"
                notes.append(note)
        except Exception as e:
            notes.append(f"人声电平对齐失败（跳过）：{e}")

    # 5c) 人声峰值控制：响度对齐后人声瞬时峰值可能超过 1.0（写 WAV 硬削波、
    #     母带限幅器过载泵感），先对人声单独软限幅
    progress(STAGES[4], 0.4)
    try:
        vocals, limited_db = control_vocal_peaks(vocals, sr)
        if limited_db >= 0.5:
            notes.append(f"人声峰值控制：软限幅 {limited_db:.1f} dB（防削波与母带过载）")
    except Exception as e:
        notes.append(f"人声峰值控制失败（跳过）：{e}")

    progress(STAGES[4], 0.5)
    # 处理后的人声写回 converted_vocals.wav，保证"分轨试听"听到的与成品一致
    save_wav(converted, vocals, sr)
    mixed = mix_tracks(vocals, inst, vocal_gain_db=vocal_gain,
                       instrumental_gain_db=inst_gain)
    progress(STAGES[4], 0.7)
    mixed = master(mixed, target_db=-14.0, sr=sr)
    mixed = fade(mixed, sr, in_s=0.015, out_s=0.05)
    cover_wav = job_dir / "cover.wav"
    save_wav(cover_wav, mixed, sr)
    progress(STAGES[4], 1.0)

    # 6) 导出（wav + mp3）
    checkpoint(5)
    cover_mp3 = export_mp3(cover_wav, job_dir / "cover.mp3")
    progress(STAGES[5], 1.0)

    return {
        "cover_wav": str(cover_wav),
        "cover_mp3": str(cover_mp3) if cover_mp3 else None,
        "vocals": str(sep_res.vocals),
        "instrumental": str(sep_res.instrumental),
        "converted": str(converted),
        "engine": engine.name,
        "separator": separator.name,
        "duration_seconds": round(dur, 2),
    }


class Pipeline:
    """后台任务队列：默认单并发（音频处理是 CPU 密集型）。"""

    def __init__(self, store):
        self.store = store
        self._q: queue.Queue = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=max(1, config.MAX_CONCURRENT_JOBS),
                                        thread_name_prefix="cover")
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cancel_events: dict[str, threading.Event] = {}

    def start(self):
        # 服务重启会丢失内存队列：把中断在 processing 的任务自动重新排队
        # （分离结果有缓存，重跑损失可控；排队中任务本就保留在库里）
        for job in self.store.list_jobs():
            if job["status"] == "processing":
                log.warning("任务 %s 因服务重启中断，自动重新排队", job["id"])
                notes = list(job.get("notes") or [])
                notes.append("检测到服务重启导致任务中断，已自动重新排队")
                self.store.update_job(job["id"], status="queued", stage=STAGES[0],
                                      progress=0.01, error=None,
                                      notes=json.dumps(notes, ensure_ascii=False))
                self._q.put(job["id"])
        self._thread = threading.Thread(target=self._loop, name="job-dispatcher", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._q.put(None)
        # 不阻塞等待在途任务完成（worker 是普通线程，进程退出前自然结束）
        self._pool.shutdown(wait=False)

    def enqueue(self, jid: str):
        self.store.update_job(jid, status="queued")
        self._q.put(jid)

    def cancel(self, jid: str) -> str:
        """取消任务。返回 "cancelled"（已终态）或 "cancelling"（等阶段边界生效）。

        排队中：直接置为 cancelled，worker 取到后看到非 queued 状态即跳过；
        进行中：设置取消事件，worker 在下一个阶段边界抛出 JobCancelled。
        """
        job = self.store.get_job(jid)
        if not job:
            return "unknown"
        if job["status"] == "queued":
            # 同时置事件：若 worker 恰在状态切换窗口内已开跑，检查点也能立即感知
            self._cancel_events.setdefault(jid, threading.Event()).set()
            self.store.update_job(jid, status="cancelled", stage="已取消")
            return "cancelled"
        if job["status"] == "processing":
            self._cancel_events.setdefault(jid, threading.Event()).set()
            return "cancelling"
        return job["status"]

    def is_running(self, jid: str) -> bool:
        """worker 线程是否正在执行该任务（供 API 展示存活状态）。"""
        return jid in self._cancel_events

    def _loop(self):
        while not self._stop.is_set():
            jid = self._q.get()
            if jid is None:
                break
            self._pool.submit(self._run_job, jid)

    def _run_job(self, jid: str):
        job = self.store.get_job(jid)
        if not job or job["status"] != "queued":
            return
        job_dir = config.JOBS_DIR / jid
        job_dir.mkdir(parents=True, exist_ok=True)
        notes: list[str] = []
        sample = self.store.get_sample(job["voice_sample_id"])
        cancel_event = threading.Event()
        self._cancel_events[jid] = cancel_event

        def progress(stage: str, p: float):
            self.store.update_job(jid, stage=stage, progress=round(float(p), 4))

        self.store.update_job(jid, status="processing", stage=STAGES[0], progress=0.01, error=None)
        try:
            if not sample:
                raise RuntimeError(f"声音样本不存在: {job['voice_sample_id']}")
            result = _execute(song_path=Path(job["song_path"]), ref_wav=Path(sample["path"]),
                              engine_name=job["engine"], separator_name=job["separator"],
                              options=job.get("options") or {}, job_dir=job_dir,
                              progress=progress, notes=notes,
                              should_cancel=cancel_event.is_set)
            self.store.update_job(jid, status="finished", stage="完成", progress=1.0,
                                  notes=json.dumps(notes, ensure_ascii=False))
            log.info("任务 %s 完成（engine=%s separator=%s）", jid, result["engine"], result["separator"])
        except JobCancelled:
            notes.append("任务已被用户取消")
            log.info("任务 %s 已取消", jid)
            self.store.update_job(jid, status="cancelled", stage="已取消",
                                  notes=json.dumps(notes, ensure_ascii=False))
        except Exception as e:
            log.exception("任务 %s 失败", jid)
            try:
                (job_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            except Exception:
                pass
            self.store.update_job(jid, status="failed", error=f"{type(e).__name__}: {e}",
                                  notes=json.dumps(notes, ensure_ascii=False))
        finally:
            self._cancel_events.pop(jid, None)
