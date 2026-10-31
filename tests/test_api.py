"""端到端 API 测试：上传样本 → 创建任务 → 轮询 → 下载结果（走 HTTP 全链路）。"""
import time

from fastapi.testclient import TestClient

from conftest import make_sample_wav_bytes, make_song_wav_bytes
from app.main import app


def test_health_and_engines(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True

    r = client.get("/api/engines")
    names = [e["name"] for e in r.json()["engines"]]
    assert "pitchfit" in names
    by_name = {e["name"]: e for e in r.json()["engines"]}
    assert by_name["pitchfit"]["available"] is True


def test_full_cover_flow(client):
    # 1) 上传声音样本
    r = client.post("/api/voice-samples",
                    files={"file": ("ref.wav", make_sample_wav_bytes(8.0, 200.0), "audio/wav")})
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    assert r.json()["duration"] >= 7.5

    # 2) 创建翻唱任务（显式选择演示引擎与中心声道分离，保证测试确定性）
    r = client.post("/api/covers",
                    data={"voice_sample_id": sid, "engine": "pitchfit", "separator": "center",
                          "vocal_gain_db": 1.0, "pitch_shift_semitones": 0.0, "title": "测试歌曲"},
                    files={"song": ("song.wav", make_song_wav_bytes(4.0, 440.0), "audio/wav")})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]

    # 3) 轮询直到完成
    deadline = time.time() + 180
    job = {}
    while time.time() < deadline:
        job = client.get(f"/api/covers/{jid}").json()
        if job["status"] in ("finished", "failed"):
            break
        time.sleep(0.4)
    assert job["status"] == "finished", f"任务失败: {job.get('error')}"

    # 4) 下载成品
    r = client.get(f"/api/covers/{jid}/result")
    assert r.status_code == 200
    assert len(r.content) > 1000

    # 5) 分轨预览
    for kind in ("song", "vocals", "instrumental", "converted"):
        r = client.get(f"/api/covers/{jid}/stems/{kind}")
        assert r.status_code == 200, kind

    # 6) 任务列表
    jobs = client.get("/api/covers").json()
    assert any(x["id"] == jid for x in jobs)

    # 7) 删除
    assert client.delete(f"/api/covers/{jid}").status_code == 200


def test_sample_too_short_rejected(client):
    r = client.post("/api/voice-samples",
                    files={"file": ("ref.wav", make_sample_wav_bytes(2.0, 200.0), "audio/wav")})
    assert r.status_code == 400


def test_silent_sample_rejected(client):
    """静音样本必须拒绝：否则提取出随机音色向量。"""
    import io
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros((int(8 * 44100), 2), dtype=np.float32), 44100,
             subtype="PCM_16", format="WAV")
    r = client.post("/api/voice-samples",
                    files={"file": ("silent.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 400
    assert "静音" in r.json()["detail"]


def test_cover_requires_existing_sample(client):
    r = client.post("/api/covers",
                    data={"voice_sample_id": "ffffffffffffffff", "engine": "pitchfit",
                          "separator": "center"},
                    files={"song": ("song.wav", make_song_wav_bytes(2.0), "audio/wav")})
    assert r.status_code == 404


def test_cancel_finished_job_conflict(client):
    """未在跑的任务取消 → 409。"""
    r = client.post("/api/voice-samples",
                    files={"file": ("ref.wav", make_sample_wav_bytes(8.0, 200.0), "audio/wav")})
    sid = r.json()["id"]
    r = client.post("/api/covers",
                    data={"voice_sample_id": sid, "engine": "pitchfit", "separator": "center"},
                    files={"song": ("s.wav", make_song_wav_bytes(2.0), "audio/wav")})
    jid = r.json()["id"]
    # 轮询到终态（任务很小，通常在首次取消前已完成）
    deadline = time.time() + 120
    job = {}
    while time.time() < deadline:
        job = client.get(f"/api/covers/{jid}").json()
        if job["status"] in ("finished", "failed"):
            break
        time.sleep(0.3)
    r = client.post(f"/api/covers/{jid}/cancel")
    assert r.status_code == 409
    client.delete(f"/api/covers/{jid}")


def test_cancel_queued_job(client):
    """并发=1 时第二个任务排队；取消它应得到 cancelled 终态。"""
    r = client.post("/api/voice-samples",
                    files={"file": ("ref.wav", make_sample_wav_bytes(8.0, 200.0), "audio/wav")})
    sid = r.json()["id"]
    ids = []
    for _ in range(2):
        r = client.post("/api/covers",
                        data={"voice_sample_id": sid, "engine": "pitchfit", "separator": "center"},
                        files={"song": ("s.wav", make_song_wav_bytes(3.0), "audio/wav")})
        ids.append(r.json()["id"])
    first, second = ids
    # 取消第二个任务（若已完成则跳过断言，避免时序脆弱）
    deadline = time.time() + 10
    status = client.get(f"/api/covers/{second}").json()["status"]
    while status in ("queued", "processing") and time.time() < deadline:
        if status == "queued":
            break
        time.sleep(0.1)
        status = client.get(f"/api/covers/{second}").json()["status"]
    if status == "queued":
        r = client.post(f"/api/covers/{second}/cancel")
        assert r.status_code == 200 and r.json()["status"] == "cancelled"
        assert client.get(f"/api/covers/{second}").json()["status"] == "cancelled"
        assert client.delete(f"/api/covers/{second}").status_code == 200
    # 清理第一个任务
    deadline = time.time() + 120
    while time.time() < deadline:
        job = client.get(f"/api/covers/{first}").json()
        if job["status"] in ("finished", "failed", "cancelled"):
            break
        time.sleep(0.4)
    client.delete(f"/api/covers/{first}")
