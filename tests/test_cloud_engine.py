"""云端付费 API 引擎测试：可用性状态与 auto 模式计费安全。"""
import pytest

from app import config
from app.voice import CloudEngine, get_engine, iter_engines, resolve_engine


def test_cloud_unconfigured_unavailable(monkeypatch):
    monkeypatch.setattr(config, "CLOUD_PROVIDER", "")
    monkeypatch.setattr(config, "CLOUD_API_KEY", "")
    ok, reason = CloudEngine().available()
    assert ok is False
    assert "VC_CLOUD_PROVIDER" in reason


def test_cloud_missing_key(monkeypatch):
    monkeypatch.setattr(config, "CLOUD_PROVIDER", "elevenlabs")
    monkeypatch.setattr(config, "CLOUD_API_KEY", "")
    ok, reason = CloudEngine().available()
    assert ok is False and "API_KEY" in reason


def test_cloud_missing_voice_id(monkeypatch):
    monkeypatch.setattr(config, "CLOUD_PROVIDER", "elevenlabs")
    monkeypatch.setattr(config, "CLOUD_API_KEY", "sk-test")
    monkeypatch.setattr(config, "CLOUD_VOICE_ID", "")
    ok, reason = CloudEngine().available()
    assert ok is False and "VOICE_ID" in reason


def test_cloud_unknown_provider(monkeypatch):
    monkeypatch.setattr(config, "CLOUD_PROVIDER", "nope")
    monkeypatch.setattr(config, "CLOUD_API_KEY", "sk-test")
    monkeypatch.setattr(config, "CLOUD_VOICE_ID", "v1")
    ok, reason = CloudEngine().available()
    assert ok is False and "未知 provider" in reason


def test_cloud_configured_available(monkeypatch):
    monkeypatch.setattr(config, "CLOUD_PROVIDER", "replicate")
    monkeypatch.setattr(config, "CLOUD_API_KEY", "r8_test")
    monkeypatch.setattr(config, "CLOUD_VOICE_ID", "owner/model")
    ok, reason = CloudEngine().available()
    assert ok is True and "replicate" in reason


def test_auto_mode_never_picks_cloud(monkeypatch):
    """安全设计：即使云端引擎完全配置好，auto 也绝不能选中付费引擎。"""
    monkeypatch.setattr(config, "CLOUD_PROVIDER", "replicate")
    monkeypatch.setattr(config, "CLOUD_API_KEY", "r8_test")
    monkeypatch.setattr(config, "CLOUD_VOICE_ID", "owner/model")
    engine = resolve_engine("auto")
    assert engine.kind != "api"
    # 明确选择时可以拿到
    assert get_engine("cloud").kind == "api"
    assert any(e.name == "cloud" for e in iter_engines())
