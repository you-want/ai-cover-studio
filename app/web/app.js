/* 声替 CoverMe 前端逻辑（无框架依赖） */
"use strict";

const $ = (s) => document.querySelector(s);

const AUDIO_EXT = /\.(wav|mp3|m4a|aac|ogg|oga|flac|webm|aiff|aif|opus|wma)$/i;

const state = {
  sampleId: null,
  songs: [],
  engine: "auto",
  separator: "auto",
  activeId: null,
  pollTimer: null,
};

/* ---------------- 通用 ---------------- */

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch (_) { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

let toastTimer = null;
function toast(msg, isErr = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isErr ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), isErr ? 6000 : 2600);
}

function fmtDur(s) {
  s = Math.round(s);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function updateGoButton() {
  const ready = !!(state.sampleId && state.songs.length);
  $("#btn-go").disabled = !ready;
  $("#go-hint").textContent = ready ? "" :
    (!state.sampleId ? "先完成第 1 步：录制或上传你的声音" : "再选择一首想唱的歌");
}

/* ---------------- 声音样本 ---------------- */

let mediaRecorder = null;
let recChunks = [];
let recTimerInt = null;
let recSeconds = 0;

/* ---------------- 录音教练：实时电平表 + 体检 ---------------- */

let recAudioCtx = null;
let recStats = { totalFrames: 0, clippedFrames: 0, maxRms: 0 };

function startLevelMeter(stream) {
  try {
    recAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = recAudioCtx.createMediaStreamSource(stream);
    const analyser = recAudioCtx.createAnalyser();
    analyser.fftSize = 1024;
    src.connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    recStats = { totalFrames: 0, clippedFrames: 0, maxRms: 0 };
    $("#level-meter").classList.remove("hidden");
    const loop = () => {
      if (!mediaRecorder || mediaRecorder.state !== "recording") return;
      analyser.getFloatTimeDomainData(buf);
      let sum = 0, peak = 0;
      for (let i = 0; i < buf.length; i++) {
        sum += buf[i] * buf[i];
        const a = Math.abs(buf[i]);
        if (a > peak) peak = a;
      }
      const rms = Math.sqrt(sum / buf.length);
      recStats.totalFrames += 1;
      if (peak >= 0.99) recStats.clippedFrames += 1;
      if (rms > recStats.maxRms) recStats.maxRms = rms;
      const pct = Math.min(100, Math.round(rms * 320));
      $("#level-fill").style.width = pct + "%";
      $("#level-fill").style.background =
        pct > 88 ? "var(--err)" : pct < 8 ? "#eab308" : "var(--ok)";
      $("#level-hint").textContent =
        pct > 88 ? "🔴 太响了，会爆音" : pct < 8 ? "🟡 太轻了，靠近点" : "🟢 音量刚刚好";
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  } catch (_) { /* 电平表失败不影响录音 */ }
}

function stopLevelMeter() {
  $("#level-meter").classList.add("hidden");
  if (recAudioCtx) { recAudioCtx.close().catch(() => {}); recAudioCtx = null; }
}

$("#btn-record").addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    toast("无法访问麦克风，请检查系统权限，或改用「上传音频文件」", true);
    return;
  }
  const mime = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm"
    : MediaRecorder.isTypeSupported("audio/mp4") ? "audio/mp4" : "";
  mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  recChunks = [];
  startLevelMeter(stream);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size) recChunks.push(e.data); };
  mediaRecorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    stopLevelMeter();
    clearInterval(recTimerInt);
    $("#btn-record").textContent = "🎙 开始录音";
    $("#btn-record").classList.remove("recording");
    $("#rec-timer").classList.add("hidden");
    $("#rec-hint").textContent = "";
    if (recSeconds < 5) {
      toast(`只录了 ${recSeconds} 秒，太短啦，请至少录 10 秒`, true);
      recSeconds = 0;
      return;
    }
    // 录音教练体检：爆音 / 声音太小 → 录完即时提醒
    if (recStats.clippedFrames > recStats.totalFrames * 0.02) {
      toast("⚠️ 检测到较多爆音——离麦克风远一点、音量小一点，重录效果更好", true);
    } else if (recStats.maxRms < 0.02) {
      toast("⚠️ 声音太小（房间噪比人声还响）——靠近麦克风重录，效果更好", true);
    }
    const ext = mediaRecorder.mimeType.includes("mp4") ? "m4a" : "webm";
    const file = new File(recChunks, `我的录音.${ext}`, { type: mediaRecorder.mimeType || "audio/webm" });
    recSeconds = 0;
    await uploadVoice(file);
  };
  mediaRecorder.start();
  recSeconds = 0;
  $("#btn-record").textContent = "⏹ 停止录音";
  $("#btn-record").classList.add("recording");
  $("#rec-timer").classList.remove("hidden");
  $("#rec-hint").textContent = "录音中…（最长 60 秒）";
  recTimerInt = setInterval(() => {
    recSeconds += 1;
    $("#rec-timer").textContent = fmtDur(recSeconds);
    if (recSeconds >= 60 && mediaRecorder.state === "recording") mediaRecorder.stop();
  }, 1000);
});

async function uploadVoice(file) {
  toast("正在上传声音样本…");
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await api("/api/voice-samples", { method: "POST", body: fd });
    state.sampleId = r.id;
    if (r.quality === "clipped") toast("⚠️ 这段录音爆音较多，建议重录一版（音色会受损）", true);
    else if (r.quality === "quiet") toast("⚠️ 这段录音声音太轻，建议重录一版", true);
    // 试听始终走服务端存下的 ref.wav（持久 URL），避免 Blob URL 刷新后失效
    $("#rec-preview").src = `/api/voice-samples/${r.id}/preview`;
    $("#rec-preview").classList.remove("hidden");
    toast(`声音样本已就绪（${r.duration} 秒）`);
    await loadSamples();
  } catch (e) {
    toast(`声音样本上传失败：${e.message}`, true);
  }
  updateGoButton();
}

$("#voice-file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (f) await uploadVoice(f);
  e.target.value = "";
});

async function loadSamples() {
  const list = $("#sample-list");
  try {
    const samples = await api("/api/voice-samples");
    list.innerHTML = "";
    if (samples.length && !samples.some((s) => s.id === state.sampleId)) {
      state.sampleId = samples[0].id;
    }
    for (const s of samples) {
      const div = document.createElement("div");
      div.className = "sample" + (s.id === state.sampleId ? " selected" : "");
      div.innerHTML = `<span class="dot"></span>
        <span class="meta"><div class="name"></div><div class="dur">${s.duration} 秒 · ${fmtTime(s.created_at)}</div></span>
        <button class="play" title="试听">▶</button>
        <button class="del" title="删除">✕</button>`;
      div.querySelector(".name").textContent = s.name;
      div.addEventListener("click", () => {
        state.sampleId = s.id;
        loadSamples();
        updateGoButton();
      });
      div.querySelector(".play").addEventListener("click", (e) => {
        e.stopPropagation();
        const a = $("#rec-preview");
        a.src = `/api/voice-samples/${s.id}/preview`;
        a.classList.remove("hidden");
        a.currentTime = 0;
        a.play().catch(() => {});
      });
      div.querySelector(".del").addEventListener("click", async (e) => {
        e.stopPropagation();
        try {
          await api(`/api/voice-samples/${s.id}`, { method: "DELETE" });
          if (state.sampleId === s.id) state.sampleId = null;
          await loadSamples();
          updateGoButton();
        } catch (err) { toast(err.message, true); }
      });
      list.appendChild(div);
    }
    if (!samples.length) {
      list.innerHTML = `<div class="loading">还没有声音样本，先录一段吧</div>`;
      state.sampleId = null;
    }
  } catch (e) {
    list.innerHTML = `<div class="loading">样本加载失败：${e.message}</div>`;
  }
  updateGoButton();
}

/* ---------------- 歌曲（支持批量多选） ---------------- */

const drop = $("#drop");
drop.addEventListener("click", () => $("#song-file").click());
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  addSongs(Array.from(e.dataTransfer.files));
});
$("#song-file").addEventListener("change", (e) => {
  addSongs(Array.from(e.target.files));
  e.target.value = "";
});

function addSongs(files) {
  const audio = files.filter((f) => AUDIO_EXT.test(f.name));
  const skipped = files.length - audio.length;
  if (skipped > 0) toast(`已跳过 ${skipped} 个非音频文件`, true);
  for (const f of audio) {
    if (!state.songs.some((s) => s.name === f.name && s.size === f.size)) state.songs.push(f);
  }
  renderSongs();
  updateGoButton();
  setSmartEnabled();
}

function removeSong(idx) {
  state.songs.splice(idx, 1);
  renderSongs();
  updateGoButton();
  setSmartEnabled();
}

function renderSongs() {
  const box = $("#song-list");
  box.innerHTML = "";
  box.classList.toggle("hidden", !state.songs.length);
  for (const [i, f] of state.songs.entries()) {
    const div = document.createElement("div");
    div.className = "song-info";
    div.innerHTML = `<span class="name"></span><span class="dur"></span>
      <button class="btn ghost small">移除</button>`;
    div.querySelector(".name").textContent = `🎵 ${f.name}`;
    div.querySelector(".dur").textContent = `${(f.size / 1024 / 1024).toFixed(1)} MB`;
    div.querySelector("button").addEventListener("click", () => removeSong(i));
    box.appendChild(div);
  }
}

/* ---------------- 智能匹配（懒人模式） ---------------- */

$("#btn-smart").addEventListener("click", async () => {
  if (!state.songs.length) { toast("先选择一首歌", true); return; }
  const box = $("#smart-result");
  box.classList.remove("hidden");
  box.innerHTML = "🎵 正在分析歌曲（人声分离 + 音域/亮度测定，约 1~2 分钟）…";
  $("#btn-smart").disabled = true;
  const fd = new FormData();
  fd.append("file", state.songs[0]);
  try {
    const rec = await api("/api/analyze-song", { method: "POST", body: fd });
    if (!rec.voice_id) {
      box.innerHTML = `<b>未能自动匹配</b><br>${(rec.reasons || []).join("<br>")}`;
      return;
    }
    box.innerHTML = `<b>🎯 推荐「${rec.voice_name}」</b>
      <div class="smart-reasons">${rec.reasons.map((r) => `· ${r}`).join("<br>")}</div>
      <div class="smart-actions">
        <button class="btn primary btn-sm" id="btn-apply-rec">✓ 应用推荐并选中</button>
      </div>`;
    $("#btn-apply-rec").addEventListener("click", () => {
      state.sampleId = rec.voice_id;
      state.engine = rec.engine;
      loadSamples();
      loadEngines();
      updateGoButton();
      toast("已应用推荐设置，直接点「生成翻唱」即可");
      box.classList.add("hidden");
    });
  } catch (e) {
    box.innerHTML = `分析失败：${e.message}`;
  } finally {
    $("#btn-smart").disabled = false;
  }
});

function setSmartEnabled() {
  $("#btn-smart").disabled = !state.songs.length;
}

/* ---------------- 引擎与设置 ---------------- */

async function loadEngines() {
  try {
    const data = await api("/api/engines");
    renderOpts("#engine-list", data.engines, "engine", (e) =>
      e.available ? "" : e.reason);
    renderOpts("#separator-list", data.separators, "separator", (s) =>
      s.available ? "" : s.reason);
  } catch (e) {
    $("#engine-list").innerHTML = `<div class="loading">引擎检测失败：${e.message}</div>`;
  }
}

function renderOpts(sel, items, key, why) {
  const box = $(sel);
  box.innerHTML = "";
  const opts = [{ name: "auto", label: "自动（优先选最高质量）", kind: "auto",
    description: "自动使用当前环境下可用的最佳方案；若高质量组件不可用则回退，绝不阻塞。", available: true }]
    .concat(items);
  if (!opts.some((o) => o.name === state[key])) state[key] = "auto";
  for (const o of opts) {
    const reason = why(o);
    const div = document.createElement("div");
    div.className = "opt" + (o.name === state[key] ? " selected" : "") + (reason ? " disabled" : "");
    div.innerHTML = `<div class="top"><span></span>
      <span class="badge ${reason ? "off" : "ok"}">${reason ? "不可用" : "可用"}</span></div>
      <div class="desc">${o.description || ""}${reason ? `<br>⚠️ ${reason}` : ""}</div>`;
    div.querySelector(".top span:first-child").textContent = o.label;
    if (!reason) {
      div.addEventListener("click", () => { state[key] = o.name; renderOpts(sel, items, key, why); });
    } else {
      div.addEventListener("click", () => toast(`${o.label} 暂不可用：${reason}`, true));
    }
    box.appendChild(div);
  }
}

$("#vocal-gain").addEventListener("input", (e) => { $("#vg-val").textContent = `${e.target.value} dB`; });
$("#inst-gain").addEventListener("input", (e) => { $("#ig-val").textContent = `${e.target.value} dB`; });
$("#pitch").addEventListener("input", (e) => { $("#pt-val").textContent = `${e.target.value} 半音`; });
$("#openvoice-tau").addEventListener("input", (e) => { $("#tau-val").textContent = Number(e.target.value).toFixed(2); });

/* ---------------- 生成与轮询（批量逐个提交） ---------------- */

async function submitOne(songFile, title) {
  const fd = new FormData();
  fd.append("song", songFile);
  fd.append("voice_sample_id", state.sampleId);
  fd.append("engine", state.engine);
  fd.append("separator", state.separator);
  fd.append("vocal_gain_db", $("#vocal-gain").value);
  fd.append("instrumental_gain_db", $("#inst-gain").value);
  fd.append("pitch_shift_semitones", $("#pitch").value);
  fd.append("openvoice_tau", $("#openvoice-tau").value);
  fd.append("auto_balance", $("#auto-balance").checked);
  fd.append("denoise_gate", $("#denoise-gate").checked);
  fd.append("title", title);
  return api("/api/covers", { method: "POST", body: fd });
}

$("#btn-go").addEventListener("click", async () => {
  if (!state.sampleId || !state.songs.length) return;
  const base = $("#title").value.trim();
  $("#btn-go").disabled = true;
  const jobs = [];
  try {
    for (const [i, f] of state.songs.entries()) {
      // 批量时自定义标题作前缀；留空则由后端取文件名
      const title = base && state.songs.length > 1 ? `${base} ${i + 1}` : base;
      jobs.push(await submitOne(f, title));
    }
    state.songs = [];
    renderSongs();
    const first = jobs[0];
    state.activeId = first.id;
    $("#result-box").classList.add("hidden");
    $("#active-box").classList.remove("hidden");
    $("#active-title").textContent = jobs.length > 1 ? `${first.title} 等 ${jobs.length} 首` : first.title;
    $("#result-audio").pause();
    ["#stem-vocals", "#stem-inst", "#stem-conv"].forEach((s) => $(s).pause());
    toast(jobs.length > 1 ? `已提交 ${jobs.length} 个任务，开始生成…` : "任务已提交，开始生成…");
    startPolling(first.id);
    await loadJobs();
  } catch (e) {
    toast(`提交失败：${e.message}`, true);
    await loadJobs();
  } finally {
    updateGoButton();
  }
});

const STAGE_ORDER = ["预处理", "声音分析", "轨道分离", "声音转换", "混音", "导出", "完成"];

function startPolling(jid) {
  stopPolling();
  state.pollTimer = setInterval(async () => {
    try {
      const j = await api(`/api/covers/${jid}`);
      renderActive(j);
      if (j.status === "finished") {
        stopPolling();
        showResult(j);
        await loadJobs();
      } else if (j.status === "failed") {
        stopPolling();
        $("#active-status").textContent = "失败";
        $("#active-status").className = "badge off";
        $("#job-error").textContent = j.error || "未知错误";
        $("#job-error").classList.remove("hidden");
        await loadJobs();
      } else if (j.status === "cancelled") {
        stopPolling();
        $("#active-status").textContent = "已取消";
        $("#active-status").className = "badge off";
        await loadJobs();
      }
    } catch (_) { /* 网络抖动时继续轮询 */ }
  }, 1200);
}

$("#btn-cancel").addEventListener("click", async () => {
  const id = state.activeId;
  if (!id) return;
  if (!confirm("取消当前生成任务？")) return;
  try {
    await api(`/api/covers/${id}/cancel`, { method: "POST" });
    toast("已请求取消，将在当前阶段结束后停止");
  } catch (e) {
    toast(`取消失败：${e.message}`, true);
  }
});

function stopPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

function renderActive(j) {
  $("#active-title").textContent = j.title || j.id;
  $("#active-status").textContent = j.status === "processing" ? "生成中" : "排队中";
  $("#active-status").className = "badge ok";
  $("#btn-cancel").classList.toggle("hidden", false);
  $("#bar-fill").style.width = `${Math.round((j.progress || 0) * 100)}%`;
  $("#bar-fill").style.background = "";
  renderLiveness(j);
  const row = $("#stage-row");
  row.innerHTML = "";
  const curIdx = STAGE_ORDER.indexOf(j.stage);
  for (const s of STAGE_ORDER.slice(0, 6)) {
    const chip = document.createElement("span");
    const idx = STAGE_ORDER.indexOf(s);
    chip.className = "stage-chip" + (idx < curIdx ? " done" : idx === curIdx ? " doing" : "");
    chip.textContent = s;
    row.appendChild(chip);
  }
  $("#job-error").classList.add("hidden");
}

function renderLiveness(j) {
  const lv = $("#job-liveness");
  if (j.status !== "processing") { lv.classList.add("hidden"); return; }
  lv.classList.remove("hidden");
  if (j.worker_alive === false) {
    lv.className = "liveness dead";
    lv.innerHTML = '<span class="ldot"></span>任务线程已中断（服务可能重启过）—— 若长时间无变化，请删除后重新提交';
    return;
  }
  const age = j.updated_at ? Math.max(0, Math.round(Date.now() / 1000 - j.updated_at)) : 0;
  if (age > 300) {
    lv.className = "liveness slow";
    lv.innerHTML = `<span class="ldot"></span>算力运行中，但 ${Math.round(age / 60)} 分钟无新进度（大模型加载 / 长阶段重计算中）`;
  } else {
    lv.className = "liveness live";
    lv.innerHTML = `<span class="ldot"></span>推理进行中 · 心跳 ${age} 秒前`;
  }
}

function showResult(j) {
  state.activeId = j.id;
  $("#active-box").classList.add("hidden");
  $("#btn-cancel").classList.add("hidden");
  $("#result-box").classList.remove("hidden");
  $("#result-title").textContent = `《${j.title}》 · 用你的声音`;
  $("#result-audio").src = `/api/covers/${j.id}/result`;
  $("#btn-download").href = `/api/covers/${j.id}/result`;
  $("#stem-vocals").src = `/api/covers/${j.id}/stems/vocals`;
  $("#stem-inst").src = `/api/covers/${j.id}/stems/instrumental`;
  $("#stem-conv").src = `/api/covers/${j.id}/stems/converted`;
  const notes = $("#notes");
  notes.innerHTML = "";
  for (const n of j.notes || []) {
    const li = document.createElement("li");
    li.textContent = n;
    notes.appendChild(li);
  }
  $("#notes-box").classList.toggle("hidden", !(j.notes || []).length);
  $("#result-audio").play().catch(() => {});
}

$("#btn-again").addEventListener("click", () => {
  $("#result-box").classList.add("hidden");
  $("#active-box").classList.add("hidden");
  state.activeId = null;
  window.scrollTo({ top: 0, behavior: "smooth" });
});

$("#btn-delete-cover").addEventListener("click", async () => {
  const id = state.activeId;
  if (!id) return;
  const title = $("#result-title").textContent;
  if (!confirm(`删除「${title}」及其生成文件？`)) return;
  try {
    await api(`/api/covers/${id}`, { method: "DELETE" });
    ["result-audio", "stem-vocals", "stem-inst", "stem-conv"].forEach((k) => {
      const a = $("#" + k);
      a.pause();
      a.removeAttribute("src");
      a.load();
    });
    $("#result-box").classList.add("hidden");
    state.activeId = null;
    toast("已删除");
    await loadJobs();
  } catch (err) {
    toast(`删除失败：${err.message}`, true);
  }
});

async function loadJobs() {
  const ul = $("#job-list");
  try {
    const jobs = await api("/api/covers");
    ul.innerHTML = "";
    for (const j of jobs.slice(0, 12)) {
      const li = document.createElement("li");
      const cls = j.status === "finished" ? "ok" : j.status === "failed" ? "err" : "run";
      const txt = j.status === "finished" ? "✓ 完成" : j.status === "failed" ? "✗ 失败"
        : j.status === "cancelled" ? "⊘ 已取消"
        : j.status === "processing" ? `⟳ ${(j.progress * 100 || 0) | 0}%` : "… 排队";
      li.innerHTML = `<span class="t"></span><span class="st ${cls}"></span><span class="st">${fmtTime(j.created_at)}</span>
        <button class="del-job" title="删除">✕</button>`;
      li.querySelector(".t").textContent = j.title;
      li.querySelector(".st").textContent = txt;
      li.querySelector(".del-job").addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm(`删除「${j.title}」及其生成文件？`)) return;
        try {
          await api(`/api/covers/${j.id}`, { method: "DELETE" });
          // 正在看的这条被删了 → 收起结果区
          if (state.activeId === j.id) {
            stopPolling();
            $("#result-box").classList.add("hidden");
            $("#active-box").classList.add("hidden");
            state.activeId = null;
          }
          toast("已删除");
          await loadJobs();
        } catch (err) {
          toast(`删除失败：${err.message}`, true);
        }
      });
      li.addEventListener("click", () => {
        if (j.status === "finished") {
          state.activeId = j.id;
          $("#active-box").classList.add("hidden");
          showResult(j);
        } else if (j.status === "failed") {
          $("#active-box").classList.remove("hidden");
          $("#active-title").textContent = j.title;
          $("#active-status").textContent = "失败";
          $("#active-status").className = "badge off";
          $("#bar-fill").style.width = "100%";
          $("#bar-fill").style.background = "var(--err)";
          $("#job-error").textContent = j.error || "未知错误";
          $("#job-error").classList.remove("hidden");
        } else {
          state.activeId = j.id;
          $("#active-box").classList.remove("hidden");
          startPolling(j.id);
        }
      });
      ul.appendChild(li);
    }
    if (!jobs.length) ul.innerHTML = `<div class="loading">还没有作品，去生成第一首吧 🎤</div>`;
    // 有进行中的任务且当前没有在轮询 → 恢复轮询（只在首屏加载时静默接管；
    // 生成流程中的 loadJobs 已有轮询器，不会走到这里）
    const running = jobs.find((j) => j.status === "queued" || j.status === "processing");
    if (running && !state.pollTimer) {
      $("#active-box").classList.remove("hidden");
      $("#active-title").textContent = running.title;
      startPolling(running.id);
    }
  } catch (e) {
    ul.innerHTML = `<div class="loading">任务加载失败：${e.message}</div>`;
  }
}

/* ---------------- 启动 ---------------- */

loadSamples();
loadEngines();
loadJobs();
updateGoButton();
