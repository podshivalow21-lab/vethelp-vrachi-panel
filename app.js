/**
 * Помощник врача · Ветхэлп
 * Вход по паролю → запись (MediaRecorder) → локальный Whisper в браузере → текст → черновик → копирование.
 * Без SpeechKit, без Cloud Function, без API-ключей. Модель скачивается с Hugging Face один раз и кэшируется.
 */
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";

/* Модель: multilingual whisper-base (лучше русский). ~75–80 МБ q8, кэш в IndexedDB. */
var WHISPER_MODEL = "Xenova/whisper-base";
env.allowLocalModels = false;
env.useBrowserCache = true;

// Смените перед выкладкой. Клиентская проверка — базовая защита, не fort knox.
var PANEL_PASSWORD = "vethelp-vrachi";
var AUTH_KEY = "vethelp-panel-auth";

var SECTIONS = [
  { id: "complaints", label: "Жалобы / анамнез" },
  { id: "exam", label: "Осмотр" },
  { id: "diagnosis", label: "Диагноз (рабочий)" },
  { id: "treatment", label: "Назначения / лечение" },
  { id: "advice", label: "Рекомендации владельцу" },
  { id: "other", label: "Прочее / заметки" },
];

var MARKERS = [
  { re: /жалоб/i, id: "complaints" },
  { re: /анамнез/i, id: "complaints" },
  { re: /осмотр/i, id: "exam" },
  { re: /диагноз/i, id: "diagnosis" },
  { re: /назнач/i, id: "treatment" },
  { re: /лечен/i, id: "treatment" },
  { re: /рекоменд/i, id: "advice" },
  { re: /план/i, id: "advice" },
  { re: /прочее/i, id: "other" },
  { re: /заметк/i, id: "other" },
];

var els = {
  gate: document.getElementById("gate"),
  panel: document.getElementById("panel"),
  loginForm: document.getElementById("login-form"),
  password: document.getElementById("password"),
  loginError: document.getElementById("login-error"),
  btnRecord: document.getElementById("btn-record"),
  btnRecordLabel: document.getElementById("btn-record-label"),
  recordStatus: document.getElementById("record-status"),
  recordStatusText: document.getElementById("record-status-text"),
  speechFallback: document.getElementById("speech-fallback"),
  visitText: document.getElementById("visit-text"),
  btnDraft: document.getElementById("btn-draft"),
  btnCopyAll: document.getElementById("btn-copy-all"),
  installCard: document.getElementById("install-card"),
  btnInstall: document.getElementById("btn-install"),
  installManual: document.getElementById("install-manual"),
};

var mediaStream = null;
var mediaRecorder = null;
var recordedChunks = [];
var chosenMime = "";
var isRecording = false;
var isTranscribing = false;
var panelReady = false;

/* PWA install: hide «Установить» forever after first install on this device/browser */
var PWA_INSTALLED_KEY = "vethelp-panel-pwa-installed";
/** @type {null | Event} */
var deferredInstallPrompt = null;
var installUiReady = false;

/** @type {null | ((audio: Float32Array, opts?: object) => Promise<{text?: string}>)} */
var whisperPipeline = null;
var whisperLoadPromise = null;
var whisperDevice = "";

/* ---------- Auth ---------- */
function isAuthed() {
  try {
    return sessionStorage.getItem(AUTH_KEY) === "1";
  } catch (e) {
    return false;
  }
}

function setAuthed() {
  try {
    sessionStorage.setItem(AUTH_KEY, "1");
  } catch (e) {
    /* private mode — ок, просто спросит снова */
  }
}

function showPanel() {
  els.gate.hidden = true;
  els.panel.hidden = false;
  if (!panelReady) {
    panelReady = true;
    setupSpeech();
    wirePanel();
    setupPwaInstall();
  } else {
    refreshInstallUi();
  }
}

function showGate() {
  els.gate.hidden = false;
  els.panel.hidden = true;
  if (els.password) {
    setTimeout(function () {
      els.password.focus();
    }, 50);
  }
}

function onLogin(event) {
  if (event) event.preventDefault();
  var value = (els.password.value || "").trim();
  if (value === PANEL_PASSWORD) {
    els.loginError.hidden = true;
    els.loginError.textContent = "";
    els.password.value = "";
    setAuthed();
    showPanel();
    return;
  }
  els.loginError.hidden = false;
  els.loginError.textContent = "Неверный пароль";
  els.password.select();
}

/* ---------- Clipboard ---------- */
function copyText(text, btn) {
  text = text || "";
  function flash() {
    if (!btn) return;
    var prev = btn.textContent;
    btn.textContent = "Скопировано";
    btn.classList.add("copied");
    setTimeout(function () {
      btn.textContent = prev;
      btn.classList.remove("copied");
    }, 1400);
  }

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(flash).catch(function () {
      fallbackCopy(text, flash);
    });
  } else {
    fallbackCopy(text, flash);
  }
}

function fallbackCopy(text, onOk) {
  var ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.left = "-9999px";
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand("copy");
    if (onOk) onOk();
  } catch (e) {
    /* ignore */
  }
  document.body.removeChild(ta);
}

/* ---------- Draft heuristics ---------- */
function findMarkers(text) {
  var hits = [];
  for (var i = 0; i < MARKERS.length; i++) {
    var m = MARKERS[i];
    var match = m.re.exec(text);
    if (match) {
      hits.push({ index: match.index, id: m.id, len: match[0].length });
    }
  }
  hits.sort(function (a, b) {
    return a.index - b.index;
  });
  var seen = {};
  var unique = [];
  for (var j = 0; j < hits.length; j++) {
    if (!seen[hits[j].id]) {
      seen[hits[j].id] = true;
      unique.push(hits[j]);
    }
  }
  unique.sort(function (a, b) {
    return a.index - b.index;
  });
  return unique;
}

function splitIntoDraft(text) {
  var result = {};
  SECTIONS.forEach(function (s) {
    result[s.id] = "";
  });

  text = (text || "").trim();
  if (!text) return result;

  var hits = findMarkers(text);
  if (hits.length === 0) {
    result.complaints = text;
    return result;
  }

  if (hits[0].index > 0) {
    var preamble = text.slice(0, hits[0].index).trim();
    if (preamble) {
      result.complaints = preamble;
    }
  }

  for (var i = 0; i < hits.length; i++) {
    var start = hits[i].index + hits[i].len;
    while (start < text.length && /[\s:.\-–—]/.test(text.charAt(start))) {
      start++;
    }
    var end = i + 1 < hits.length ? hits[i + 1].index : text.length;
    var chunk = text.slice(start, end).trim();
    if (!chunk) continue;
    if (result[hits[i].id]) {
      result[hits[i].id] += "\n" + chunk;
    } else {
      result[hits[i].id] = chunk;
    }
  }

  return result;
}

function applyDraft(draft) {
  SECTIONS.forEach(function (s) {
    var ta = document.getElementById("field-" + s.id);
    if (ta) ta.value = draft[s.id] || "";
  });
}

function buildAllText() {
  var parts = [];
  SECTIONS.forEach(function (s) {
    var ta = document.getElementById("field-" + s.id);
    var val = ta ? ta.value.trim() : "";
    if (val) {
      parts.push(s.label + ":\n" + val);
    }
  });
  return parts.join("\n\n");
}

/* ---------- UI helpers ---------- */
function showFallback(msg) {
  els.speechFallback.textContent = msg;
  els.speechFallback.classList.add("visible");
}

function hideFallback() {
  els.speechFallback.classList.remove("visible");
  els.speechFallback.textContent = "";
}

function setStatus(mode, text) {
  /* mode: '', 'recording', 'transcribing', 'loading' */
  els.recordStatus.classList.remove(
    "visible",
    "is-recording",
    "is-transcribing",
    "is-loading"
  );
  if (!mode) {
    if (els.recordStatusText) els.recordStatusText.textContent = "";
    return;
  }
  var cls =
    mode === "recording"
      ? "is-recording"
      : mode === "loading"
        ? "is-loading"
        : "is-transcribing";
  els.recordStatus.classList.add("visible", cls);
  if (els.recordStatusText) {
    els.recordStatusText.textContent =
      text ||
      (mode === "recording"
        ? "Идёт запись…"
        : mode === "loading"
          ? "Загружаю модель…"
          : "Расшифровываю…");
  }
}

function setRecordButtonIdle() {
  els.btnRecord.classList.remove("is-recording", "is-busy");
  els.btnRecord.setAttribute("aria-pressed", "false");
  els.btnRecord.disabled = false;
  els.btnRecordLabel.textContent = "Старт записи";
}

/* ---------- Capability checks ---------- */
function hasWebAssembly() {
  try {
    if (typeof WebAssembly === "undefined") return false;
    if (typeof WebAssembly.Module !== "function") return false;
    var bytes = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]);
    return WebAssembly.validate(bytes);
  } catch (e) {
    return false;
  }
}

function hasWebGpu() {
  return !!(navigator.gpu && typeof navigator.gpu.requestAdapter === "function");
}

/* ---------- Speech (MediaRecorder → локальный Whisper) ---------- */
function pickMimeType() {
  if (typeof MediaRecorder === "undefined") return "";
  var candidates = [
    "audio/webm;codecs=opus",
    "audio/webm; codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg; codecs=opus",
    "audio/ogg",
  ];
  for (var i = 0; i < candidates.length; i++) {
    if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(candidates[i])) {
      return candidates[i];
    }
  }
  return "";
}

function setupSpeech() {
  if (!hasWebAssembly()) {
    showFallback(
      "В этом браузере нет WebAssembly — локальная расшифровка недоступна. Откройте панель в Яндекс.Браузере (актуальная версия) или вставьте текст вручную."
    );
    els.btnRecord.disabled = true;
    els.btnRecordLabel.textContent = "Расшифровка недоступна";
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showFallback(
      "Браузер не даёт доступ к микрофону. Откройте панель в Яндекс.Браузере по HTTPS (или localhost) и разрешите микрофон, либо вставьте текст вручную."
    );
    els.btnRecord.disabled = true;
    els.btnRecordLabel.textContent = "Микрофон недоступен";
    return;
  }
  if (typeof MediaRecorder === "undefined") {
    showFallback(
      "MediaRecorder не поддерживается в этом браузере. Откройте панель в Яндекс.Браузере или вставьте текст вручную."
    );
    els.btnRecord.disabled = true;
    els.btnRecordLabel.textContent = "Микрофон недоступен";
    return;
  }
  chosenMime = pickMimeType();
}

function stopMediaTracks() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(function (t) {
      try {
        t.stop();
      } catch (e) {
        /* ignore */
      }
    });
    mediaStream = null;
  }
}

function startRecording() {
  if (isRecording || isTranscribing) return;

  hideFallback();
  recordedChunks = [];

  navigator.mediaDevices
    .getUserMedia({ audio: true })
    .then(function (stream) {
      mediaStream = stream;
      var options = {};
      if (chosenMime) options.mimeType = chosenMime;

      try {
        mediaRecorder = new MediaRecorder(stream, options);
      } catch (e) {
        try {
          mediaRecorder = new MediaRecorder(stream);
        } catch (e2) {
          stopMediaTracks();
          showFallback("Не удалось начать запись. Проверьте микрофон или вставьте текст вручную.");
          return;
        }
      }

      chosenMime = mediaRecorder.mimeType || chosenMime || "audio/webm";

      mediaRecorder.ondataavailable = function (ev) {
        if (ev.data && ev.data.size > 0) {
          recordedChunks.push(ev.data);
        }
      };

      mediaRecorder.onerror = function () {
        showFallback("Ошибка записи с микрофона. Попробуйте ещё раз или вставьте текст вручную.");
        finishRecordingUi();
        stopMediaTracks();
      };

      mediaRecorder.onstop = function () {
        stopMediaTracks();
        var blob = new Blob(recordedChunks, { type: chosenMime || "audio/webm" });
        recordedChunks = [];
        if (!blob.size) {
          setRecordButtonIdle();
          setStatus("");
          isTranscribing = false;
          showFallback("Пустая запись — речь не услышана. Говорите ближе к микрофону.");
          return;
        }
        transcribeBlob(blob);
      };

      mediaRecorder.start(250);
      isRecording = true;
      els.btnRecord.classList.add("is-recording");
      els.btnRecord.setAttribute("aria-pressed", "true");
      els.btnRecordLabel.textContent = "Стоп";
      setStatus("recording", "Идёт запись…");
    })
    .catch(function (err) {
      var name = (err && err.name) || "";
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        showFallback(
          "Нет доступа к микрофону. Разрешите микрофон в браузере (замок у адреса) или вставьте текст вручную."
        );
      } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
        showFallback("Микрофон не найден. Проверьте устройство в настройках Windows/браузера.");
      } else if (name === "NotReadableError" || name === "TrackStartError") {
        showFallback("Микрофон занят другим приложением. Закройте его и попробуйте снова.");
      } else {
        showFallback(
          "Не удалось открыть микрофон. Проверьте разрешения браузера или вставьте текст вручную."
        );
      }
    });
}

function finishRecordingUi() {
  isRecording = false;
  els.btnRecord.classList.remove("is-recording");
  els.btnRecord.setAttribute("aria-pressed", "false");
}

function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  finishRecordingUi();
  els.btnRecord.disabled = true;
  els.btnRecordLabel.textContent = "Стоп";
  try {
    if (mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    } else {
      stopMediaTracks();
      setRecordButtonIdle();
      setStatus("");
    }
  } catch (e) {
    stopMediaTracks();
    setRecordButtonIdle();
    setStatus("");
  }
}

/**
 * Отсекает multilingual gibberish (CJK/Hangul/латиница без кириллицы) —
 * типичная галлюцинация tiny/base на короткой русской речи.
 */
function looksLikeRussianGarbage(text) {
  var t = (text || "").trim();
  if (!t) return false;

  /* Known Whisper subtitle-credit hallucinations (all-Cyrillic — old heuristic missed them) */
  if (/редактор\s+субтитров/i.test(t)) return true;
  if (/корректор\s+а\./i.test(t)) return true;
  if (/синецк/i.test(t) && /ег[оа]ров/i.test(t)) return true;
  if (/субтитров\s+а\.?\s*синецк/i.test(t)) return true;

  /* Heavy exact-phrase repetition: same 20+ char chunk appears twice */
  var norm = t.replace(/\s+/g, " ");
  if (norm.length >= 40) {
    for (var i = 0; i + 40 <= norm.length; i++) {
      var chunk = norm.slice(i, i + 20);
      if (norm.indexOf(chunk, i + 20) !== -1) return true;
    }
  }

  var cyr = (t.match(/[\u0400-\u04FF]/g) || []).join("").length;
  var cjk = (t.match(/[\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]/g) || []).join("").length;
  var latin = (t.match(/[A-Za-z]/g) || []).join("").length;
  var letters = cyr + cjk + latin;
  if (cjk >= 2) return true;
  if (letters < 6) {
    return cyr === 0 && (cjk > 0 || latin >= 3);
  }
  if (cyr === 0 && letters >= 6) return true;
  var cyrRatio = cyr / letters;
  if (cyrRatio < 0.35 && latin > cyr) return true;
  return false;
}

function appendTranscript(text) {
  text = (text || "").trim();
  if (!text) return;
  if (looksLikeRussianGarbage(text)) return;
  var cur = (els.visitText.value || "").replace(/\s+$/, "");
  if (cur) {
    els.visitText.value = cur + " " + text;
  } else {
    els.visitText.value = text;
  }
}

/* ---------- Audio → Float32 16 kHz mono (для Whisper) ---------- */
function downsample(buffer, fromRate, toRate) {
  if (fromRate === toRate) return buffer;
  var ratio = fromRate / toRate;
  var newLen = Math.round(buffer.length / ratio);
  var result = new Float32Array(newLen);
  for (var i = 0; i < newLen; i++) {
    var start = Math.floor(i * ratio);
    var end = Math.floor((i + 1) * ratio);
    var sum = 0;
    var count = 0;
    for (var j = start; j < end && j < buffer.length; j++) {
      sum += buffer[j];
      count++;
    }
    result[i] = count ? sum / count : 0;
  }
  return result;
}

function audioBufferToMono16k(audioBuffer) {
  var srcRate = audioBuffer.sampleRate;
  var src = audioBuffer.getChannelData(0);
  if (audioBuffer.numberOfChannels > 1) {
    var ch1 = audioBuffer.getChannelData(1);
    var mixed = new Float32Array(src.length);
    for (var i = 0; i < src.length; i++) {
      mixed[i] = (src[i] + ch1[i]) * 0.5;
    }
    src = mixed;
  }
  return downsample(src, srcRate, 16000);
}

function blobToFloat32_16k(blob) {
  return blob.arrayBuffer().then(function (buf) {
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) {
      return Promise.reject(Object.assign(new Error("no-audiocontext"), { code: "no-audiocontext" }));
    }
    /* Xenova/Whisper: декодировать в 16 kHz — дефолт 44100 даёт «кашу» в модели. */
    var ctx;
    try {
      ctx = new Ctx({ sampleRate: 16000 });
    } catch (e) {
      ctx = new Ctx();
    }
    return ctx.decodeAudioData(buf.slice(0)).then(
      function (audioBuffer) {
        /* Даже если браузер проигнорировал sampleRate — downsample до 16 kHz. */
        var samples = audioBufferToMono16k(audioBuffer);
        try {
          ctx.close();
        } catch (e) {
          /* ignore */
        }
        return samples;
      },
      function () {
        try {
          ctx.close();
        } catch (e) {
          /* ignore */
        }
        return Promise.reject(Object.assign(new Error("decode-failed"), { code: "decode-failed" }));
      }
    );
  });
}

/* ---------- Whisper pipeline (ленивая загрузка) ---------- */
function ensureWhisper(onProgress) {
  if (whisperPipeline) {
    return Promise.resolve(whisperPipeline);
  }
  if (whisperLoadPromise) {
    return whisperLoadPromise;
  }

  if (!hasWebAssembly()) {
    return Promise.reject(
      Object.assign(new Error("no-wasm"), { code: "no-wasm" })
    );
  }

  whisperLoadPromise = (async function () {
    var progressHandler = function (ev) {
      if (!onProgress || !ev) return;
      if (ev.status === "progress" && typeof ev.progress === "number") {
        var pct = Math.max(0, Math.min(100, Math.round(ev.progress)));
        onProgress("Загружаю модель… " + pct + "%");
      } else if (ev.status === "download" || ev.status === "initiate") {
        onProgress("Загружаю модель…");
      } else if (ev.status === "done" || ev.status === "ready") {
        onProgress("Модель готова…");
      }
    };

    /* webgpu+q8 даёт multilingual gibberish (известный баг) — не используем.
       WebGPU: encoder fp32 + decoder q4; WASM: q8 (стабильнее для клиники). */
    var tryConfigs = [];
    if (hasWebGpu()) {
      tryConfigs.push({
        device: "webgpu",
        dtype: { encoder_model: "fp32", decoder_model_merged: "q4" },
        label: "Загружаю модель (WebGPU)…",
      });
      tryConfigs.push({
        device: "webgpu",
        dtype: "fp32",
        label: "Загружаю модель (WebGPU)…",
      });
    }
    tryConfigs.push({
      device: "wasm",
      dtype: "q8",
      label: "Загружаю модель…",
    });

    var lastErr = null;
    for (var d = 0; d < tryConfigs.length; d++) {
      var cfg = tryConfigs[d];
      try {
        if (onProgress) onProgress(cfg.label);
        var pipe = await pipeline("automatic-speech-recognition", WHISPER_MODEL, {
          device: cfg.device,
          dtype: cfg.dtype,
          progress_callback: progressHandler,
        });
        whisperPipeline = pipe;
        whisperDevice = cfg.device;
        return pipe;
      } catch (err) {
        lastErr = err;
        /* пробуем следующий device/dtype */
      }
    }

    whisperLoadPromise = null;
    var msg = (lastErr && lastErr.message) || String(lastErr || "");
    if (/fetch|network|Failed to fetch|Load failed|CORS/i.test(msg)) {
      throw Object.assign(new Error("model-network"), { code: "model-network", cause: lastErr });
    }
    if (/WebGPU|gpu|adapter/i.test(msg) && !hasWebAssembly()) {
      throw Object.assign(new Error("no-runtime"), { code: "no-runtime", cause: lastErr });
    }
    throw Object.assign(new Error("model-load"), {
      code: "model-load",
      cause: lastErr,
      message: msg || "model-load",
    });
  })();

  return whisperLoadPromise;
}

function transcribeBlob(blob) {
  isTranscribing = true;
  els.btnRecord.disabled = true;
  els.btnRecord.classList.add("is-busy");
  els.btnRecordLabel.textContent = "Старт записи";
  setStatus("loading", "Загружаю модель…");
  hideFallback();

  ensureWhisper(function (statusText) {
    setStatus("loading", statusText);
  })
    .then(function (pipe) {
      setStatus("transcribing", "Расшифровываю…");
      return blobToFloat32_16k(blob).then(function (samples) {
        if (!samples || !samples.length) {
          return { text: "" };
        }
        /* Float32Array @ 16 kHz; language "ru" + task transcribe (transformers.js 3.5). */
        return pipe(samples, {
          language: "ru",
          task: "transcribe",
          return_timestamps: false,
          condition_on_prev_tokens: false,
          compression_ratio_threshold: 2.4,
          logprob_threshold: -1.0,
          no_speech_threshold: 0.6,
        });
      });
    })
    .then(function (result) {
      var text = "";
      if (result && typeof result.text === "string") {
        text = result.text;
      } else if (typeof result === "string") {
        text = result;
      }
      text = (text || "").trim();
      if (!text) {
        showFallback(
          "Речь не распознана. Говорите чуть громче и ближе к микрофону, затем снова нажмите «Старт записи»."
        );
        return;
      }
      if (looksLikeRussianGarbage(text)) {
        /* Не дописываем gibberish — textarea остаётся как была. */
        showFallback(
          "Не удалось нормально распознать речь. Подойдите ближе к микрофону, говорите чётче и нажмите «Старт записи» ещё раз."
        );
        return;
      }
      appendTranscript(text);
    })
    .catch(function (err) {
      var code = (err && err.code) || "";
      var msg = (err && err.message) || "";
      if (code === "no-wasm" || code === "no-runtime") {
        showFallback(
          "Браузер не умеет запускать модель (нет WebAssembly/WebGPU). Откройте панель в Яндекс.Браузере актуальной версии или вставьте текст вручную."
        );
      } else if (code === "model-network") {
        showFallback(
          "Не удалось скачать модель Whisper. Нужен интернет при первом запуске (потом работает из кэша). Проверьте сеть и откройте снова в Яндекс.Браузере."
        );
      } else if (code === "model-load") {
        showFallback(
          "Не удалось загрузить модель расшифровки. Обновите Яндекс.Браузер или вставьте текст вручную." +
            (msg && msg !== "model-load" ? " (" + msg.slice(0, 120) + ")" : "")
        );
      } else if (code === "decode-failed" || code === "no-audiocontext" || msg === "decode-failed") {
        showFallback(
          "Не удалось прочитать запись. Откройте панель в Яндекс.Браузере или вставьте текст вручную."
        );
      } else {
        showFallback(
          "Не удалось расшифровать запись. Попробуйте ещё раз или вставьте текст вручную."
        );
      }
    })
    .then(function () {
      isTranscribing = false;
      setRecordButtonIdle();
      setStatus("");
    });
}

function toggleRecording() {
  if (isTranscribing) return;
  if (isRecording) stopRecording();
  else startRecording();
}

/* ---------- PWA install («Установить на рабочий стол» ≠ «Старт записи») ---------- */
function isRunningStandalone() {
  try {
    if (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches) {
      return true;
    }
    if (window.matchMedia && window.matchMedia("(display-mode: window-controls-overlay)").matches) {
      return true;
    }
  } catch (e) {
    /* ignore */
  }
  if (typeof navigator !== "undefined" && navigator.standalone === true) {
    return true;
  }
  return false;
}

function wasInstalledFlag() {
  try {
    return localStorage.getItem(PWA_INSTALLED_KEY) === "1";
  } catch (e) {
    return false;
  }
}

function markInstalled() {
  try {
    localStorage.setItem(PWA_INSTALLED_KEY, "1");
  } catch (e) {
    /* private mode — hide for this session anyway */
  }
  deferredInstallPrompt = null;
  hideInstallUi();
}

function hideInstallUi() {
  if (els.installCard) els.installCard.hidden = true;
  if (els.btnInstall) els.btnInstall.hidden = true;
  if (els.installManual) els.installManual.hidden = true;
}

function shouldHideInstallForever() {
  return isRunningStandalone() || wasInstalledFlag();
}

function showInstallButtonAlways() {
  if (!els.installCard || !els.btnInstall) return;
  els.installCard.hidden = false;
  els.btnInstall.hidden = false;
  els.btnInstall.disabled = false;
}

function refreshInstallUi() {
  if (shouldHideInstallForever()) {
    hideInstallUi();
    return;
  }
  showInstallButtonAlways();
  /* Manual steps stay hidden until click (when no native prompt) or after dismiss */
  if (els.installManual && deferredInstallPrompt) {
    els.installManual.hidden = true;
  }
}

function revealManualInstallSteps() {
  if (!els.installManual) return;
  els.installManual.hidden = false;
  try {
    els.installManual.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) {
    /* ignore */
  }
}

function onInstallClick() {
  var ev = deferredInstallPrompt;
  if (!ev) {
    /* No beforeinstallprompt — still useful: show Яндекс/Chrome menu steps */
    showInstallButtonAlways();
    revealManualInstallSteps();
    return;
  }
  deferredInstallPrompt = null;
  if (els.btnInstall) els.btnInstall.disabled = true;
  Promise.resolve(ev.prompt())
    .then(function () {
      return ev.userChoice;
    })
    .then(function (choice) {
      if (choice && choice.outcome === "accepted") {
        markInstalled();
      } else {
        /* Dismissed — keep primary button + show manual fallback */
        showInstallButtonAlways();
        revealManualInstallSteps();
      }
    })
    .catch(function () {
      showInstallButtonAlways();
      revealManualInstallSteps();
    })
    .then(function () {
      if (els.btnInstall && !shouldHideInstallForever()) {
        els.btnInstall.disabled = false;
      }
    });
}

function setupPwaInstall() {
  if (installUiReady) {
    refreshInstallUi();
    return;
  }
  installUiReady = true;

  if (els.btnInstall) {
    els.btnInstall.addEventListener("click", onInstallClick);
  }

  if (shouldHideInstallForever()) {
    hideInstallUi();
    return;
  }

  /* Always show install card + primary button after login (do not wait for beforeinstallprompt) */
  showInstallButtonAlways();
  if (els.installManual) els.installManual.hidden = true;
}

var PANEL_ASSET_VERSION = "3";
var PANEL_CACHE_PREFIX = "vethelp-panel-";

function purgeStalePanelCaches() {
  if (!("caches" in window)) return Promise.resolve();
  return caches.keys().then(function (keys) {
    return Promise.all(
      keys.map(function (key) {
        if (key.indexOf(PANEL_CACHE_PREFIX) === 0 && key !== PANEL_CACHE_PREFIX + "v" + PANEL_ASSET_VERSION) {
          return caches.delete(key);
        }
      })
    );
  }).catch(function () {
    /* ignore */
  });
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  window.addEventListener("load", function () {
    /* One-time: drop old vethelp-panel-* caches when page asset version bumped */
    try {
      var seen = localStorage.getItem("vethelp-panel-asset-v");
      if (seen !== PANEL_ASSET_VERSION) {
        purgeStalePanelCaches().then(function () {
          try {
            localStorage.setItem("vethelp-panel-asset-v", PANEL_ASSET_VERSION);
          } catch (e) {
            /* ignore */
          }
        });
      }
    } catch (e) {
      purgeStalePanelCaches();
    }

    navigator.serviceWorker
      .register("./sw.js")
      .then(function (registration) {
        /* Force check for updated SW (fixes stale cache-first SW on doctors' machines) */
        if (registration && typeof registration.update === "function") {
          registration.update().catch(function () {});
        }
        if (registration && registration.waiting) {
          try {
            registration.waiting.postMessage({ type: "SKIP_WAITING" });
          } catch (e) {
            /* ignore */
          }
        }
      })
      .catch(function () {
        /* SW optional for local file://; needed on HTTPS for installability */
      });
  });
}

/* ---------- Wire panel ---------- */
function onBuildDraft() {
  var text = (els.visitText.value || "").trim();
  if (!text) {
    els.visitText.classList.add("highlight");
    els.visitText.focus();
    setTimeout(function () {
      els.visitText.classList.remove("highlight");
    }, 1600);
    return;
  }
  applyDraft(splitIntoDraft(text));
}

function wirePanel() {
  els.btnRecord.addEventListener("click", toggleRecording);
  els.btnDraft.addEventListener("click", onBuildDraft);
  els.btnCopyAll.addEventListener("click", function () {
    copyText(buildAllText(), els.btnCopyAll);
  });

  SECTIONS.forEach(function (s) {
    var btn = document.getElementById("copy-" + s.id);
    var ta = document.getElementById("field-" + s.id);
    if (btn && ta) {
      btn.addEventListener("click", function () {
        copyText(ta.value, btn);
      });
    }
  });
}

function init() {
  registerServiceWorker();
  /* Early: if already installed PWA, never flash the install card after login */
  if (shouldHideInstallForever()) {
    hideInstallUi();
  }
  window.addEventListener("beforeinstallprompt", function (e) {
    /* Capture even before login so prompt is not lost */
    e.preventDefault();
    deferredInstallPrompt = e;
    if (panelReady && !shouldHideInstallForever()) {
      refreshInstallUi();
    }
  });
  window.addEventListener("appinstalled", function () {
    markInstalled();
  });
  els.loginForm.addEventListener("submit", onLogin);
  if (isAuthed()) {
    showPanel();
  } else {
    showGate();
  }
}

init();
