import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.webgpu.mjs";
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";
import { IrodoriTTS } from "https://cdn.jsdelivr.net/gh/ngc-shj/irodori-tts-webgpu@aa3b6390018bb09a2e461c95d1f55992c06e197d/runtime/pipeline.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "https://cdn.jsdelivr.net/gh/ngc-shj/irodori-tts-webgpu@aa3b6390018bb09a2e461c95d1f55992c06e197d/tokenizer/";

const MODEL_REPO = "https://huggingface.co/noguchis/irodori-tts-onnx/resolve/main";
const MODEL_BASE = `${MODEL_REPO}/onnx_fp16`;
const CACHE_NAME = "internal-irodori-models-v05";
const MODELS = {
  text: "text_encoder", speaker: "speaker_encoder", duration: "duration",
  dit: "dit", dac: "dacvae_decoder", enc: "dacvae_encoder",
};
const LOW_MEMORY_MODE = /iPad|iPhone|iPod/.test(navigator.userAgent)
  || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
// Load the largest model first on iPhone/iPad. This avoids keeping three
// already-created sessions alive while Safari downloads the 701 MB DiT file.
const LOW_MEMORY_LOAD_ORDER = [
  ["dit", "dit"], ["dac", "dacvae_decoder"], ["text", "text_encoder"],
  ["speaker", "speaker_encoder"], ["enc", "dacvae_encoder"], ["duration", "duration"],
];

const $ = (id) => document.getElementById(id);
const els = {
  text: $("text"), ref: $("ref"), run: $("run"), out: $("out"), dl: $("download"),
  result: $("result"), resultStats: $("resultStats"), log: $("log"), fileTitle: $("fileTitle"),
  charCount: $("charCount"), badge: $("deviceBadge"), gpuSupport: $("gpuSupport"), gpuName: $("gpuName"),
  browserName: $("browserName"), progressCard: $("progressCard"), progressTitle: $("progressTitle"),
  progressText: $("progressText"), progressPercent: $("progressPercent"), bar: $("barInner"),
  voiceSelect: $("voiceSelect"), voiceStatus: $("voiceStatus"),
};

let tts = null;
let tokenizer = null;
let webgpuReady = false;
let objectUrl = null;
let voiceConfig = [];
let gpuAdapter = null;

function log(message) {
  const now = new Date().toLocaleTimeString("ja-JP", { hour12:false });
  els.log.textContent += `\n[${now}] ${message}`;
  els.log.scrollTop = els.log.scrollHeight;
}
function setProgress(percent, title, text) {
  els.progressCard.hidden = false;
  els.progressPercent.textContent = `${Math.round(percent)}%`;
  els.bar.style.width = `${Math.max(2, Math.min(100, percent))}%`;
  els.progressTitle.textContent = title;
  els.progressText.textContent = text;
}
function browserLabel() {
  const ua = navigator.userAgent;
  if (/Edg\//.test(ua)) return "Microsoft Edge";
  if (/Chrome\//.test(ua)) return "Google Chrome";
  if (/Safari\//.test(ua) && !/Chrome\//.test(ua)) return "Safari";
  return navigator.userAgentData?.brands?.map(b=>b.brand).join(" / ") || "不明";
}
async function checkDevice() {
  els.browserName.textContent = browserLabel();
  if (!navigator.gpu) {
    els.badge.textContent = "この端末は非対応";
    els.badge.className = "badge bad";
    els.gpuSupport.textContent = "非対応";
    els.gpuName.textContent = "WebGPUなし";
    log("WebGPUが利用できません。Chrome最新版などで再確認してください。");
    return;
  }
  try {
    gpuAdapter = await navigator.gpu.requestAdapter();
    if (!gpuAdapter) throw new Error("GPU adapterを取得できませんでした");
    let label = "WebGPU対応GPU";
    try {
      const info = gpuAdapter.info || {};
      label = [info.vendor, info.architecture, info.device].filter(Boolean).join(" / ") || label;
    } catch (detailError) {
      log(`GPU詳細情報を取得できませんでした: ${detailError.message || detailError}`);
    }
    els.gpuSupport.textContent = "対応";
    els.gpuName.textContent = label;
    els.badge.textContent = "この端末で生成できます";
    els.badge.className = "badge ok";
    els.badge.title = "";
    webgpuReady = true;
    updateRunState();
    log(`WebGPU OK: ${label}`);
    try {
      const maxBuffer = Number(gpuAdapter.limits?.maxBufferSize || 0);
      if (maxBuffer) log(`GPU maxBufferSize: ${(maxBuffer / 1024 / 1024).toFixed(0)}MB`);
    } catch (detailError) {
      log(`GPU上限値を取得できませんでした: ${detailError.message || detailError}`);
    }
    if (LOW_MEMORY_MODE) log("iPhone/iPad低メモリ読み込みモードを使用します。");
  } catch (e) {
    const message = e.message || String(e);
    els.badge.textContent = "GPU確認エラー";
    els.badge.className = "badge bad";
    els.badge.title = message;
    els.gpuSupport.textContent = "エラー";
    els.gpuName.textContent = message;
    document.querySelector(".diagnostics:not(.adminOnly)")?.setAttribute("open", "");
    log(`WebGPU ERROR: ${message}`);
  }
}

async function loadVoiceConfig() {
  try {
    const r = await fetch("./config/voices.json", { cache: "no-store" });
    if (!r.ok) throw new Error(`voices.json ${r.status}`);
    const json = await r.json();
    voiceConfig = Array.isArray(json.voices) ? json.voices : [];
  } catch (e) {
    log(`VOICE CONFIG ERROR: ${e.message || e}`);
    voiceConfig = [];
  }
  renderVoices();
}
function renderVoices() {
  els.voiceSelect.innerHTML = "";
  const enabled = voiceConfig.filter(v => v.enabled);
  if (!enabled.length) {
    const o = document.createElement("option");
    o.value = ""; o.textContent = "標準ボイス未登録（管理・テスト用音声を選択してください）";
    els.voiceSelect.appendChild(o);
    els.voiceSelect.disabled = true;
    els.voiceStatus.textContent = "現在は標準ボイス未登録です。下の管理・テスト用欄から参照音声を選べます。";
  } else {
    for (const v of enabled) {
      const o = document.createElement("option");
      o.value = v.id; o.textContent = v.name;
      els.voiceSelect.appendChild(o);
    }
    els.voiceSelect.disabled = false;
    const first = enabled[0];
    els.voiceStatus.textContent = first.description || "社内共通ボイスを使用します。";
  }
  updateRunState();
}
function selectedConfiguredVoice() {
  const id = els.voiceSelect.value;
  return voiceConfig.find(v => v.enabled && v.id === id) || null;
}
function hasReferenceVoice() {
  return !!(els.ref.files?.[0] || selectedConfiguredVoice());
}
function updateRunState() {
  els.run.disabled = !(webgpuReady && hasReferenceVoice() && els.text.value.trim());
}
function qualitySteps() {
  return parseInt(document.querySelector('input[name="quality"]:checked')?.value || "8", 10);
}

async function fetchCached(url) {
  if (!("caches" in globalThis)) {
    const r = await fetch(url); if (!r.ok) throw new Error(`${r.status}: ${url}`); return new Uint8Array(await r.arrayBuffer());
  }
  const cache = await caches.open(CACHE_NAME);
  let r = await cache.match(url);
  if (!r) {
    r = await fetch(url);
    if (!r.ok) throw new Error(`モデル取得失敗 ${r.status}`);
    await cache.put(url, r.clone());
  }
  return new Uint8Array(await r.arrayBuffer());
}
const sessionOptions = (name, data) => ({
  executionProviders: ["webgpu"], graphOptimizationLevel: "all",
  externalData: [{ path: `${name}.onnx.data`, data }],
});
async function createSession(name) {
  const modelUrl = `${MODEL_BASE}/${name}.onnx`;
  const dataUrl = `${MODEL_BASE}/${name}.onnx.data`;
  if (LOW_MEMORY_MODE) {
    // Let ONNX Runtime fetch the files directly. Avoids the app-level
    // Response.clone() + ArrayBuffer copies that can exhaust Safari's tab memory.
    return await ort.InferenceSession.create(modelUrl, sessionOptions(name, dataUrl));
  }
  const [model, data] = await Promise.all([fetchCached(modelUrl), fetchCached(dataUrl)]);
  return await ort.InferenceSession.create(model, sessionOptions(name, data));
}
function readableError(error) {
  const raw = error?.message || String(error);
  if (LOW_MEMORY_MODE && /load failed|memory|allocation|buffer/i.test(raw)) {
    return "大容量モデルの通信またはメモリ確保に失敗しました。Wi-Fiへ接続し、Safariを一度閉じてから再度お試しください。繰り返す場合、このiPhoneでは約1.25GBのモデルを端末内実行できません。";
  }
  return raw;
}
async function loadModel() {
  if (tts) return tts;
  setProgress(3, "AIモデルを準備しています", "初回は約1.25GB取得します。Wi-Fi環境でお待ちください。");
  if (LOW_MEMORY_MODE && "caches" in globalThis) {
    // Remove incomplete v0.5 downloads; v0.6 relies on normal HTTP caching on iOS.
    await caches.delete(CACHE_NAME);
  }
  const sessions = {};
  const entries = LOW_MEMORY_MODE ? LOW_MEMORY_LOAD_ORDER : Object.entries(MODELS);
  for (let i=0; i<entries.length; i++) {
    const [key, name] = entries[i];
    const started = performance.now();
    setProgress(6 + i * 11, "AIモデルを準備しています", `${i+1}/${entries.length} ${name} を読み込んでいます…`);
    sessions[key] = await createSession(name);
    log(`${name}: ${((performance.now()-started)/1000).toFixed(1)}秒`);
  }
  setProgress(74, "AIモデルを準備しています", "日本語処理を準備しています…");
  tokenizer = tokenizer || await AutoTokenizer.from_pretrained("llmjp_tok");
  tts = new IrodoriTTS({ ort, sessions, tokenizer });
  setProgress(78, "準備完了", "音声を生成します。");
  log("models ready (fp16)");
  return tts;
}

async function configuredVoiceBlob() {
  const v = selectedConfiguredVoice();
  if (!v) return null;
  const r = await fetch(v.src, { cache: "force-cache" });
  if (!r.ok) throw new Error(`標準ボイスを取得できません (${r.status})`);
  return await r.blob();
}
async function currentReferenceBlob() {
  const temp = els.ref.files?.[0];
  if (temp) return temp;
  return await configuredVoiceBlob();
}
async function blobToMono48k(blob) {
  const arr = await blob.arrayBuffer();
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(arr);
  await ctx.close();
  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 48000), 48000);
  const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start();
  return (await off.startRendering()).getChannelData(0).slice();
}
function encodeWav(f32, sr) {
  const n=f32.length, buf=new ArrayBuffer(44+n*2), dv=new DataView(buf);
  const ws=(o,s)=>{for(let i=0;i<s.length;i++)dv.setUint8(o+i,s.charCodeAt(i));};
  ws(0,"RIFF");dv.setUint32(4,36+n*2,true);ws(8,"WAVE");ws(12,"fmt ");dv.setUint32(16,16,true);dv.setUint16(20,1,true);dv.setUint16(22,1,true);dv.setUint32(24,sr,true);dv.setUint32(28,sr*2,true);dv.setUint16(32,2,true);dv.setUint16(34,16,true);ws(36,"data");dv.setUint32(40,n*2,true);
  for(let i=0;i<n;i++)dv.setInt16(44+i*2,Math.max(-1,Math.min(1,f32[i]))*32767,true);
  return new Blob([buf],{type:"audio/wav"});
}
async function generate() {
  const text = els.text.value.trim();
  if (!text || !hasReferenceVoice()) return;
  els.run.disabled = true; els.result.hidden = true;
  const steps = qualitySteps();
  try {
    const model = await loadModel();
    setProgress(80, "参照音声を準備しています", "店内アナウンス用ボイスを準備しています…");
    const refBlob = await currentReferenceBlob();
    if (!refBlob) throw new Error("参照音声がありません");
    const ref = await blobToMono48k(refBlob);
    log(`reference: ${(ref.length/48000).toFixed(2)}秒${els.ref.files?.[0] ? " (temporary)" : " (configured)"}`);
    setProgress(84, "音声を生成しています", `品質: ${steps === 8 ? "標準" : "高品質"} / ページを閉じずにお待ちください。`);
    const start = performance.now();
    const { audio, sampleRate, seqLen } = await model.synthesize(text, ref, 48000, { numSteps: steps, seed: 0 });
    const sec=(performance.now()-start)/1000, dur=audio.length/sampleRate;
    log(`done: ${dur.toFixed(2)}秒音声 / 生成 ${sec.toFixed(1)}秒 / RTF ${(sec/dur).toFixed(2)}x / seqLen ${seqLen}`);
    setProgress(100, "生成完了", "再生して内容をご確認ください。");
    const blob=encodeWav(audio,sampleRate);
    if(objectUrl)URL.revokeObjectURL(objectUrl);
    objectUrl=URL.createObjectURL(blob);
    els.out.src=objectUrl; els.dl.href=objectUrl;
    els.dl.download=`announcement_${new Date().toISOString().slice(0,10)}.wav`;
    els.resultStats.textContent=`${dur.toFixed(1)}秒の音声を ${sec.toFixed(1)}秒で生成しました。`;
    els.result.hidden=false;
    setTimeout(()=>els.result.scrollIntoView({behavior:"smooth",block:"center"}),100);
  } catch(e) {
    log(`ERROR: ${e.stack || e.message || e}`);
    const message = readableError(e);
    setProgress(0, "生成できませんでした", message);
    alert(`音声生成中にエラーが発生しました。\n\n${message}\n\n下部の「端末情報・処理ログ」を確認してください。`);
  } finally {
    updateRunState();
  }
}

els.text.addEventListener("input",()=>{els.charCount.textContent=`${els.text.value.length}文字`;updateRunState();});
els.ref.addEventListener("change",()=>{
  const f=els.ref.files?.[0];
  els.fileTitle.textContent=f?`${f.name}（テスト用として優先）`:"テスト用音声ファイルを選択";
  if (f) els.voiceStatus.textContent = "管理・テスト用の一時音声を使用します。";
  else renderVoices();
  updateRunState();
});
els.voiceSelect.addEventListener("change",()=>{
  const v=selectedConfiguredVoice();
  if(v && !els.ref.files?.[0]) els.voiceStatus.textContent=v.description || `${v.name}を使用します。`;
  updateRunState();
});
els.run.addEventListener("click",generate);
for(const q of document.querySelectorAll('input[name="quality"]'))q.addEventListener("change",()=>log(`quality: ${qualitySteps()} steps`));
els.charCount.textContent=`${els.text.value.length}文字`;

await loadVoiceConfig();
await checkDevice();
