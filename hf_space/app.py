"""
Emotion Detector — WavLM-large local app
Run: python emotion_app.py
Then open http://localhost:7860
"""
import csv
import io
import re
import torch
import numpy as np
import soundfile as sf
import librosa
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "clean_ser_best"

CLASSES = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad"]
EMOJIS  = {"Angry":"😡","Disgusted":"🤢","Fearful":"😨","Happy":"😄","Neutral":"😐","Sad":"😢"}
COLORS  = {"Angry":"#e53935","Disgusted":"#43a047","Fearful":"#7b1fa2",
           "Happy":"#f9a825","Neutral":"#1565c0","Sad":"#00838f"}

# ── Result logging ────────────────────────────────────────────────────────────
# Audio is only written to disk when the participant opts in (see save_audio).
RESULTS_PATH = Path(__file__).parent / "results.csv"
RECORDINGS_DIR = Path(__file__).parent / "recordings"
RESULTS_FIELDS = ["timestamp_utc", "name", "top_emotion"] + CLASSES + ["audio_filename"]

def sanitize_name(name):
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name or "anonymous"

def log_result(name, emotion, probs, audio_filename=""):
    is_new = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(RESULTS_FIELDS)
        writer.writerow(
            [datetime.now(timezone.utc).isoformat(timespec="seconds"), name, emotion]
            + [probs[c] for c in CLASSES]
            + [audio_filename]
        )

def migrate_results_csv():
    """Backfill the audio_filename column into a results.csv written before it existed."""
    if not RESULTS_PATH.exists():
        return
    with open(RESULTS_PATH, newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] == RESULTS_FIELDS:
        return
    migrated = [RESULTS_FIELDS] + [row + [""] for row in rows[1:]]
    with open(RESULTS_PATH, "w", newline="") as f:
        csv.writer(f).writerows(migrated)

migrate_results_csv()

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Loading WavLM-large from {MODEL_PATH} on {DEVICE} …")
from transformers import AutoFeatureExtractor, WavLMForSequenceClassification
fe    = AutoFeatureExtractor.from_pretrained(str(MODEL_PATH))
model = WavLMForSequenceClassification.from_pretrained(str(MODEL_PATH)).to(DEVICE)
model.eval()
print("Model ready.")

# ── HTML ───────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Emotion Detector</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Segoe UI',sans-serif;
    background:#0d0d1a;color:#fff;
    min-height:100vh;
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    gap:20px;padding:30px 16px;
  }
  h1{font-size:1.8rem;letter-spacing:.06em;opacity:.9}

  /* ── slot window ── */
  .slot-window{
    width:200px;height:200px;
    border:3px solid #2a2a44;
    border-radius:20px;
    display:flex;align-items:center;justify-content:center;
    background:#12122a;
    position:relative;overflow:hidden;
    transition:border-color .5s,box-shadow .5s;
  }
  .slot-window.glow{
    border-color:var(--ec);
    box-shadow:0 0 0 4px color-mix(in srgb,var(--ec) 25%,transparent),
               0 0 50px color-mix(in srgb,var(--ec) 40%,transparent);
  }

  #spin-emoji{
    font-size:6.5rem;line-height:1;
    transition:transform .45s cubic-bezier(.34,1.56,.64,1),filter .4s;
    user-select:none;
  }
  #spin-emoji.land{transform:scale(1.25)}

  /* shimmer overlay while spinning */
  .slot-window::after{
    content:'';position:absolute;inset:0;
    background:linear-gradient(180deg,rgba(255,255,255,.06) 0%,transparent 40%,transparent 60%,rgba(255,255,255,.06) 100%);
    pointer-events:none;
  }

  /* ── emotion label ── */
  #emo-label{
    font-size:1.7rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
    min-height:2.2rem;opacity:0;
    transition:opacity .4s,color .3s;
  }
  #emo-label.show{opacity:1;color:var(--ec)}

  /* ── record button ── */
  #rec-btn{
    padding:13px 38px;font-size:1rem;font-weight:600;border:none;border-radius:50px;
    cursor:pointer;background:#4a90e2;color:#fff;
    display:flex;align-items:center;gap:8px;
    transition:background .2s,transform .1s,box-shadow .2s;
  }
  #rec-btn:hover:not(:disabled){background:#357abd}
  #rec-btn:active:not(:disabled){transform:scale(.97)}
  #rec-btn.rec{background:#e53935;animation:pulse 1.1s ease-in-out infinite}
  #rec-btn:disabled{opacity:.45;cursor:not-allowed}

  #status{font-size:.85rem;color:#777;min-height:1.3rem;text-align:center}

  /* ── name field ── */
  #name-field{display:flex;flex-direction:column;align-items:center;gap:4px}
  #name-field label{font-size:.78rem;color:#8fb8ee;letter-spacing:.03em;font-weight:600}
  #name-input{
    width:220px;padding:9px 14px;font-size:.9rem;
    border:2px solid #4a90e2;border-radius:10px;
    background:#12122a;color:#fff;text-align:center;
    box-shadow:0 0 10px rgba(74,144,226,.4);
    transition:box-shadow .3s,border-color .3s;
  }
  #name-input::placeholder{color:#8fb8ee;opacity:.85}
  #name-input:focus{outline:2px solid #4a90e2;outline-offset:1px;box-shadow:0 0 16px rgba(74,144,226,.65)}
  #name-input.invalid{
    border-color:#e53935;
    box-shadow:0 0 0 4px rgba(229,57,53,.25),0 0 16px rgba(229,57,53,.5);
    animation:shake .4s ease;
  }
  #name-error{font-size:.72rem;color:#ff6b64;min-height:1rem;opacity:0;transition:opacity .25s}
  #name-error.show{opacity:1}
  @keyframes shake{
    0%,100%{transform:translateX(0)}
    20%{transform:translateX(-6px)}
    40%{transform:translateX(6px)}
    60%{transform:translateX(-4px)}
    80%{transform:translateX(4px)}
  }

  /* ── consent checkbox ── */
  #consent-field{
    display:flex;align-items:flex-start;gap:8px;
    width:260px;font-size:.74rem;color:#999;font-style:italic;line-height:1.35;
  }
  #consent-field label{display:flex;align-items:flex-start;gap:8px;cursor:pointer}
  #consent-field input[type=checkbox]{
    margin-top:2px;width:15px;height:15px;accent-color:#4a90e2;cursor:pointer;flex-shrink:0;
  }

  /* ── probability bars ── */
  #bars{width:320px;display:flex;flex-direction:column;gap:7px;opacity:0;transition:opacity .5s}
  #bars.show{opacity:1}
  .bar-row{display:flex;align-items:center;gap:8px;transition:filter .3s}
  .bar-row.winner{filter:brightness(1.35)}
  .b-ico{font-size:1.15rem;width:26px;text-align:center}
  .b-lbl{width:76px;font-size:.76rem;color:#aaa;letter-spacing:.02em}
  .b-track{flex:1;height:10px;background:#1e1e3a;border-radius:5px;overflow:hidden}
  .b-fill{height:100%;border-radius:5px;width:0%;transition:width .7s ease}
  .b-pct{width:38px;text-align:right;font-size:.73rem;color:#ccc}

  @keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(229,57,53,.45)}50%{box-shadow:0 0 0 10px rgba(229,57,53,0)}}

  /* ── recording panel (dot + wave bars + timer) ── */
  #rec-panel{
    display:none;flex-direction:column;align-items:center;gap:10px;
    width:260px;
  }
  #rec-panel.show{display:flex}

  .rec-dot-row{display:flex;align-items:center;gap:8px}
  .rec-dot{width:12px;height:12px;border-radius:50%;background:#e53935;animation:dotpulse 1s ease-in-out infinite}
  .rec-text{font-size:.95rem;font-weight:700;color:#ff6b64;letter-spacing:.04em}

  .wave-bars{display:flex;align-items:flex-end;justify-content:center;gap:5px;height:36px}
  .wave-bar{width:6px;height:6px;border-radius:3px;background:#e53935;transition:height .06s linear}
  .wave-bar.loop{animation:waveloop .9s ease-in-out infinite}
  .wave-bar:nth-child(1){animation-delay:0s}
  .wave-bar:nth-child(2){animation-delay:.1s}
  .wave-bar:nth-child(3){animation-delay:.2s}
  .wave-bar:nth-child(4){animation-delay:.3s}
  .wave-bar:nth-child(5){animation-delay:.4s}

  .timer-row{display:flex;flex-direction:column;align-items:center;gap:6px;width:100%}
  #timer-text{font-size:.85rem;color:#ccc;font-variant-numeric:tabular-nums}
  .timer-track{width:100%;height:6px;background:#1e1e3a;border-radius:3px;overflow:hidden}
  .timer-fill{height:100%;width:0%;background:#e53935;border-radius:3px}

  @keyframes dotpulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(.75)}}
  @keyframes waveloop{0%,100%{height:6px}50%{height:30px}}
</style>
</head>
<body>

<h1>🎭 Emotion Detector</h1>

<div id="name-field">
  <label for="name-input">Your name</label>
  <input id="name-input" name="participant_name" type="text" placeholder="e.g. Sam"
         maxlength="40" autocomplete="name" spellcheck="false">
  <div id="name-error">Please enter your name before recording</div>
</div>

<div id="consent-field">
  <label>
    <input type="checkbox" id="consent-checkbox">
    <span>Optional: allow saving my voice recording for research. No audio is shared publicly.</span>
  </label>
</div>

<div class="slot-window" id="slot">
  <span id="spin-emoji">🎤</span>
</div>

<div id="rec-panel">
  <div class="rec-dot-row">
    <span class="rec-dot"></span>
    <span class="rec-text">Recording…</span>
  </div>
  <div class="wave-bars">
    <span class="wave-bar"></span>
    <span class="wave-bar"></span>
    <span class="wave-bar"></span>
    <span class="wave-bar"></span>
    <span class="wave-bar"></span>
  </div>
  <div class="timer-row">
    <div id="timer-text">0.0s / 5.0s</div>
    <div class="timer-track"><div class="timer-fill" id="timer-fill"></div></div>
  </div>
</div>

<div id="emo-label"></div>

<div id="bars"></div>

<button id="rec-btn">🎤 Record</button>
<div id="status">Press Record and say something (1–5 sec)</div>

<script>
const EMO = [
  {name:"Angry",    emoji:"😡", color:"#e53935"},
  {name:"Disgusted",emoji:"🤢", color:"#43a047"},
  {name:"Fearful",  emoji:"😨", color:"#7b1fa2"},
  {name:"Happy",    emoji:"😄", color:"#f9a825"},
  {name:"Neutral",  emoji:"😐", color:"#1565c0"},
  {name:"Sad",      emoji:"😢", color:"#00838f"},
];

// build bars
const barsDiv = document.getElementById('bars');
EMO.forEach(e => {
  barsDiv.insertAdjacentHTML('beforeend',
    `<div class="bar-row" id="row-${e.name}">
       <div class="b-ico">${e.emoji}</div>
       <div class="b-lbl">${e.name}</div>
       <div class="b-track"><div class="b-fill" id="fill-${e.name}" style="background:${e.color}"></div></div>
       <div class="b-pct" id="pct-${e.name}">0%</div>
     </div>`);
});

const btn       = document.getElementById('rec-btn');
const status    = document.getElementById('status');
const emoEl     = document.getElementById('spin-emoji');
const label     = document.getElementById('emo-label');
const slot      = document.getElementById('slot');
const nameInput = document.getElementById('name-input');
const nameError = document.getElementById('name-error');
const consentCheckbox = document.getElementById('consent-checkbox');

nameInput.addEventListener('input', ()=>{
  if(nameInput.value.trim()){
    nameInput.classList.remove('invalid');
    nameError.classList.remove('show');
  }
});

let mr, chunks=[], spinTimer, spinIdx=0;
let audioCtx, analyser, waveData, waveRafId, timerInterval, recStartTime;
const MAX_RECORD_MS = 5000;

const recPanel  = document.getElementById('rec-panel');
const waveBars  = document.querySelectorAll('.wave-bar');
const timerText = document.getElementById('timer-text');
const timerFill = document.getElementById('timer-fill');

function setEC(color){
  slot.style.setProperty('--ec', color);
  label.style.setProperty('--ec', color);
}

function resetUI(){
  setEC('#4a90e2');
  slot.classList.remove('glow');
  emoEl.classList.remove('land');
  emoEl.textContent = '🎤';
  label.classList.remove('show');
  label.textContent = '';
  barsDiv.classList.remove('show');
  document.querySelectorAll('.bar-row').forEach(r=>r.classList.remove('winner'));
  EMO.forEach(e=>{
    document.getElementById('fill-'+e.name).style.width='0%';
    document.getElementById('pct-'+e.name).textContent='0%';
  });
  recPanel.classList.remove('show');
  timerText.textContent = '0.0s / 5.0s';
  timerFill.style.width = '0%';
  waveBars.forEach(bar=>{ bar.style.height='6px'; bar.classList.remove('loop'); });
}

function startWave(stream){
  audioCtx = new (window.AudioContext||window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 64;
  analyser.smoothingTimeConstant = 0.75;
  source.connect(analyser);
  waveData = new Uint8Array(analyser.frequencyBinCount);

  const n = waveBars.length;
  const chunk = Math.max(1, Math.floor(waveData.length / n));
  (function tick(){
    analyser.getByteFrequencyData(waveData);
    waveBars.forEach((bar,i)=>{
      let sum=0, count=0;
      for(let j=i*chunk;j<(i+1)*chunk && j<waveData.length;j++){ sum+=waveData[j]; count++; }
      const level = Math.min(1,(sum/(count||1))/160);
      bar.style.height = (6 + level*30)+'px';
    });
    waveRafId = requestAnimationFrame(tick);
  })();
}

function stopWave(){
  if(waveRafId) cancelAnimationFrame(waveRafId);
  waveRafId = null;
  if(audioCtx){ audioCtx.close(); audioCtx = null; }
  analyser = null;
  waveBars.forEach(bar=>{ bar.style.height='6px'; bar.classList.remove('loop'); });
}

function startTimer(){
  recStartTime = performance.now();
  timerInterval = setInterval(()=>{
    const elapsed = Math.min(MAX_RECORD_MS, performance.now()-recStartTime);
    timerText.textContent = (elapsed/1000).toFixed(1)+'s / 5.0s';
    timerFill.style.width = (elapsed/MAX_RECORD_MS*100)+'%';
    if(elapsed >= MAX_RECORD_MS){
      clearInterval(timerInterval);
      timerInterval = null;
      if(mr && mr.state==='recording') mr.stop();
    }
  }, 100);
}

function stopTimer(){
  if(timerInterval){ clearInterval(timerInterval); timerInterval=null; }
}

function spinFast(){
  clearTimeout(spinTimer);
  spinTimer = setInterval(()=>{
    emoEl.textContent = EMO[spinIdx++ % EMO.length].emoji;
  }, 80);
}

function spinLand(targetName, done){
  clearInterval(spinTimer);
  const tIdx = EMO.findIndex(e=>e.name===targetName);
  const cur  = spinIdx % EMO.length;
  // steps needed to reach target with ≥3 full rotations
  let steps  = (tIdx - cur + EMO.length) % EMO.length;
  if(steps < 2) steps += EMO.length;
  steps += EMO.length * 3;

  function tick(i){
    if(i > steps){ emoEl.textContent = EMO[tIdx].emoji; done(); return; }
    const progress = i / steps;
    const delay = 80 + Math.pow(progress, 2) * 340;   // ease out
    emoEl.textContent = EMO[(cur + i) % EMO.length].emoji;
    spinTimer = setTimeout(()=>tick(i+1), delay);
  }
  tick(0);
}

btn.addEventListener('click', async ()=>{
  if(mr && mr.state==='recording'){ mr.stop(); return; }

  if(!nameInput.value.trim()){
    nameInput.classList.add('invalid');
    nameError.classList.add('show');
    nameInput.focus();
    return;
  }
  nameInput.classList.remove('invalid');
  nameError.classList.remove('show');

  let stream;
  try{
    stream = await navigator.mediaDevices.getUserMedia({audio:true});
  }catch(err){
    status.textContent = '❌ Mic access denied.';
    return;
  }
  chunks=[];
  mr = new MediaRecorder(stream);
  mr.ondataavailable = e=>chunks.push(e.data);

  mr.onstart = ()=>{
    resetUI();
    btn.textContent='⏹ Stop';
    btn.classList.add('rec');
    status.textContent='🔴 Recording… auto-stops at 5s, or click Stop';
    emoEl.textContent='🎤';
    recPanel.classList.add('show');
    try{
      startWave(stream);
    }catch(err){
      waveBars.forEach(bar=>bar.classList.add('loop'));
    }
    startTimer();
  };

  mr.onstop = async ()=>{
    stream.getTracks().forEach(t=>t.stop());
    stopWave();
    stopTimer();
    recPanel.classList.remove('show');
    btn.disabled=true;
    btn.textContent='⏳';
    btn.classList.remove('rec');
    status.textContent='🔄 Analyzing…';
    spinFast();

    try{
      const blob = new Blob(chunks, {type: mr.mimeType});
      const arrayBuf = await blob.arrayBuffer();
      const ctx = new AudioContext({sampleRate:16000});
      const decoded = await ctx.decodeAudioData(arrayBuf);
      // mix down to mono
      let pcm = decoded.getChannelData(0);
      if(decoded.numberOfChannels > 1){
        const right = decoded.getChannelData(1);
        pcm = pcm.map((v,i)=>(v+right[i])/2);
      }
      const wav = encodePCM(pcm, 16000);

      const fd = new FormData();
      fd.append('audio', new Blob([wav],{type:'audio/wav'}), 'clip.wav');
      fd.append('name', nameInput.value.trim());
      fd.append('save_audio', consentCheckbox.checked ? 'true' : 'false');
      const resp = await fetch('/predict',{method:'POST',body:fd});
      if(!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();

      spinLand(data.emotion, ()=>{
        const emo = EMO.find(e=>e.name===data.emotion);
        setEC(emo.color);
        slot.classList.add('glow');
        emoEl.classList.add('land');

        label.textContent = emo.emoji + ' ' + data.emotion;
        label.classList.add('show');

        EMO.forEach(e=>{
          const pct = Math.round((data.probabilities[e.name]||0)*100);
          document.getElementById('fill-'+e.name).style.width = pct+'%';
          document.getElementById('pct-'+e.name).textContent = pct+'%';
        });
        document.getElementById('row-'+data.emotion).classList.add('winner');
        barsDiv.classList.add('show');

        status.textContent='✅ Done!  Press Record to try again.';
        btn.disabled=false;
        btn.textContent='🎤 Record Again';
      });
    }catch(err){
      clearInterval(spinTimer);
      emoEl.textContent='❌';
      status.textContent='Error: '+err.message;
      btn.disabled=false;
      btn.textContent='🎤 Record Again';
    }
  };

  mr.start();
});

function encodePCM(samples, sr){
  const buf  = new ArrayBuffer(44 + samples.length*2);
  const view = new DataView(buf);
  const str  = (off,s)=>{ for(let i=0;i<s.length;i++) view.setUint8(off+i,s.charCodeAt(i)); };
  str(0,'RIFF'); view.setUint32(4,36+samples.length*2,true);
  str(8,'WAVE'); str(12,'fmt ');
  view.setUint32(16,16,true);  view.setUint16(20,1,true);
  view.setUint16(22,1,true);   view.setUint32(24,sr,true);
  view.setUint32(28,sr*2,true);view.setUint16(32,2,true);
  view.setUint16(34,16,true);  str(36,'data');
  view.setUint32(40,samples.length*2,true);
  let off=44;
  for(let i=0;i<samples.length;i++,off+=2){
    const s=Math.max(-1,Math.min(1,samples[i]));
    view.setInt16(off,s<0?s*0x8000:s*0x7FFF,true);
  }
  return buf;
}
</script>
</body>
</html>"""

# ── API ────────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

@app.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    name: str = Form(""),
    save_audio: str = Form("false"),
):
    data = await audio.read()

    # Write to disk (if opted in) before model inference, so a model error
    # never loses the clip. Filename is finalized once the emotion is known.
    should_save = save_audio.strip().lower() == "true"
    pending_path = None
    audio_filename = ""
    if should_save:
        RECORDINGS_DIR.mkdir(exist_ok=True)
        safe_name = sanitize_name(name)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        pending_path = RECORDINGS_DIR / f"{timestamp}_{safe_name}_pending.wav"
        pending_path.write_bytes(data)

    try:
        wav, sr = sf.read(io.BytesIO(data))
    except Exception as e:
        return JSONResponse({"error": f"Could not decode audio: {e}"}, status_code=400)

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
    wav = wav.astype("float32")

    inputs = fe(wav, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().float().numpy()

    probs = np.exp(logits) / np.exp(logits).sum()   # softmax
    pred  = int(probs.argmax())
    emotion = CLASSES[pred]
    probs_dict = {c: round(float(p), 4) for c, p in zip(CLASSES, probs)}

    if pending_path is not None:
        final_path = pending_path.with_name(pending_path.name.replace("_pending.wav", f"_{emotion.lower()}.wav"))
        pending_path.rename(final_path)
        audio_filename = final_path.name

    log_result(name.strip(), emotion, probs_dict, audio_filename)

    return {
        "emotion": emotion,
        "emoji":   EMOJIS[emotion],
        "color":   COLORS[emotion],
        "probabilities": probs_dict,
    }

@app.get("/results")
def results():
    if not RESULTS_PATH.exists():
        with open(RESULTS_PATH, "w", newline="") as f:
            csv.writer(f).writerow(RESULTS_FIELDS)
    return FileResponse(str(RESULTS_PATH), media_type="text/csv", filename="results.csv")

if __name__ == "__main__":
    print("Open http://localhost:7860 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
