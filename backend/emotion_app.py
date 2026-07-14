"""
Emotion Detector — WavLM-large local app
Run: python emotion_app.py
Then open http://localhost:7860
"""
import io
import json
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "w2v2_ser_best"

CLASSES = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad"]
EMOJIS  = {"Angry":"😡","Disgusted":"🤢","Fearful":"😨","Happy":"😄","Neutral":"😐","Sad":"😢"}
COLORS  = {"Angry":"#e53935","Disgusted":"#43a047","Fearful":"#7b1fa2",
           "Happy":"#f9a825","Neutral":"#1565c0","Sad":"#00838f"}

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
</style>
</head>
<body>

<h1>🎭 Emotion Detector</h1>

<div class="slot-window" id="slot">
  <span id="spin-emoji">🎤</span>
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

const btn    = document.getElementById('rec-btn');
const status = document.getElementById('status');
const emoEl  = document.getElementById('spin-emoji');
const label  = document.getElementById('emo-label');
const slot   = document.getElementById('slot');

let mr, chunks=[], spinTimer, spinIdx=0;

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
    status.textContent='🔴 Recording… click Stop when done';
    emoEl.textContent='🎤';
  };

  mr.onstop = async ()=>{
    stream.getTracks().forEach(t=>t.stop());
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
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    data = await audio.read()
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

    return {
        "emotion": emotion,
        "emoji":   EMOJIS[emotion],
        "color":   COLORS[emotion],
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    }

if __name__ == "__main__":
    print("Open http://localhost:7860 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
