"""
GTZAN Music Genre Classifier — Streamlit App
Run: streamlit run app.py
Place your trained model file (gtzan_efficientnet_final.pth) in the same directory.
Install: pip install streamlit torch torchaudio timm numpy librosa matplotlib plotly
"""

import io, time, random, tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import librosa

def load_audio(path):
    """Load audio using librosa — no FFmpeg required. Supports WAV, MP3, FLAC, OGG, M4A."""
    audio_np, sr = librosa.load(path, sr=None, mono=False)
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, :]   # (1, samples)
    waveform = torch.tensor(audio_np, dtype=torch.float32)
    return waveform, sr
import timm
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SoundLens · Genre AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Theming: dark editorial / music-magazine aesthetic ────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root variables ── */
:root {
  --bg:        #0a0a0f;
  --bg2:       #12121a;
  --bg3:       #1a1a26;
  --accent:    #c084fc;
  --accent2:   #818cf8;
  --accent3:   #38bdf8;
  --gold:      #fbbf24;
  --text:      #e2e8f0;
  --text-muted:#94a3b8;
  --border:    rgba(192,132,252,0.18);
  --glow:      rgba(192,132,252,0.25);
}

/* ── Base ── */
html, body, .stApp {
  background-color: var(--bg) !important;
  color: var(--text);
  font-family: 'Syne', sans-serif;
}

/* Noise texture overlay */
.stApp::before {
  content: "";
  position: fixed; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events: none; z-index: 0; opacity: 0.4;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d0d18 0%, #0a0a0f 100%);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Headers ── */
h1 { 
  font-family: 'Syne', sans-serif !important;
  font-weight: 800 !important;
  font-size: 2.6rem !important;
  background: linear-gradient(135deg, var(--accent), var(--accent2), var(--accent3));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -1px;
  line-height: 1.1 !important;
}
h2, h3 {
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  color: var(--text) !important;
}

/* ── Cards / containers ── */
.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem 1.8rem;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
}
.card::before {
  content:"";
  position:absolute; top:-60px; right:-60px;
  width:180px; height:180px;
  background: radial-gradient(circle, var(--glow) 0%, transparent 70%);
  border-radius:50%;
}
.result-card {
  background: linear-gradient(135deg, #1a1030 0%, #0d1428 100%);
  border: 1px solid var(--accent);
  border-radius: 20px;
  padding: 2rem 2.4rem;
  box-shadow: 0 0 40px var(--glow), 0 0 80px rgba(129,140,248,0.08);
  text-align: center;
}
.genre-badge {
  display: inline-block;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #fff;
  font-family: 'Space Mono', monospace;
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: 3px;
  text-transform: uppercase;
  padding: 0.5rem 2rem;
  border-radius: 50px;
  margin: 0.8rem 0;
  box-shadow: 0 4px 24px var(--glow);
}
.confidence-text {
  font-family: 'Space Mono', monospace;
  color: var(--accent3);
  font-size: 1.1rem;
}
.mono {
  font-family: 'Space Mono', monospace;
  font-size: 0.85rem;
  color: var(--text-muted);
}
.pill {
  display: inline-block;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 50px;
  padding: 0.2rem 0.8rem;
  font-family: 'Space Mono', monospace;
  font-size: 0.75rem;
  color: var(--accent);
  margin: 0.15rem;
}
.divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.2rem 0;
}
.waveform-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  color: var(--text-muted);
  letter-spacing: 2px;
  text-transform: uppercase;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--bg2) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 16px !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
  box-shadow: 0 0 20px var(--glow) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  font-family: 'Space Mono', monospace !important;
  font-weight: 700 !important;
  letter-spacing: 1px !important;
  padding: 0.6rem 2rem !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 15px var(--glow) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 25px var(--glow) !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 1rem 1.2rem !important;
}
[data-testid="stMetricValue"] {
  color: var(--accent) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

/* ── Progress bar ── */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
  border-radius: 50px !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* ── Plotly transparent bg ── */
.js-plotly-plot { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants & config ────────────────────────────────────────
MODEL_PATH   = "E:\DEEP_LEARNING\gtzan_efficientnet_final.pth"
SAMPLE_RATE  = 22050
N_MELS       = 128
N_FFT        = 2048
HOP_LENGTH   = 512
SEGMENT_DUR  = 5
SAMPLES_SEG  = SAMPLE_RATE * SEGMENT_DUR
IMG_SIZE     = 224
DEVICE       = torch.device("cpu")   # CPU for Streamlit inference

GENRE_EMOJIS = {
    "blues":     "🎸", "classical": "🎻", "country": "🤠",
    "disco":     "🕺", "hiphop":    "🎤", "jazz":    "🎷",
    "metal":     "🤘", "pop":       "🎵", "reggae":  "🌴",
    "rock":      "⚡",
}
GENRE_COLORS = {
    "blues":     "#3b82f6", "classical": "#a78bfa", "country": "#f59e0b",
    "disco":     "#ec4899", "hiphop":    "#10b981", "jazz":    "#f97316",
    "metal":     "#ef4444", "pop":       "#8b5cf6", "reggae":  "#22c55e",
    "rock":      "#06b6d4",
}
GENRE_DESC = {
    "blues":     "Soulful, expressive guitar and vocal-driven — rooted in African American history",
    "classical": "Orchestral compositions with complex harmonic structure",
    "country":   "Storytelling melodies with twangy guitars and rural themes",
    "disco":     "Four-on-the-floor beats, bass lines, and danceable grooves from the 70s",
    "hiphop":    "Rhythmic beats, sampling, and vocal rap — genre-defining urban sound",
    "jazz":      "Improvisation, swing, and syncopated rhythms across complex chord changes",
    "metal":     "Heavy distorted guitars, aggressive drumming, and high-energy vocals",
    "pop":       "Hook-driven melodies designed for broad appeal and radio airplay",
    "reggae":    "Offbeat rhythms, Jamaican roots, and socially conscious lyrics",
    "rock":      "Guitar-driven energy with a back beat — the backbone of modern music",
}

# ── Model definition (must match training) ───────────────────
class GTZANClassifier(nn.Module):
    def __init__(self, num_classes=10, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False,
                                          num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim), nn.Dropout(dropout),
            nn.Linear(feat_dim, 512), nn.GELU(),
            nn.BatchNorm1d(512), nn.Dropout(dropout * 0.75),
            nn.Linear(512, 256), nn.GELU(),
            nn.BatchNorm1d(256), nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.head(self.backbone(x))

@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        return None, None
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    genres = ckpt['genres']
    model = GTZANClassifier(num_classes=len(genres))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, genres

# ── Audio processing ─────────────────────────────────────────
mel_transform  = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                                   hop_length=HOP_LENGTH, n_mels=N_MELS,
                                   f_min=20, f_max=8000, power=2.0)
amplitude_to_db = T.AmplitudeToDB(top_db=80)

def preprocess_audio(waveform, sr):
    if sr != SAMPLE_RATE:
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform

def waveform_to_segments(waveform):
    segments, specs = [], []
    total = waveform.shape[1]
    for start in range(0, total - SAMPLES_SEG + 1, SAMPLES_SEG):
        seg = waveform[:, start:start + SAMPLES_SEG]
        mel = mel_transform(seg)
        mel_db = amplitude_to_db(mel)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        spec_img = F.interpolate(mel_db.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                                  mode='bilinear', align_corners=False).squeeze(0)
        spec_img = spec_img.repeat(3, 1, 1)
        m, s = spec_img.mean(), spec_img.std() + 1e-8
        spec_img = (spec_img - m) / s
        segments.append(spec_img)
        specs.append(mel_db.squeeze().numpy())
    return segments, specs

@torch.no_grad()
def predict(model, segments, genres):
    all_probs = []
    for seg in segments:
        logit = model(seg.unsqueeze(0))
        prob  = torch.softmax(logit, dim=1).squeeze().numpy()
        all_probs.append(prob)
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx  = int(np.argmax(avg_probs))
    return genres[pred_idx], avg_probs, all_probs

# ── Plotly chart helpers ──────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Space Mono, monospace', color='#94a3b8'),
    margin=dict(l=10, r=10, t=40, b=10),
)

def make_bar_chart(genres, probs, pred_genre):
    colors = [GENRE_COLORS.get(g, '#6366f1') for g in genres]
    border = ['#fff' if g == pred_genre else 'rgba(0,0,0,0)' for g in genres]
    fig = go.Figure(go.Bar(
        x=[f"{GENRE_EMOJIS.get(g,'')} {g.capitalize()}" for g in genres],
        y=[p * 100 for p in probs],
        marker=dict(color=colors, line=dict(color=border, width=2)),
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='outside',
        textfont=dict(size=11, family='Space Mono'),
        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2f}%<extra></extra>',
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Confidence per Genre", font=dict(size=14, color='#e2e8f0')),
        yaxis=dict(gridcolor='rgba(255,255,255,0.07)', range=[0, max(probs)*120]),
        xaxis=dict(tickangle=-25),
        height=380,
    )
    return fig

def make_radar_chart(genres, probs):
    cats = [g.capitalize() for g in genres] + [genres[0].capitalize()]
    vals = list(probs) + [probs[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill='toself',
        line=dict(color='#c084fc', width=2),
        fillcolor='rgba(192,132,252,0.15)',
        hovertemplate='<b>%{theta}</b>: %{r:.3f}<extra></extra>',
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(gridcolor='rgba(255,255,255,0.08)',
                            tickfont=dict(size=9), range=[0, max(probs)*1.1]),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.08)',
                             tickfont=dict(size=11)),
        ),
        title=dict(text="Genre Radar", font=dict(size=14, color='#e2e8f0')),
        height=350,
    )
    return fig

def make_mel_spec_fig(spec_np, genre):
    color = GENRE_COLORS.get(genre, '#c084fc')
    fig = go.Figure(go.Heatmap(
        z=spec_np, colorscale='Magma', showscale=False,
        hovertemplate='Mel bin: %{y}<br>Frame: %{x}<br>dB: %{z:.1f}<extra></extra>',
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Mel Spectrogram (Segment 1)", font=dict(size=13, color='#e2e8f0')),
        xaxis=dict(title="Time frames", showgrid=False),
        yaxis=dict(title="Mel bins",    showgrid=False),
        height=280,
    )
    return fig

def make_segment_timeline(all_probs, genres):
    """Show predicted genre confidence across segments."""
    n_segs = len(all_probs)
    top_genre = [genres[np.argmax(p)] for p in all_probs]
    top_conf  = [np.max(p) * 100 for p in all_probs]
    colors = [GENRE_COLORS.get(g, '#6366f1') for g in top_genre]
    fig = go.Figure()
    for i, (g, c, col) in enumerate(zip(top_genre, top_conf, colors)):
        fig.add_trace(go.Bar(
            x=[i+1], y=[c],
            name=g.capitalize(),
            marker_color=col,
            text=f"{GENRE_EMOJIS.get(g,'')}",
            textposition='inside',
            hovertemplate=f"Seg {i+1}: {g.capitalize()} ({c:.1f}%)<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Prediction per Segment", font=dict(size=13, color='#e2e8f0')),
        xaxis=dict(title="Segment #", tickvals=list(range(1, n_segs+1))),
        yaxis=dict(title="Top Confidence (%)", gridcolor='rgba(255,255,255,0.07)'),
        height=260, bargap=0.3,
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem;'>
      <span style='font-family:Space Mono;font-size:1.3rem;font-weight:700;
                   background:linear-gradient(135deg,#c084fc,#818cf8);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   letter-spacing:2px;'>SOUNDLENS</span><br>
      <span style='font-size:0.72rem;color:#64748b;letter-spacing:3px;
                   font-family:Space Mono;'>GENRE INTELLIGENCE</span>
    </div>
    <hr style='border-color:rgba(192,132,252,0.2);margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**Supported Genres**")
    for g, emoji in GENRE_EMOJIS.items():
        color = GENRE_COLORS[g]
        st.markdown(
            f"<span class='pill' style='border-color:{color}40;color:{color};'>"
            f"{emoji} {g.capitalize()}</span>",
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color:rgba(192,132,252,0.15);margin:1rem 0;'>",
                unsafe_allow_html=True)
    st.markdown("**Model Info**")
    model_info = {
        "Architecture": "EfficientNet-B0",
        "Input":        "Mel Spectrogram",
        "Segment":      "5s windows",
        "Voting":       "Majority / Avg",
    }
    for k, v in model_info.items():
        st.markdown(
            f"<div class='mono'><span style='color:#c084fc'>{k}:</span> {v}</div>",
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border-color:rgba(192,132,252,0.15);margin:1rem 0;'>",
                unsafe_allow_html=True)
    show_debug = st.toggle("Show debug info", value=False)
    vote_mode  = st.radio("Prediction mode",
                           ["Average probabilities", "Majority vote"],
                           index=0)

# ── Main Layout ───────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:0.2rem;'>
  <span style='font-family:Space Mono;font-size:0.75rem;color:#64748b;letter-spacing:4px;'>
    AI · AUDIO · ANALYSIS
  </span>
</div>
""", unsafe_allow_html=True)
st.markdown("# SoundLens")
st.markdown(
    "<p style='color:#94a3b8;font-size:1.05rem;margin-top:-0.4rem;margin-bottom:1.5rem;'>"
    "Drop any song and watch neural networks decode its genre DNA in seconds.</p>",
    unsafe_allow_html=True
)

# Load model
model, genres = load_model()

if model is None:
    st.markdown(f"""
    <div class='card' style='border-color:#ef4444;'>
      <h3 style='color:#ef4444;'>⚠ Model Not Found</h3>
      <p>Place <code style='color:#c084fc;'>{MODEL_PATH}</code> in the same directory as this app.</p>
      <p class='mono'>Train the model in Google Colab using the training script, then download the .pth file.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Upload zone ───────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    label_visibility="collapsed"
)
st.markdown(
    "<p class='mono' style='text-align:center;margin-top:0.5rem;'>"
    "WAV · MP3 · FLAC · OGG · M4A &nbsp;|&nbsp; Any duration</p>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Load & preview ────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        waveform, sr = load_audio(tmp_path)
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        st.stop()

    duration = waveform.shape[1] / sr
    channels = waveform.shape[0]

    # Info row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration",   f"{duration:.1f}s")
    c2.metric("Sample Rate", f"{sr:,} Hz")
    c3.metric("Channels",   "Stereo" if channels > 1 else "Mono")
    c4.metric("Segments",   f"{int(duration // SEGMENT_DUR)}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Waveform visualization
    wave_np = waveform.mean(0).numpy()
    times   = np.linspace(0, duration, len(wave_np))
    fig_wave = go.Figure(go.Scatter(
        x=times, y=wave_np,
        mode='lines',
        line=dict(color='#c084fc', width=0.8),
        fill='tozeroy',
        fillcolor='rgba(192,132,252,0.08)',
        hovertemplate='Time: %{x:.2f}s<br>Amp: %{y:.4f}<extra></extra>',
    ))
    fig_wave.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Waveform", font=dict(size=13, color='#e2e8f0')),
        xaxis=dict(title="Time (s)", gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title="Amplitude", gridcolor='rgba(255,255,255,0.05)'),
        height=180,
    )
    st.plotly_chart(fig_wave, use_container_width=True, config={'displayModeBar': False})

    # Audio player
    st.audio(tmp_path)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Classify ──────────────────────────────────────────────
    if st.button("🎯  Classify Genre", use_container_width=True):
        with st.spinner(""):
            progress = st.progress(0, text="Loading audio...")
            waveform = preprocess_audio(waveform, sr)
            progress.progress(20, text="Slicing into segments...")
            segments, specs = waveform_to_segments(waveform)

            if not segments:
                st.error("Audio too short — need at least 5 seconds.")
                st.stop()

            progress.progress(50, text="Running neural network...")
            pred_genre, avg_probs, all_probs = predict(model, segments, genres)

            if vote_mode == "Majority vote":
                from collections import Counter
                votes = [genres[np.argmax(p)] for p in all_probs]
                pred_genre = Counter(votes).most_common(1)[0][0]

            progress.progress(90, text="Rendering results...")
            time.sleep(0.3)
            progress.progress(100, text="Done!")
            time.sleep(0.4)
            progress.empty()

        # ── Result card ───────────────────────────────────────
        genre_color = GENRE_COLORS.get(pred_genre, '#c084fc')
        emoji  = GENRE_EMOJIS.get(pred_genre, '🎵')
        conf   = float(avg_probs[genres.index(pred_genre)]) * 100

        st.markdown(f"""
        <div class='result-card' style='border-color:{genre_color};
             box-shadow: 0 0 50px {genre_color}30, 0 0 100px {genre_color}10;'>
          <div style='font-size:3.5rem;margin-bottom:0.3rem;'>{emoji}</div>
          <div style='color:#94a3b8;font-family:Space Mono;font-size:0.8rem;
                      letter-spacing:4px;text-transform:uppercase;margin-bottom:0.5rem;'>
            Detected Genre
          </div>
          <div class='genre-badge' style='background:linear-gradient(135deg,{genre_color},{genre_color}99);'>
            {pred_genre.upper()}
          </div><br>
          <div class='confidence-text'>{conf:.1f}% confidence</div>
          <hr style='border-color:rgba(255,255,255,0.08);margin:1rem 0;'>
          <p style='color:#94a3b8;font-size:0.92rem;max-width:500px;margin:0 auto;'>
            {GENRE_DESC.get(pred_genre, '')}
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.plotly_chart(make_bar_chart(genres, avg_probs, pred_genre),
                            use_container_width=True, config={'displayModeBar': False})

        with col_right:
            st.plotly_chart(make_radar_chart(genres, avg_probs),
                            use_container_width=True, config={'displayModeBar': False})

        # Mel spectrogram
        if specs:
            st.plotly_chart(make_mel_spec_fig(specs[0], pred_genre),
                            use_container_width=True, config={'displayModeBar': False})

        # Segment timeline
        if len(all_probs) > 1:
            st.plotly_chart(make_segment_timeline(all_probs, genres),
                            use_container_width=True, config={'displayModeBar': False})

        # ── Probability table ─────────────────────────────────
        st.markdown("<br><div class='card'>", unsafe_allow_html=True)
        st.markdown("**📊 Full Probability Breakdown**")
        sorted_idx = np.argsort(avg_probs)[::-1]
        for rank, idx in enumerate(sorted_idx):
            g     = genres[idx]
            p     = avg_probs[idx] * 100
            color = GENRE_COLORS.get(g, '#6366f1')
            bar_w = int(p)
            bold  = "font-weight:700;" if g == pred_genre else ""
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.8rem;margin:0.4rem 0;'>
              <span style='width:1.2rem;font-family:Space Mono;font-size:0.75rem;
                           color:#64748b;'>#{rank+1}</span>
              <span style='width:5rem;font-family:Space Mono;font-size:0.85rem;{bold}'
                    >{GENRE_EMOJIS.get(g,'')} {g.capitalize()}</span>
              <div style='flex:1;height:8px;background:rgba(255,255,255,0.05);
                          border-radius:50px;overflow:hidden;'>
                <div style='width:{bar_w}%;height:100%;
                            background:{color};border-radius:50px;
                            transition:width 0.8s ease;'></div>
              </div>
              <span style='width:4rem;text-align:right;font-family:Space Mono;
                           font-size:0.82rem;color:{color};'>{p:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if show_debug:
            st.markdown("<br><div class='card'>", unsafe_allow_html=True)
            st.markdown("**🛠 Debug Info**")
            st.markdown(f"""
            <div class='mono'>
            Segments processed: {len(segments)}<br>
            Vote mode: {vote_mode}<br>
            Avg probs shape: ({len(avg_probs)},)<br>
            Top-3: {', '.join([genres[i].upper() for i in np.argsort(avg_probs)[::-1][:3]])}<br>
            Entropy: {-np.sum(avg_probs * np.log(avg_probs + 1e-8)):.4f} nats
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

else:
    # ── Landing / empty state ─────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:3rem 1rem;'>
      <div style='font-size:4rem;'>🎧</div>
      <h3 style='color:#64748b;font-weight:400;margin-top:0.5rem;'>
        Upload a track to begin
      </h3>
      <p style='color:#475569;max-width:420px;margin:0.5rem auto;font-size:0.9rem;'>
        SoundLens uses an EfficientNet-B0 model trained on mel spectrograms to classify
        music into 10 genres with high accuracy.
      </p>
    </div>
    <div style='display:flex;justify-content:center;gap:0.6rem;flex-wrap:wrap;
                margin-top:0.5rem;padding:0 2rem;'>
    """, unsafe_allow_html=True)
    for g, emoji in GENRE_EMOJIS.items():
        c = GENRE_COLORS[g]
        st.markdown(
            f"<span class='pill' style='border-color:{c}60;color:{c};font-size:0.85rem;'>"
            f"{emoji} {g.capitalize()}</span>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)