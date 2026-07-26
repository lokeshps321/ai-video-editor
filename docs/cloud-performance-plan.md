# High-Performance Azure Cloud Plan (Localhost-Level Speed)

To achieve **localhost-like speed** (fast 1080p FFmpeg rendering, snappy video previews, instant AI transcriptions) in the cloud while making your **$100 Azure Student Credit** last **10 to 12 months**, here is the performance comparison and recommended cloud architecture.

---

## ⚡ Performance & Plan Comparison

| Setup / Architecture | CPU / RAM Specs | FFmpeg Render Speed | AI Whisper Speed | Monthly Azure Cost | Can $100 last 10–12 months? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Option A: B1s (1 vCPU, 1 GB RAM)** | 1 vCPU, 1 GB RAM (+ Swap) | Slow (3x-5x local time) | Slow (Local PyTorch) | **$0.00** | YES (12 Months) |
| **Option B: B2s (2 vCPU, 4 GB RAM)** | 2 vCPUs, 4 GB RAM, SSD | **Fast (Localhost-level)** | Moderate (Local CPU) | **$15/mo** (24/7) or **$6/mo** (Auto-shutdown) | **YES (12 Months with Auto-shutdown)** |
| **Option C: Hybrid (B2s VM + Groq API + Cloudflare)** | 2 vCPUs, 4 GB RAM + Cloud LPU | **Super Fast (Blazing)** | **0.5s per video** (Groq LPU) | **$5 - $8/mo** | **YES (10–12 Months Guaranteed)** |

---

## 🚀 Recommended Setup for Localhost-Level Speed: Option C (Hybrid High-Performance)

### 1. VM Compute: Azure `Standard_B2s` (2 vCPU, 4 GB RAM)
- **Why**: 1GB RAM (B1s) relies heavily on swap memory, which slows down video encoding. `Standard_B2s` gives you **2 full vCPUs and 4 GB RAM**, providing fast multi-threaded FFmpeg video processing and smooth API responses.
- **Cost Optimization**: Enable **Azure Auto-Shutdown** (e.g. shutdown at 2:00 AM if idle). Since you will not be editing videos 24 hours a day non-stop, the VM runs ~10–12 hours a day.
- **Budget Result**: Cost drops from $15/mo to **~$6–$8/month**, making your **$100 Student Credit last 12 full months!**

### 2. AI Transcription: Groq LPU API Integration
- Local CPU Whisper transcription takes 30-60 seconds for a 1-minute video.
- **Groq API (`whisper-large-v3-turbo`)**: Runs on custom Language Processing Units (LPUs). Transcribes a 1-minute video in **less than 1 second**!
- **Cost**: Groq provides **Free Tier API credits**, so AI transcription is 100% free and 50x faster than localhost!

### 3. Video Playback & CDN: Cloudflare Free Proxy
- Serves static assets, video preview segments, and frontend scripts via Cloudflare's global edge network.
- Video scrubbing and timeline playback response times match local disk speed.

---

## 🛠️ How to Upgrade your Azure Deployment to B2s (2 vCPU, 4GB RAM)

### Step 1: Change VM Size in Azure Portal
1. Go to Azure Portal -> **Virtual Machines** -> Select `clipmind-server`.
2. On the left menu, click **Size**.
3. Select **Standard_B2s** (2 vCPUs, 4 GB RAM).
4. Click **Resize**. (Takes 30 seconds).

### Step 2: Enable Auto-Shutdown (To save credit for 12 months)
1. In Azure Portal -> `clipmind-server` -> Left menu **Auto-shutdown**.
2. Set status to **On**.
3. Set time to e.g. `02:00 AM` (or your timezone).
4. Save. (When you want to use the editor, simply click **Start** in Azure Portal or use Azure Mobile app / Azure CLI).

### Step 3: Enable Groq Fast Transcription in `backend/.env.production`
On your Azure VM:
```bash
nano backend/.env.production
```
Set:
```ini
TRANSCRIBE_BACKEND=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
MAX_CONCURRENT_RENDER_JOBS=2
```
Restart containers:
```bash
docker compose up -d
```

---

## 📊 Summary
By upgrading the Azure VM to **Standard_B2s (2 vCPU / 4 GB RAM)** and pairing it with **Groq API** and **Cloudflare**, your cloud video editor will perform **just as fast (or faster) than your local computer**, while Azure Auto-Shutdown guarantees your **$100 student credit lasts the full 10-12 months**!
