# Complete Azure Student ($100 Credit) + Namecheap Domain Deployment Guide

This guide details how to host **ClipMind (AI Video Editor)** on **Azure** for **10 to 12 months** completely free / within your **$100 Azure Student Credit**, linked to your **Namecheap domain** from the **GitHub Student Developer Pack**.

---

## 💡 Architecture & Cost Budgeting Strategy

| Component | Cloud Solution | Cost / Month | 12-Month Total |
| :--- | :--- | :--- | :--- |
| **Compute (Server)** | Azure Linux VM (`Standard_B1s` with 4GB Swap) | **$0.00** (Free Tier in Student Pack) | **$0.00** |
| **Domain Name** | Namecheap (`.me` / `.tech` via GitHub Student Pack) | **$0.00** (Included in Pack) | **$0.00** |
| **SSL & CDN** | Cloudflare Free Tier (DNS + SSL + DDoS Protection) | **$0.00** | **$0.00** |
| **Database** | PostgreSQL 16 (Container inside Azure VM) | **$0.00** | **$0.00** |
| **Video Engine** | FFmpeg + RQ Worker (Container inside Azure VM) | **$0.00** | **$0.00** |
| **Remaining Azure Credit** | **$100.00 Credit Left Unused / Reserved for Scaling** | — | **$100.00 Safe** |

> ⚠️ **Why standard Azure App Service / Database for PostgreSQL Flexible Server fails the $100 budget:**
> Managed PostgreSQL costs ~$15-30/mo and App Service Basic costs ~$13-54/mo. That burns your $100 credit in less than 2-3 months!
> 
> **The Solution:** Azure Student Pack includes **750 hours/month of Standard_B1s Linux VM for FREE for 12 months**. Running Docker Compose on this VM with 4GB Swap file keeps your monthly spend at **$0.00/mo**, so your $100 credit remains completely untouched and your app runs **24/7 for 12 months uninterrupted!**

---

## 🛠️ Step-by-Step Deployment Instructions

### Step 1: Create Azure VM (12 Months Free)

1. Open [Azure Portal](https://portal.azure.com) and log in with your Azure Student account.
2. Search for **Virtual Machines** and click **Create** -> **Azure virtual machine**.
3. Configure the VM settings:
   - **Subscription:** Azure for Students
   - **Resource Group:** `clipmind-rg` (Create new)
   - **Virtual Machine Name:** `clipmind-server`
   - **Region:** Choose nearest (e.g. `East US` or `South India` or `West Europe`)
   - **Image:** **Ubuntu Server 22.04 LTS - x64 Gen2**
   - **Architecture:** x64
   - **Size:** Select **Standard_B1s** (1 vCPU, 1 GB RAM) — *Marked as "Free account eligible"*.
   - **Authentication type:** SSH public key (or Password for easy login)
   - **Inbound Port Rules:** Allow `SSH (22)`, `HTTP (80)`, `HTTPS (443)`
4. Click **Review + Create**, then **Create**.
5. Copy the **Public IP Address** of your VM once created.

---

### Step 2: Set Up Namecheap Domain + Cloudflare (Free SSL + HTTPS)

1. Go to [Cloudflare](https://dash.cloudflare.com) (Create a free account if needed).
2. Click **Add a Site** and enter your Namecheap domain (e.g., `clipmind.me`). Select the **Free Plan**.
3. Cloudflare will give you 2 **Nameservers** (e.g., `dash.ns.cloudflare.com`, `eric.ns.cloudflare.com`).
4. Open [Namecheap Dashboard](https://ap.namecheap.com):
   - Go to **Domain List** -> Click **Manage** next to your domain.
   - Under **Nameservers**, change to **Custom DNS** and paste the 2 Cloudflare Nameservers. Save changes.
5. In Cloudflare DNS Settings:
   - Add **A Record**: `Name: @` | `IPv4 address: <YOUR_AZURE_VM_PUBLIC_IP>` | `Proxy status: Proxied (Orange cloud)`
   - Add **A Record**: `Name: www` | `IPv4 address: <YOUR_AZURE_VM_PUBLIC_IP>` | `Proxy status: Proxied (Orange cloud)`
6. In Cloudflare **SSL/TLS** tab -> Set encryption mode to **Full** or **Flexible**.

---

### Step 3: Server Optimization & Deployment Script

Connect to your Azure VM via SSH:
```bash
ssh azureuser@<YOUR_AZURE_VM_PUBLIC_IP>
```

Run the following all-in-one setup commands on the VM:

```bash
# 1. Update server & install Docker + Git
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-plugin git curl

# 2. Add current user to docker group
sudo usermod -aG docker $USER
newgrp docker

# 3. Create 4GB Swap File (Crucial for FFmpeg / Python worker on 1GB RAM)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 4. Clone your repository
git clone https://github.com/<YOUR_GITHUB_USERNAME>/ai_video_editor.git
cd ai_video_editor

# 5. Create production environment file
cp backend/.env.production.example backend/.env.production
```

Now edit `backend/.env.production` using `nano`:
```bash
nano backend/.env.production
```
Make sure `POSTGRES_PASSWORD` and API keys (Groq, OpenAI, etc.) are populated.

---

### Step 4: Build and Launch the Application

Build and start all 5 docker containers (Frontend Nginx, Backend FastAPI, Worker, Postgres, Redis):

```bash
docker compose up -d --build
```

Verify everything is running:
```bash
docker compose ps
```

You should see:
- `db` (Postgres) - Up
- `redis` - Up
- `backend` - Up
- `worker` - Up
- `frontend` - Up (Port 80)

---

### Step 5: Test Your Live App

Open your browser and navigate to:
`https://yourdomain.com`

- ✅ Frontend loads with SSL (`https`).
- ✅ Fast API endpoint is proxied automatically under `https://yourdomain.com/api/v1/...`.
- ✅ Video upload & rendering runs smoothly backed by the swap memory!

---

## 🛡️ Maintenance & 10-Month Reliability Tips

1. **Auto-Restart on Server Reboot:**
   The `docker-compose.yml` already includes `restart: unless-stopped`, so if Azure restarts your VM for host maintenance, all containers will start automatically.

2. **Monitor Memory & Swap Usage:**
   Run `free -h` or `htop` to verify RAM & Swap usage.

3. **Updating Your App:**
   Whenever you push changes to GitHub, update the server with:
   ```bash
   cd ~/ai_video_editor
   git pull
   docker compose up -d --build
   ```

4. **Zero Cost Assurance:**
   Because `Standard_B1s` is eligible for 750 free hours every month for 12 months under Azure for Students, your Azure bill will remain **$0.00/month**. You can check your remaining credit at `https://www.microsoftazuresponsorships.com/`.
