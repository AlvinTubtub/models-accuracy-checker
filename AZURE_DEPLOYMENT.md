# Azure for Students Deployment & Operations Guide

This guide details how to run the **PSE Model Accuracy Recheck** and **Exact Model Training Pipeline** on an **Azure for Students** Virtual Machine, hosting the **Streamlit Dashboard** online with **manual, on-demand execution** (no automated cron jobs).

---

## 1. Architecture & Azure for Students Tier Usage

Your Azure for Students subscription includes:
* **750 hours/month** of free B-series VMs (`B2ats v2` or `B1s`), which covers 24/7 uptime for an entire month at **\$0 cost**.
* **1,500 hours/month** of free Public IP addresses.
* **15 GB/month** of outbound internet bandwidth.
* **\$100 annual student credits** (available if you ever want to temporarily burst to a larger 4-core / 16GB RAM VM for ultra-fast training).

---

## 2. Step-by-Step Azure VM Provisioning

### Step 2.1: Create Virtual Machine in Azure Portal
1. Navigate to [portal.azure.com](https://portal.azure.com) and search for **Virtual machines**.
2. Click **Create** > **Azure virtual machine**.
3. Configure the following:
   * **Subscription**: `Azure for Students`
   * **Resource group**: Click *Create new* (e.g., `rg-pse-recheck`)
   * **Virtual machine name**: `vm-pse-accuracy`
   * **Region**: Select a nearby region (e.g., `East US`, `Southeast Asia`)
   * **Security type**: Standard
   * **Image**: `Ubuntu Server 24.04 LTS - x64 Gen2`
   * **Size**: Select `Standard_B2ats_v2` (2 vCPUs, 1 GiB RAM) or `Standard_B1s` (both eligible under your 750 free monthly hours).
4. **Administrator account**:
   * Choose **SSH public key** (recommended) or **Password**.
   * Username: `azureuser`
5. **Inbound port rules**:
   * Allow: `SSH (22)`

### Step 2.2: Configure Network Security Group (Open Dashboard Port)
1. Once the VM is created, navigate to **Networking** in the VM's left-hand menu.
2. Under **Inbound port rules**, click **Add inbound port rule**:
   * **Source**: `Any`
   * **Source port ranges**: `*`
   * **Destination**: `Any`
   * **Service**: Custom
   * **Destination port ranges**: `8501` (or `80` if using standard HTTP)
   * **Protocol**: `TCP`
   * **Action**: `Allow`
   * **Priority**: `300`
   * **Name**: `Allow-Streamlit-8501`
3. Click **Add**.

---

## 3. Server Setup & 4 GB Swap Space

Because the free tier VM has 1 GiB of physical RAM, PyTorch and Statsmodels require additional virtual memory to prevent Out-Of-Memory (OOM) errors during training. Setting up a 4 GB swapfile solves this completely.

### Step 3.1: Connect via SSH
```bash
ssh azureuser@<YOUR_AZURE_PUBLIC_IP>
```

### Step 3.2: Configure 4 GB Swap
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify swap is active
free -h
```

### Step 3.3: Install System Dependencies
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git curl
```

---

## 4. Deploying the Codebase

### Step 4.1: Clone and Setup Virtual Environment
```bash
git clone <YOUR_NEW_REPO_URL> models-accuracy-checker
cd models-accuracy-checker

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4.2: Fetch Raw Historical Market Data
Download all 15 raw PSE CSVs directly to `data/raw/`:
```bash
python scripts/fetch_raw_data.py
```

---

## 5. Manual Model Training & Audit Workflow

You have complete manual control over when to train models and run audits. **No automated cron jobs or background scripts will run without your explicit command.**

### Train All 15 Companies (Exact Computations)
```bash
source .venv/bin/activate
python scripts/train_models.py --all
```

### Train a Single Company
```bash
python scripts/train_models.py --symbol ALI
```

### Fast Iteration Mode (Skip LSTM)
If you want quick ARIMA and LIR results without deep neural network epochs:
```bash
python scripts/train_models.py --all --skip-lstm
```

### Run the Independent Verification Audit (CLI)
Inspect model win-rates against the Naive benchmark and verify 100% metric agreement:
```bash
python scripts/run_recheck_cli.py --data-dir data/evaluations
```

---

## 6. Keeping Streamlit Online 24/7 (`systemd`)

To keep the Streamlit dashboard online even after you disconnect your SSH session, set up a simple `systemd` service.

### Step 6.1: Create Systemd Service File
```bash
sudo nano /etc/systemd/system/streamlit.service
```

Paste the following configuration (replace `azureuser` and `/home/azureuser` if your username differs):
```ini
[Unit]
Description=PSE Model Accuracy Recheck Streamlit Dashboard
After=network.target

[Service]
User=azureuser
WorkingDirectory=/home/azureuser/models-accuracy-checker
ExecStart=/home/azureuser/models-accuracy-checker/.venv/bin/streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Step 6.2: Enable and Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit

# Check status
sudo systemctl status streamlit
```

### Step 6.3: View Your Live Dashboard
Open your browser and navigate to:
```
http://<YOUR_AZURE_PUBLIC_IP>:8501
```

---

## 7. Useful Operations Commands

| Task | Command |
|---|---|
| Check Dashboard Service Status | `sudo systemctl status streamlit` |
| Restart Dashboard | `sudo systemctl restart streamlit` |
| View Dashboard Live Logs | `journalctl -u streamlit -f` |
| Run Unit Tests | `./.venv/bin/pytest -v` |
| Trigger Manual Training | `./.venv/bin/python scripts/train_models.py --all` |
| Trigger Manual Accuracy Audit | `./.venv/bin/python scripts/run_recheck_cli.py --data-dir data/evaluations` |
