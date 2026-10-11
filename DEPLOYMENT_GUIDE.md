# 🚀 How to Deploy AURA-GRID AI Alert System to Vercel

This web application runs the complete TinyML Neural Network directly on the client side using the real mathematical weights trained on the **1,750-sample high-voltage insulator research paper dataset**.

Because it runs in standard JavaScript/HTML/CSS without requiring heavy Python containers or server cold-starts, it deploys to **Vercel in under 60 seconds with 100% free hosting and zero configuration!**

---

## ⚡ Method 1: Deploy with Vercel CLI (Fastest - 1 Command)

If you have Node.js installed on your computer:

1. Open PowerShell and navigate to the `webapp` folder:
   ```powershell
   cd C:\Users\Admin\.gemini\antigravity\scratch\TinyML_Fault_Detection\webapp
   ```

2. Run Vercel deploy:
   ```powershell
   npx vercel
   ```

3. Follow the simple prompts:
   - *Set up and deploy?* Press **Y**
   - *Which scope?* Select your Vercel account
   - *Link to existing project?* **N**
   - *Project name?* Enter `auragrid-fault-alert` (or press Enter)
   - *Directory?* `./` (press Enter)
   - *Want to modify settings?* **N**

4. **Done!** Vercel will output your live URL:
   `https://auragrid-fault-alert.vercel.app`

---

## 🌐 Method 2: Deploy via GitHub (Recommended for CI/CD)

1. Create a new GitHub repository (e.g., `auragrid-tinyml-alert-system`).
2. Push the contents of the `webapp` folder to your repository:
   ```bash
   git init
   git add .
   git commit -m "Deploy AURA-GRID TinyML Alert System"
   git branch -M main
   git remote add origin https://github.com/<your-username>/auragrid-tinyml-alert-system.git
   git push -u origin main
   ```
3. Go to [vercel.com/dashboard](https://vercel.com/dashboard).
4. Click **"Add New..."** &rarr; **"Project"**.
5. Click **"Import"** next to your GitHub repository.
6. Leave all Build/Output settings as default (Framework Preset: **Other** / Static).
7. Click **"Deploy"**!

---

## 💻 Method 3: Run Locally on your PC

To test the alert dashboard locally on your machine:

### Option A (Using Python):
```powershell
cd C:\Users\Admin\.gemini\antigravity\scratch\TinyML_Fault_Detection\webapp
python -m http.server 3000
```
Then open your browser at: **`http://localhost:3000`**

### Option B (Using Node.js):
```powershell
cd C:\Users\Admin\.gemini\antigravity\scratch\TinyML_Fault_Detection\webapp
npx serve .
```

---

## 🎯 Key Webpage Features Included:
- **Real-Time Alert Banner**: Live transitions between Normal (Green), Advisory (Yellow), Warning (Orange), and Critical Flashover (Red).
- **Web Audio Alarm Siren**: Built-in audio alerts (beeps for warnings, pulsing two-tone siren for flashovers).
- **8 Sensor Telemetry Gauges**: Real-time monitoring of Voltage, Leakage Current, Temperature, Humidity, Pollution Index, Harmonics THD, Phase Angle, and Partial Discharges.
- **Research Dataset Presets**: One-click injection of real samples (Healthy, Light Pol, Moderate, Heavy Po, Wet Contamination, Surface Cracking, Flashover).
- **Hardware Safety Interlock**: Deterministic threshold override (trips alarm if leakage > 25 mA or voltage > 14 kV).
- **Interactive Tuning Sliders**: Manually tweak any sensor to see AI classifications in sub-millisecond real time.
- **Telemetry Streamer**: Toggle automated streaming to simulate continuous live ESP32 wireless telemetry.
- **Incident Audit Trail**: Table recording all classified events with timestamp, confidence, and CSV Export.
