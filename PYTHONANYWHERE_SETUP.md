# 🚀 PythonAnywhere Deployment Guide

Deploy your Chess Shorts automation to run daily for **FREE** on PythonAnywhere.

## 📋 Prerequisites

Before deploying, ensure locally:

- ✅ `python youtube_uploader.py --auth-only` works
- ✅ `token.json` exists (YouTube authentication)
- ✅ Test video generation works

## 🆓 PythonAnywhere Free Tier Limits

| Feature           | Free Tier                      |
| ----------------- | ------------------------------ |
| Scheduled Tasks   | **1 task**                     |
| CPU seconds/day   | 100                            |
| Storage           | 512 MB                         |
| Outbound Internet | ✅ Allowed (allowlisted sites) |

**Good news:** Chess.com and YouTube APIs are accessible on free tier!

---

## 📦 Step 1: Create PythonAnywhere Account

1. Go to [pythonanywhere.com](https://www.pythonanywhere.com/)
2. Click **"Start running Python online"** → **"Create a Beginner account"**
3. Sign up with your email
4. Verify your email

---

## 📁 Step 2: Upload Project Files

### Option A: Using Git (Recommended)

1. First, push your code to GitHub (without secrets!):

   ```bash
   cd /Users/waditya/chess-videos-automation
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/chess-videos-automation.git
   git push -u origin main
   ```

2. On PythonAnywhere, open a **Bash console**:

   - Dashboard → Consoles → Bash

3. Clone your repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/chess-videos-automation.git
   cd chess-videos-automation
   ```

### Option B: Manual Upload

1. Go to **Files** tab on PythonAnywhere
2. Create folder: `chess-videos-automation`
3. Upload these files:
   - `config.py`
   - `fetch_chess_games.py`
   - `generate_video.py`
   - `youtube_uploader.py`
   - `daily_automation.py`
   - `requirements.txt`

---

## 🔐 Step 3: Upload Secrets (IMPORTANT!)

**Never commit these to Git!** Upload manually:

1. Go to **Files** tab
2. Navigate to `chess-videos-automation/`
3. Click **Upload a file**
4. Upload from your local machine:
   - `token.json` (YouTube auth token)
   - `client_secrets.json` (OAuth credentials)

---

## 🐍 Step 4: Set Up Virtual Environment

In PythonAnywhere **Bash console**:

```bash
cd ~/chess-videos-automation

# Create virtual environment with Python 3.10
mkvirtualenv --python=/usr/bin/python3.10 chess-venv

# Activate it
workon chess-venv

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Step 5: Configure Environment Variables

PythonAnywhere doesn't have a `.env` file system, so we'll use a different approach.

### Create a secrets config file:

```bash
nano ~/chess-videos-automation/.secrets.py
```

Add:

```python
# DO NOT COMMIT THIS FILE
# YouTube OAuth paths (relative to project)
YOUTUBE_CLIENT_SECRETS = "client_secrets.json"
YOUTUBE_TOKEN_FILE = "token.json"
```

Save: `Ctrl+X`, then `Y`, then `Enter`

---

## ⏰ Step 6: Create Scheduled Task

This is where the magic happens!

1. Go to **Tasks** tab on PythonAnywhere
2. Under **Scheduled tasks**, find the time selector
3. Set time to **12:30 UTC** (= 6:00 PM IST)

   > 🕐 **Time Conversion:** IST = UTC + 5:30
   >
   > - 6:00 PM IST = 12:30 UTC
   > - 5:00 PM IST = 11:30 UTC

4. Enter this command:

   ```
   /home/YOUR_USERNAME/.virtualenvs/chess-venv/bin/python /home/YOUR_USERNAME/chess-videos-automation/daily_automation.py
   ```

   Replace `YOUR_USERNAME` with your PythonAnywhere username.

5. Click **Create**

---

## 🧪 Step 7: Test the Setup

### Test in Console:

```bash
cd ~/chess-videos-automation
workon chess-venv

# Test fetch only (quick test)
python daily_automation.py --fetch-only

# Test full pipeline (dry run - no upload)
python daily_automation.py --dry-run

# Test full pipeline (actual upload)
python daily_automation.py
```

### Check Logs:

```bash
cat ~/chess-videos-automation/logs/automation_$(date +%Y%m%d).log
```

---

## 📊 Step 8: Monitor Your Task

### View Task Logs:

1. Go to **Tasks** tab
2. Click on your task
3. View **stdout** and **stderr** logs

### Check Quota:

```bash
cd ~/chess-videos-automation
workon chess-venv
python youtube_uploader.py --quota
```

---

## 🔄 Updating Your Code

When you make changes locally:

### If using Git:

```bash
# Local machine
git add .
git commit -m "Update"
git push

# On PythonAnywhere console
cd ~/chess-videos-automation
git pull
```

### If manual upload:

Just re-upload the changed files via the Files tab.

---

## 🛠️ Troubleshooting

### "No module named X"

```bash
workon chess-venv
pip install module_name
```

### "Token refresh failed"

Your YouTube token may have expired. Re-authenticate locally:

```bash
# On your local machine
rm token.json
python youtube_uploader.py --auth-only
```

Then re-upload `token.json` to PythonAnywhere.

### "CPU quota exceeded"

Free tier has 100 CPU seconds/day. Video generation is CPU-intensive.
Solutions:

- Reduce video length in `config.py`
- Upgrade to paid tier ($5/month)
- Run every other day instead

### "Task didn't run"

- Check the task time (UTC, not IST!)
- Ensure the command path is correct
- Check stderr log for errors

### Chess.com API blocked

PythonAnywhere free tier has allowlisted domains. Chess.com's API should work, but if not:

1. Go to **Account** → **API access**
2. Request access to `api.chess.com`

---

## 📁 Final File Structure on PythonAnywhere

```
/home/YOUR_USERNAME/
└── chess-videos-automation/
    ├── config.py
    ├── fetch_chess_games.py
    ├── generate_video.py
    ├── youtube_uploader.py
    ├── daily_automation.py
    ├── requirements.txt
    ├── client_secrets.json     # 🔐 Uploaded manually
    ├── token.json              # 🔐 Uploaded manually
    ├── games/                  # Created automatically
    ├── videos/                 # Created automatically
    └── logs/                   # Created automatically
```

---

## ✅ Deployment Checklist

- [ ] Created PythonAnywhere account
- [ ] Uploaded/cloned project files
- [ ] Uploaded `token.json` and `client_secrets.json` manually
- [ ] Created virtual environment with dependencies
- [ ] Created scheduled task at 12:30 UTC (6 PM IST)
- [ ] Tested with `--dry-run` successfully
- [ ] Tested full pipeline successfully

---

## 🎉 You're Done!

Your automation will now:

1. ⏰ Run daily at 6 PM IST
2. 📥 Fetch your best Blitz game of the day
3. 🎬 Generate a YouTube Shorts video
4. 📤 Upload to YouTube (scheduled or immediate)

Check your YouTube Studio the next day to see your video! 🚀

---

## 💡 Tips for Success

1. **Play some Blitz games daily** - The script needs games to work with!
2. **Check logs weekly** - Make sure everything is running smoothly
3. **Monitor quota** - You have ~6 uploads/day max
4. **Keep tokens fresh** - Re-authenticate if uploads fail

Happy automating! ♟️🎬
