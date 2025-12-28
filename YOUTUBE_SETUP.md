# 🔐 YouTube Data API v3 - OAuth2 Setup Guide

This guide will help you set up YouTube API authentication for automated video uploads.

## 📋 Prerequisites

- Google Account (same one you use for your YouTube channel)
- YouTube channel already created
- ~15 minutes for setup

## 🚀 Step-by-Step Setup

### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **"Select a project"** → **"New Project"**
3. Enter project name: `chess-youtube-shorts`
4. Click **"Create"**
5. Wait for project creation (30 seconds)

### Step 2: Enable YouTube Data API v3

1. In Cloud Console, go to **"APIs & Services"** → **"Library"**
2. Search for `YouTube Data API v3`
3. Click on it → Click **"Enable"**
4. Wait for activation (~10 seconds)

### Step 3: Configure OAuth Consent Screen

1. Go to **"APIs & Services"** → **"OAuth consent screen"**
2. Select **"External"** user type → Click **"Create"**
3. Fill in the form:
   - **App name**: `Chess Shorts Uploader`
   - **User support email**: Your email
   - **Developer contact email**: Your email
4. Click **"Save and Continue"**

5. **Scopes page**: Click **"Add or Remove Scopes"**

   - Search for `youtube.upload`
   - Check: `https://www.googleapis.com/auth/youtube.upload`
   - Click **"Update"** → **"Save and Continue"**

6. **Test users page**: Click **"Add Users"**

   - Add your Gmail address (the one with the YouTube channel)
   - Click **"Save and Continue"**

7. Click **"Back to Dashboard"**

### Step 4: Create OAuth2 Credentials

1. Go to **"APIs & Services"** → **"Credentials"**
2. Click **"+ Create Credentials"** → **"OAuth client ID"**
3. Select **"Desktop application"**
4. Name: `Chess Shorts Desktop Client`
5. Click **"Create"**

6. **⬇️ Download the JSON file** (click the download button)
7. **Rename it to `client_secrets.json`**
8. **Move it to your project folder**:
   ```bash
   mv ~/Downloads/client_secret_*.json /Users/waditya/chess-videos-automation/client_secrets.json
   ```

### Step 5: Verify Setup

Your `client_secrets.json` should look like this:

```json
{
  "installed": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "chess-youtube-shorts",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uris": ["http://localhost"]
  }
}
```

## 🔑 First-Time Authentication

Run the authentication script:

```bash
cd /Users/waditya/chess-videos-automation
source venv/bin/activate
python youtube_uploader.py --auth-only
```

This will:

1. Open a browser window
2. Ask you to sign in to Google
3. Grant permissions to the app
4. Save tokens to `token.json` (auto-refreshes)

## 📊 API Quota Information

YouTube Data API v3 has a **10,000 quota units/day** limit (free).

| Operation       | Quota Cost   | Max/Day      |
| --------------- | ------------ | ------------ |
| Video Upload    | ~1,600 units | 6 videos     |
| Update Metadata | 50 units     | 200 updates  |
| List Videos     | 1 unit       | 10,000 calls |

**Your daily upload limit: 6 videos/day**

## 🔒 Security Notes

⚠️ **NEVER commit these files to Git:**

- `client_secrets.json` (OAuth credentials)
- `token.json` (access tokens)

These are already in `.gitignore`.

## 🔄 Token Refresh

Tokens automatically refresh when:

- Access token expires (every ~1 hour)
- The upload script detects expired credentials

If you get authentication errors:

```bash
# Delete old token and re-authenticate
rm token.json
python youtube_uploader.py --auth-only
```

## ❓ Troubleshooting

### "Access blocked: This app's request is invalid"

- Make sure you added yourself as a test user in OAuth consent screen

### "quota exceeded"

- Wait until midnight Pacific Time (quota resets daily)
- Or request quota increase from Google (requires review)

### "invalid_grant" error

- Delete `token.json` and re-authenticate
- Check if your Google account is active

### "redirect_uri_mismatch"

- Make sure you selected "Desktop application" type
- Re-download `client_secrets.json`

## ✅ Setup Checklist

- [ ] Created Google Cloud project
- [ ] Enabled YouTube Data API v3
- [ ] Configured OAuth consent screen
- [ ] Added yourself as test user
- [ ] Created OAuth credentials (Desktop app)
- [ ] Downloaded and renamed `client_secrets.json`
- [ ] Moved file to project folder
- [ ] Ran first authentication

Once complete, you can upload videos automatically! 🎬
