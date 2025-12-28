# ♟️ Chess.com YouTube Shorts Automation

Automatically fetch your Chess.com Blitz games and create engaging YouTube Shorts.

## 🎯 Project Overview

This project automates the entire pipeline:

1. **Fetch** → Download your daily Chess.com Blitz games via their free public API
2. **Analyze** → Parse PGN, identify the most interesting games (wins, comebacks, blunders)
3. **Generate** → Create vertical 9:16 videos with board animation, move notation, and overlays
4. **Publish** → Upload to YouTube via API and schedule for 6 PM IST

## 📁 Project Structure

```
chess-videos-automation/
├── config.py               # Configuration settings (edit this first!)
├── fetch_chess_games.py    # Game fetching & parsing module
├── requirements.txt        # Python dependencies
├── games/                  # Downloaded games storage
│   ├── pgn/               # Raw PGN files
│   └── json/              # Parsed game metadata
├── videos/                 # Generated video files
└── logs/                   # Execution logs
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+ installed
- A Chess.com account with Blitz games played
- macOS/Linux/Windows with terminal access

### 2. Installation

```bash
# Clone or navigate to project directory
cd /Users/waditya/chess-videos-automation

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Edit `config.py` and set your Chess.com username:

```python
CHESS_COM_USERNAME = "your_actual_username"  # Replace this!
```

### 4. Run the Fetcher

```bash
# Fetch today's games
python fetch_chess_games.py

# Fetch specific date
python fetch_chess_games.py --date 2024-12-27

# Fetch for different user
python fetch_chess_games.py --user hikaru --date 2024-12-27
```

## 📊 Output

### PGN Files

Saved to `games/pgn/{date}/` with naming convention:

```
2024-12-28_blitz_win_12345678.pgn
```

### JSON Metadata

Saved to `games/json/{date}/` containing:

- Game URL
- Player colors and ratings
- Win/loss result
- Opening name
- Accuracy data (if available)

### Daily Summary

`games/json/{date}_summary.json` with aggregated stats.

## 🔌 Chess.com API Details

| Endpoint                                   | Purpose             | Rate Limit     |
| ------------------------------------------ | ------------------- | -------------- |
| `/pub/player/{username}`                   | Verify user exists  | None specified |
| `/pub/player/{username}/games/{YYYY}/{MM}` | Fetch monthly games | Be respectful  |

**Important Notes:**

- API is completely free, no authentication needed
- Set a descriptive User-Agent header (we do this automatically)
- Add delays between requests (1 second by default)

## 🛠️ Troubleshooting

### "User not found"

- Check spelling of your Chess.com username
- Username is case-insensitive

### "No games found"

- You haven't played any games on that date
- Check if you're filtering the right game type (Blitz vs Rapid vs Bullet)

### "Rate limited"

- Wait a few minutes before retrying
- Increase `API_DELAY_SECONDS` in config.py

### "Connection error"

- Check your internet connection
- Chess.com might be down (rare)

## 📅 Project Roadmap

- [x] **Phase 1**: Fetch Chess.com games via API
- [ ] **Phase 2**: Generate video from PGN (python-chess + FFmpeg)
- [ ] **Phase 3**: YouTube API integration & upload
- [ ] **Phase 4**: Daily automation (cron/scheduler)
- [ ] **Phase 5**: Add overlays (evaluation bar, captions, music)

## 💰 Cost

**$0** - This entire project uses only free services:

- Chess.com Public API: Free, unlimited
- YouTube Data API v3: 10,000 quota units/day (enough for ~6 uploads)
- FFmpeg: Open source, free
- Python libraries: All open source

## 📝 License

MIT License - Feel free to use and modify!

---

**Need help?** Open an issue or reach out!
