"""
Configuration file for Chess.com YouTube Shorts Automation
Update these settings with your information
"""

# =============================================================================
# CHESS.COM CONFIGURATION
# =============================================================================

# Your Chess.com username (case-insensitive)
CHESS_COM_USERNAME = "ChessBeast_37"  # TODO: Replace with your actual username

# Game type filters
GAME_TYPES = {
    "blitz": True,      # 3-10 minute games
    "rapid": False,     # 10-30 minute games  
    "bullet": False,    # < 3 minute games
    "daily": False      # Correspondence games
}

# =============================================================================
# FILE STORAGE CONFIGURATION
# =============================================================================

import os

# Base directory for storing games and videos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory to store fetched PGN files
PGN_OUTPUT_DIR = os.path.join(BASE_DIR, "games", "pgn")

# Directory to store processed games data (JSON)
JSON_OUTPUT_DIR = os.path.join(BASE_DIR, "games", "json")

# Directory for generated videos
VIDEO_OUTPUT_DIR = os.path.join(BASE_DIR, "videos")

# =============================================================================
# VIDEO GENERATION SETTINGS (for future use)
# =============================================================================

VIDEO_SETTINGS = {
    "width": 1080,
    "height": 1920,
    "fps": 30,
    "max_duration_seconds": 59,  # YouTube Shorts must be < 60 seconds
    "aspect_ratio": "9:16"
}

# Display name shown in videos (your brand name)
DISPLAY_NAME = "ChessBeast"

# =============================================================================
# SCHEDULING CONFIGURATION
# =============================================================================

# Time zone for scheduling (IST = UTC+5:30)
TIMEZONE = "Asia/Kolkata"

# Time to fetch games daily (24-hour format)
FETCH_TIME = "17:00"  # 5 PM IST

# Time to publish YouTube Short (24-hour format)
PUBLISH_TIME = "18:00"  # 6 PM IST

# =============================================================================
# API RATE LIMITING
# =============================================================================

# Chess.com API rate limit (be nice to their servers)
API_DELAY_SECONDS = 1  # Delay between API calls

# YouTube Data API v3 quota (10,000 units/day)
# Video upload = ~1600 units, so max 6 uploads/day
YOUTUBE_DAILY_UPLOAD_LIMIT = 6
