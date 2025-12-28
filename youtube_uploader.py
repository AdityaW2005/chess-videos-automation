#!/usr/bin/env python3
"""
YouTube Shorts Uploader with OAuth2 Authentication
===================================================
Uploads chess videos to YouTube as Shorts with scheduled publishing.

Features:
- OAuth2 authentication with automatic token refresh
- Scheduled publishing (set publish time in future)
- Auto-generated titles and descriptions
- Quota-aware (tracks daily usage)
- Hashtags for discoverability

YouTube Data API v3 Quota:
- 10,000 units/day (free tier)
- Video upload: ~1,600 units
- Maximum uploads: ~6 videos/day

Author: Chess Shorts Automation Project
"""

import os
import sys
import json
import logging
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import pytz

# Google API imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# Import configuration
try:
    from config import (
        CHESS_COM_USERNAME,
        VIDEO_OUTPUT_DIR,
        TIMEZONE,
        PUBLISH_TIME
    )
except ImportError:
    CHESS_COM_USERNAME = "ChessPlayer"
    VIDEO_OUTPUT_DIR = "./videos"
    TIMEZONE = "Asia/Kolkata"
    PUBLISH_TIME = "18:00"

# =============================================================================
# CONSTANTS
# =============================================================================

# OAuth2 scopes required for video upload
SCOPES = [
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/youtube.readonly'
]

# File paths for OAuth credentials
CLIENT_SECRETS_FILE = Path(__file__).parent / "client_secrets.json"
TOKEN_FILE = Path(__file__).parent / "token.json"
QUOTA_FILE = Path(__file__).parent / "quota_usage.json"

# YouTube API quota costs
QUOTA_COSTS = {
    'video_upload': 1600,
    'video_update': 50,
    'video_list': 1,
    'channel_list': 1
}

DAILY_QUOTA_LIMIT = 10000

# Video categories (Gaming = 20)
CATEGORY_GAMING = "20"

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("YouTubeUploader")

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VideoMetadata:
    """Metadata for a YouTube video upload."""
    title: str
    description: str
    tags: List[str]
    category_id: str = CATEGORY_GAMING
    privacy_status: str = "private"  # private, public, unlisted
    publish_at: Optional[datetime] = None  # For scheduled publishing
    made_for_kids: bool = False
    
    def to_youtube_body(self) -> Dict[str, Any]:
        """Convert to YouTube API request body format."""
        body = {
            "snippet": {
                "title": self.title[:100],  # YouTube limit
                "description": self.description[:5000],  # YouTube limit
                "tags": self.tags[:500],  # YouTube limit
                "categoryId": self.category_id
            },
            "status": {
                "privacyStatus": self.privacy_status,
                "selfDeclaredMadeForKids": self.made_for_kids
            }
        }
        
        # Add scheduled publish time if provided
        if self.publish_at and self.privacy_status == "private":
            # YouTube requires ISO 8601 format with UTC
            publish_time_utc = self.publish_at.astimezone(pytz.UTC)
            body["status"]["publishAt"] = publish_time_utc.strftime("%Y-%m-%dT%H:%M:%S.0Z")
            # For scheduled videos, YouTube requires privacyStatus to be "private"
            # The video will automatically become public at publishAt time
        
        return body

# =============================================================================
# QUOTA TRACKER
# =============================================================================

class QuotaTracker:
    """Tracks YouTube API quota usage to prevent exceeding limits."""
    
    def __init__(self, quota_file: Path = QUOTA_FILE):
        self.quota_file = quota_file
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Load quota data from file."""
        if self.quota_file.exists():
            try:
                with open(self.quota_file, 'r') as f:
                    data = json.load(f)
                # Reset if it's a new day (Pacific Time - YouTube's quota reset)
                pacific = pytz.timezone('US/Pacific')
                today = datetime.now(pacific).date().isoformat()
                if data.get('date') != today:
                    return {'date': today, 'used': 0, 'operations': []}
                return data
            except (json.JSONDecodeError, KeyError):
                pass
        
        pacific = pytz.timezone('US/Pacific')
        return {
            'date': datetime.now(pacific).date().isoformat(),
            'used': 0,
            'operations': []
        }
    
    def _save(self):
        """Save quota data to file."""
        with open(self.quota_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def can_upload(self) -> bool:
        """Check if we have enough quota for a video upload."""
        return (self.data['used'] + QUOTA_COSTS['video_upload']) <= DAILY_QUOTA_LIMIT
    
    def record_usage(self, operation: str, cost: int):
        """Record quota usage for an operation."""
        self.data['used'] += cost
        self.data['operations'].append({
            'operation': operation,
            'cost': cost,
            'time': datetime.now().isoformat()
        })
        self._save()
        logger.info(f"📊 Quota used: {self.data['used']}/{DAILY_QUOTA_LIMIT} units")
    
    def get_remaining(self) -> int:
        """Get remaining quota units."""
        return DAILY_QUOTA_LIMIT - self.data['used']
    
    def get_uploads_remaining(self) -> int:
        """Get number of video uploads remaining today."""
        return self.get_remaining() // QUOTA_COSTS['video_upload']

# =============================================================================
# YOUTUBE AUTHENTICATOR
# =============================================================================

class YouTubeAuthenticator:
    """
    Handles OAuth2 authentication for YouTube Data API.
    
    Features:
    - First-time authentication via browser
    - Automatic token refresh
    - Secure token storage
    """
    
    def __init__(
        self,
        client_secrets_file: Path = CLIENT_SECRETS_FILE,
        token_file: Path = TOKEN_FILE
    ):
        self.client_secrets_file = client_secrets_file
        self.token_file = token_file
        self.credentials: Optional[Credentials] = None
    
    def authenticate(self, force_new: bool = False) -> Credentials:
        """
        Authenticate with YouTube API.
        
        Args:
            force_new: If True, force new authentication even if token exists
            
        Returns:
            Valid Google credentials
        """
        # Check for existing token
        if not force_new and self.token_file.exists():
            logger.info("🔑 Loading existing credentials...")
            self.credentials = self._load_token()
            
            # Check if token is valid
            if self.credentials and self.credentials.valid:
                logger.info("✅ Credentials valid")
                return self.credentials
            
            # Try to refresh expired token
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                logger.info("🔄 Refreshing expired token...")
                try:
                    self.credentials.refresh(Request())
                    self._save_token()
                    logger.info("✅ Token refreshed successfully")
                    return self.credentials
                except Exception as e:
                    logger.warning(f"⚠️ Token refresh failed: {e}")
                    # Fall through to new authentication
        
        # Perform new authentication
        logger.info("🌐 Starting new OAuth2 authentication...")
        return self._new_authentication()
    
    def _new_authentication(self) -> Credentials:
        """Perform new OAuth2 authentication via browser."""
        if not self.client_secrets_file.exists():
            logger.error(f"❌ Client secrets file not found: {self.client_secrets_file}")
            logger.error("   Please follow YOUTUBE_SETUP.md to create OAuth credentials")
            raise FileNotFoundError(f"Missing {self.client_secrets_file}")
        
        flow = InstalledAppFlow.from_client_secrets_file(
            str(self.client_secrets_file),
            scopes=SCOPES
        )
        
        # Run local server for OAuth callback
        logger.info("📱 Opening browser for authentication...")
        logger.info("   (If browser doesn't open, check terminal for URL)")
        
        self.credentials = flow.run_local_server(
            port=8080,
            prompt='consent',
            success_message='Authentication successful! You can close this window.'
        )
        
        # Save token for future use
        self._save_token()
        logger.info("✅ Authentication successful!")
        logger.info(f"💾 Token saved to: {self.token_file}")
        
        return self.credentials
    
    def _load_token(self) -> Optional[Credentials]:
        """Load credentials from token file."""
        try:
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
            
            return Credentials(
                token=token_data.get('token'),
                refresh_token=token_data.get('refresh_token'),
                token_uri=token_data.get('token_uri'),
                client_id=token_data.get('client_id'),
                client_secret=token_data.get('client_secret'),
                scopes=token_data.get('scopes')
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"⚠️ Failed to load token: {e}")
            return None
    
    def _save_token(self):
        """Save credentials to token file."""
        if not self.credentials:
            return
        
        token_data = {
            'token': self.credentials.token,
            'refresh_token': self.credentials.refresh_token,
            'token_uri': self.credentials.token_uri,
            'client_id': self.credentials.client_id,
            'client_secret': self.credentials.client_secret,
            'scopes': list(self.credentials.scopes) if self.credentials.scopes else SCOPES
        }
        
        with open(self.token_file, 'w') as f:
            json.dump(token_data, f, indent=2)
    
    def get_service(self):
        """Get authenticated YouTube API service."""
        if not self.credentials:
            self.authenticate()
        
        return build('youtube', 'v3', credentials=self.credentials)

# =============================================================================
# VIDEO METADATA GENERATOR
# =============================================================================

class MetadataGenerator:
    """Generates engaging titles and descriptions for chess videos."""
    
    # Title templates based on game outcome
    TITLE_TEMPLATES_WIN = [
        "♟️ {opening} - Crushing Win vs {rating}!",
        "🔥 {rating} Rated Player DESTROYED | {opening}",
        "Chess Blitz WIN: {opening} 🏆",
        "I Beat a {rating}! | {opening} Blitz",
        "⚡ {opening} Masterclass vs {rating}",
    ]
    
    TITLE_TEMPLATES_LOSS = [
        "♟️ Learning from Defeat: {opening}",
        "Where Did I Go Wrong? | {opening} Analysis",
        "Chess Blitz vs {rating} - Close Game!",
        "💡 Instructive Loss: {opening}",
    ]
    
    DESCRIPTION_TEMPLATE = """
{emoji} {result_text} in {time_control} Blitz on Chess.com!

🎮 Opening: {opening}
⚔️ Opponent: {opponent} ({opponent_rating})
👤 My Rating: {my_rating}

{game_summary}

📺 Subscribe for daily chess content!

#Chess #ChessShorts #Blitz #ChessCom {hashtags}

🔗 Play me on Chess.com: https://www.chess.com/member/{username}
"""
    
    HASHTAGS = [
        "#ChessOpening", "#ChessStrategy", "#ChessTactics",
        "#ChessGame", "#ChessPlayer", "#ChessMoves",
        "#OnlineChess", "#ChessLife", "#LearnChess"
    ]
    
    def __init__(self, username: str = CHESS_COM_USERNAME):
        self.username = username
    
    def generate(
        self,
        opening: str,
        opponent_name: str,
        opponent_rating: int,
        player_rating: int,
        player_won: bool,
        time_control: str,
        player_color: str
    ) -> VideoMetadata:
        """
        Generate video metadata from game information.
        
        Returns:
            VideoMetadata object ready for upload
        """
        import random
        
        # Select appropriate title template
        if player_won:
            template = random.choice(self.TITLE_TEMPLATES_WIN)
            emoji = "🏆"
            result_text = "WIN"
        else:
            template = random.choice(self.TITLE_TEMPLATES_LOSS)
            emoji = "📚"
            result_text = "Educational Loss"
        
        # Generate title
        title = template.format(
            opening=opening[:30] if len(opening) > 30 else opening,
            rating=opponent_rating
        )
        
        # Generate game summary
        rating_diff = opponent_rating - player_rating
        if rating_diff > 100:
            game_summary = f"Faced a higher-rated opponent (+{rating_diff} rating difference)!"
        elif rating_diff < -100:
            game_summary = f"Expected win against lower-rated opponent."
        else:
            game_summary = "Evenly matched game with exciting tactics!"
        
        # Add color info
        game_summary += f" Playing as {'White' if player_color == 'white' else 'Black'}."
        
        # Select random hashtags
        extra_hashtags = " ".join(random.sample(self.HASHTAGS, 3))
        
        # Generate description
        description = self.DESCRIPTION_TEMPLATE.format(
            emoji=emoji,
            result_text=result_text,
            time_control=time_control,
            opening=opening,
            opponent=opponent_name,
            opponent_rating=opponent_rating,
            my_rating=player_rating,
            game_summary=game_summary,
            username=self.username,
            hashtags=extra_hashtags
        )
        
        # Generate tags
        tags = [
            "chess", "chess shorts", "blitz chess", "chess.com",
            "chess opening", opening.lower().replace(" ", ""),
            "chess game", "online chess", "chess tactics",
            self.username.lower(), "chess tutorial"
        ]
        
        return VideoMetadata(
            title=title,
            description=description,
            tags=tags,
            category_id=CATEGORY_GAMING,
            privacy_status="private",  # Will be scheduled
            made_for_kids=False
        )

# =============================================================================
# YOUTUBE UPLOADER
# =============================================================================

class YouTubeUploader:
    """
    Main class for uploading videos to YouTube.
    
    Features:
    - OAuth2 authentication with auto-refresh
    - Scheduled publishing
    - Resumable uploads for large files
    - Quota tracking
    - Retry logic for transient errors
    """
    
    def __init__(self):
        self.authenticator = YouTubeAuthenticator()
        self.quota_tracker = QuotaTracker()
        self.metadata_generator = MetadataGenerator()
        self.youtube = None
    
    def authenticate(self, force_new: bool = False) -> bool:
        """
        Authenticate with YouTube API.
        
        Args:
            force_new: Force new authentication
            
        Returns:
            True if authentication successful
        """
        try:
            self.authenticator.authenticate(force_new)
            self.youtube = self.authenticator.get_service()
            
            # Verify by getting channel info
            response = self.youtube.channels().list(
                part="snippet",
                mine=True
            ).execute()
            
            if response.get('items'):
                channel = response['items'][0]['snippet']
                logger.info(f"✅ Authenticated as: {channel['title']}")
                self.quota_tracker.record_usage('channel_list', QUOTA_COSTS['channel_list'])
                return True
            else:
                logger.error("❌ No YouTube channel found for this account")
                return False
                
        except HttpError as e:
            logger.error(f"❌ Authentication failed: {e}")
            return False
    
    def upload_video(
        self,
        video_path: str,
        metadata: VideoMetadata,
        notify_subscribers: bool = False
    ) -> Optional[str]:
        """
        Upload a video to YouTube.
        
        Args:
            video_path: Path to video file
            metadata: Video metadata
            notify_subscribers: Whether to notify subscribers
            
        Returns:
            Video ID if successful, None otherwise
        """
        video_path = Path(video_path)
        
        # Validate file exists
        if not video_path.exists():
            logger.error(f"❌ Video file not found: {video_path}")
            return None
        
        # Check quota
        if not self.quota_tracker.can_upload():
            logger.error("❌ Daily quota exceeded! Cannot upload.")
            logger.error(f"   Remaining: {self.quota_tracker.get_remaining()} units")
            logger.error("   Quota resets at midnight Pacific Time")
            return None
        
        # Ensure authenticated
        if not self.youtube:
            if not self.authenticate():
                return None
        
        logger.info(f"📤 Uploading: {video_path.name}")
        logger.info(f"   Title: {metadata.title}")
        
        # Prepare request body
        body = metadata.to_youtube_body()
        
        # Add notify subscribers setting
        body["status"]["notifySubscribers"] = notify_subscribers
        
        # Create media upload object with resumable upload
        media = MediaFileUpload(
            str(video_path),
            mimetype='video/mp4',
            resumable=True,
            chunksize=1024 * 1024  # 1MB chunks
        )
        
        try:
            # Insert video
            request = self.youtube.videos().insert(
                part=",".join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Execute with progress tracking
            response = None
            retry_count = 0
            max_retries = 3
            
            while response is None:
                try:
                    logger.info("   Uploading...")
                    status, response = request.next_chunk()
                    
                    if status:
                        progress = int(status.progress() * 100)
                        logger.info(f"   Progress: {progress}%")
                        
                except HttpError as e:
                    if e.resp.status in [500, 502, 503, 504] and retry_count < max_retries:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        logger.warning(f"   Retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise
            
            if response:
                video_id = response['id']
                logger.info(f"✅ Upload successful!")
                logger.info(f"   Video ID: {video_id}")
                logger.info(f"   URL: https://youtube.com/shorts/{video_id}")
                
                # Record quota usage
                self.quota_tracker.record_usage('video_upload', QUOTA_COSTS['video_upload'])
                
                # Log scheduled time if set
                if metadata.publish_at:
                    logger.info(f"   Scheduled for: {metadata.publish_at.strftime('%Y-%m-%d %H:%M %Z')}")
                
                return video_id
            
        except HttpError as e:
            error_content = json.loads(e.content.decode('utf-8'))
            error_reason = error_content.get('error', {}).get('errors', [{}])[0].get('reason', 'unknown')
            
            logger.error(f"❌ Upload failed: {error_reason}")
            logger.error(f"   Details: {e}")
            
            if error_reason == 'quotaExceeded':
                logger.error("   Daily quota exceeded! Try again tomorrow.")
            elif error_reason == 'uploadLimitExceeded':
                logger.error("   Upload limit reached for this video.")
            
            return None
        
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return None
    
    def upload_chess_video(
        self,
        video_path: str,
        game_json_path: str,
        publish_time: Optional[datetime] = None
    ) -> Optional[str]:
        """
        Upload a chess video with auto-generated metadata.
        
        Args:
            video_path: Path to video file
            game_json_path: Path to game JSON metadata
            publish_time: Scheduled publish time (optional)
            
        Returns:
            Video ID if successful
        """
        # Load game metadata
        try:
            with open(game_json_path, 'r') as f:
                game_data = json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load game data: {e}")
            return None
        
        # Generate metadata
        metadata = self.metadata_generator.generate(
            opening=game_data.get('opening', 'Chess Game'),
            opponent_name=game_data.get('white_player') if game_data.get('player_color') == 'black' 
                         else game_data.get('black_player'),
            opponent_rating=game_data.get('opponent_rating', 1500),
            player_rating=game_data.get('player_rating', 1500),
            player_won=game_data.get('player_won', False),
            time_control=game_data.get('time_control', 'Blitz'),
            player_color=game_data.get('player_color', 'white')
        )
        
        # Set publish time
        if publish_time:
            metadata.publish_at = publish_time
            metadata.privacy_status = "private"  # Required for scheduling
        
        return self.upload_video(video_path, metadata)
    
    def get_scheduled_publish_time(self, hour: int = 18, minute: int = 0) -> datetime:
        """
        Get the next scheduled publish time.
        
        Args:
            hour: Hour to publish (24-hour format)
            minute: Minute to publish
            
        Returns:
            datetime object for next publish time
        """
        tz = pytz.timezone(TIMEZONE)
        now = datetime.now(tz)
        
        # Create publish time for today
        publish = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # If time has passed, schedule for tomorrow
        if publish <= now:
            publish = publish + timedelta(days=1)
        
        logger.info(f"📅 Scheduled publish time: {publish.strftime('%Y-%m-%d %H:%M %Z')}")
        return publish
    
    def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota usage status."""
        return {
            'used': self.quota_tracker.data['used'],
            'remaining': self.quota_tracker.get_remaining(),
            'uploads_remaining': self.quota_tracker.get_uploads_remaining(),
            'reset_date': self.quota_tracker.data['date']
        }

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line interface for YouTube uploader."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload chess videos to YouTube Shorts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Authenticate only (first-time setup)
  python youtube_uploader.py --auth-only
  
  # Upload a video immediately (unlisted)
  python youtube_uploader.py video.mp4 --game-json game.json
  
  # Upload with scheduled publishing at 6 PM IST
  python youtube_uploader.py video.mp4 --game-json game.json --schedule
  
  # Upload with custom title
  python youtube_uploader.py video.mp4 --title "My Chess Win!" --description "Great game"
  
  # Check quota status
  python youtube_uploader.py --quota
        """
    )
    
    parser.add_argument(
        'video_file',
        nargs='?',
        help='Path to video file to upload'
    )
    
    parser.add_argument(
        '--auth-only',
        action='store_true',
        help='Only authenticate (no upload)'
    )
    
    parser.add_argument(
        '--force-auth',
        action='store_true',
        help='Force new authentication'
    )
    
    parser.add_argument(
        '--game-json', '-g',
        help='Path to game JSON metadata file'
    )
    
    parser.add_argument(
        '--title', '-t',
        help='Custom video title'
    )
    
    parser.add_argument(
        '--description', '-d',
        help='Custom video description'
    )
    
    parser.add_argument(
        '--schedule', '-s',
        action='store_true',
        help='Schedule video for 6 PM IST'
    )
    
    parser.add_argument(
        '--schedule-time',
        help='Custom schedule time (HH:MM format)'
    )
    
    parser.add_argument(
        '--public',
        action='store_true',
        help='Make video public immediately'
    )
    
    parser.add_argument(
        '--quota', '-q',
        action='store_true',
        help='Show quota status'
    )
    
    args = parser.parse_args()
    
    uploader = YouTubeUploader()
    
    # Check quota
    if args.quota:
        status = uploader.get_quota_status()
        print("\n📊 YouTube API Quota Status:")
        print(f"   Used today: {status['used']}/{DAILY_QUOTA_LIMIT} units")
        print(f"   Remaining: {status['remaining']} units")
        print(f"   Uploads remaining: {status['uploads_remaining']} videos")
        print(f"   Resets: Midnight Pacific Time")
        return 0
    
    # Auth only
    if args.auth_only:
        print("\n🔐 YouTube OAuth2 Authentication")
        print("=" * 40)
        
        if not CLIENT_SECRETS_FILE.exists():
            print(f"\n❌ Error: {CLIENT_SECRETS_FILE} not found!")
            print("\n📋 To set up authentication:")
            print("   1. Read YOUTUBE_SETUP.md for detailed instructions")
            print("   2. Create OAuth credentials in Google Cloud Console")
            print("   3. Download client_secrets.json to this folder")
            return 1
        
        success = uploader.authenticate(force_new=args.force_auth)
        return 0 if success else 1
    
    # Upload video
    if not args.video_file:
        parser.print_help()
        return 1
    
    # Authenticate
    if not uploader.authenticate(force_new=args.force_auth):
        return 1
    
    # Determine publish time
    publish_time = None
    if args.schedule:
        publish_time = uploader.get_scheduled_publish_time(18, 0)
    elif args.schedule_time:
        try:
            hour, minute = map(int, args.schedule_time.split(':'))
            publish_time = uploader.get_scheduled_publish_time(hour, minute)
        except ValueError:
            logger.error("Invalid time format. Use HH:MM (e.g., 18:00)")
            return 1
    
    # Upload with game JSON
    if args.game_json:
        video_id = uploader.upload_chess_video(
            args.video_file,
            args.game_json,
            publish_time
        )
    else:
        # Manual metadata
        metadata = VideoMetadata(
            title=args.title or "Chess Shorts Video",
            description=args.description or "Watch my chess game!",
            tags=["chess", "chess shorts", "blitz"],
            privacy_status="public" if args.public else "private",
            publish_at=publish_time
        )
        video_id = uploader.upload_video(args.video_file, metadata)
    
    if video_id:
        print(f"\n🎬 Video uploaded successfully!")
        print(f"   Video ID: {video_id}")
        print(f"   URL: https://youtube.com/shorts/{video_id}")
        if publish_time:
            print(f"   Scheduled: {publish_time.strftime('%Y-%m-%d %H:%M %Z')}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
