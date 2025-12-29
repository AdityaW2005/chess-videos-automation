#!/usr/bin/env python3
"""
Daily Chess Shorts Automation
==============================
Main orchestration script that:
1. Fetches today's best Chess.com game
2. Generates a YouTube Shorts video
3. Uploads to YouTube with scheduled publishing

Designed to run on PythonAnywhere free tier.

Usage:
    python daily_automation.py              # Run full pipeline
    python daily_automation.py --dry-run    # Test without uploading
    python daily_automation.py --fetch-only # Only fetch games
    python daily_automation.py --video-only # Only generate video (from latest game)

Author: Chess Shorts Automation Project
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# Add project root to path (for PythonAnywhere compatibility)
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import our modules
from config import (
    CHESS_COM_USERNAME,
    PGN_OUTPUT_DIR,
    JSON_OUTPUT_DIR,
    VIDEO_OUTPUT_DIR,
    TIMEZONE,
    PUBLISH_TIME
)
from fetch_chess_games import ChessComAPI, GameParser, GameFilter, GameStorage
from generate_video import ChessVideoGenerator
from youtube_uploader import YouTubeUploader, VideoMetadata

# Try to import enhanced generator (optional - falls back to basic)
try:
    from enhanced_video_generator import EnhancedVideoGenerator
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

import pytz

# =============================================================================
# LOGGING SETUP
# =============================================================================

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"automation_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DailyAutomation")

# =============================================================================
# GAME SELECTOR
# =============================================================================

class BestGameSelector:
    """Selects the best game to feature from today's games."""
    
    @staticmethod
    def score_game(game: Dict[str, Any]) -> float:
        """
        Score a game based on various factors.
        Higher score = better content potential.
        """
        score = 0.0
        
        # Prefer wins (viewers like winning content)
        if game.get('player_won'):
            score += 50
        
        # Bonus for beating higher-rated opponents
        rating_diff = game.get('opponent_rating', 1500) - game.get('player_rating', 1500)
        if rating_diff > 0:
            score += min(rating_diff / 10, 30)  # Max 30 points for rating diff
        
        # Prefer games with more moves (more action)
        num_moves = game.get('num_moves', 20)
        if 20 <= num_moves <= 50:
            score += 20  # Ideal length
        elif num_moves > 50:
            score += 10  # Still good but might be long
        
        # Prefer decisive games over draws
        result = game.get('result', '')
        if 'draw' not in result.lower():
            score += 10
        
        # Prefer games with known openings
        opening = game.get('opening', '')
        if opening and opening != 'Unknown Opening':
            score += 15
        
        return score
    
    @classmethod
    def select_best(cls, games: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best game from a list."""
        if not games:
            return None
        
        # Score and sort games
        scored_games = [(cls.score_game(g), g) for g in games]
        scored_games.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_game = scored_games[0]
        logger.info(f"🏆 Selected best game (score: {best_score:.1f})")
        logger.info(f"   Opening: {best_game.get('opening', 'Unknown')}")
        logger.info(f"   Result: {'Win' if best_game.get('player_won') else 'Loss'}")
        logger.info(f"   Opponent: {best_game.get('opponent_rating', '?')} rating")
        
        return best_game

# =============================================================================
# DAILY AUTOMATION PIPELINE
# =============================================================================

class DailyAutomation:
    """
    Main automation class that orchestrates the daily workflow.
    """
    
    def __init__(self, dry_run: bool = False, use_enhanced: bool = True):
        self.dry_run = dry_run
        self.use_enhanced = use_enhanced and ENHANCED_AVAILABLE
        self.username = CHESS_COM_USERNAME
        self.tz = pytz.timezone(TIMEZONE)
        
        # Initialize components
        self.api = ChessComAPI(self.username)
        self.parser = GameParser(self.username)
        self.game_filter = GameFilter()  # GameFilter is static methods only
        self.storage = GameStorage(PGN_OUTPUT_DIR, JSON_OUTPUT_DIR)
        
        # Initialize video generator (enhanced if available)
        if self.use_enhanced:
            logger.info("🎬 Using Enhanced Video Generator (with Stockfish analysis)")
            self.video_generator = EnhancedVideoGenerator(
                player_username=self.username,
                output_dir=Path(VIDEO_OUTPUT_DIR),
                enable_stockfish=True,
                stockfish_depth=12
            )
            self.enhanced_mode = self.video_generator.stockfish.is_available() if hasattr(self.video_generator, 'stockfish') else False
            if self.enhanced_mode:
                logger.info("✅ Stockfish available - full analysis enabled")
            else:
                logger.info("⚠️ Stockfish not available - basic mode")
        else:
            logger.info("🎬 Using Basic Video Generator")
            self.video_generator = ChessVideoGenerator(self.username)
            self.enhanced_mode = False
        
        self.uploader = YouTubeUploader()
        self.selector = BestGameSelector()
        
        # Tracking
        self.stats = {
            'games_fetched': 0,
            'games_filtered': 0,
            'video_generated': False,
            'video_uploaded': False,
            'video_id': None
        }
    
    def _game_to_dict(self, parsed_game, raw_game: Dict) -> Dict[str, Any]:
        """Convert a ChessGame object to a dictionary for video generation."""
        return {
            'game_id': parsed_game.game_id,
            'white_player': parsed_game.white_player,
            'black_player': parsed_game.black_player,
            'white_rating': parsed_game.white_rating,
            'black_rating': parsed_game.black_rating,
            'player_color': parsed_game.player_color,
            'player_won': parsed_game.player_won,
            'player_rating': parsed_game.player_rating,
            'opponent_rating': parsed_game.opponent_rating,
            'result': parsed_game.result,
            'opening': parsed_game.opening or 'Unknown Opening',
            'num_moves': parsed_game.num_moves if hasattr(parsed_game, 'num_moves') else 0,
            'time_control': parsed_game.time_control,
            'time_class': parsed_game.time_class,
            'end_time': parsed_game.end_time.isoformat() if parsed_game.end_time else None,
            'pgn': raw_game.get('pgn', '')
        }
    
    def _sanitize_filename(self, name: str) -> str:
        """Remove invalid characters from filename."""
        import re
        # Replace invalid characters with underscore
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Replace multiple underscores with single
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized.strip('_')[:30]
    
    def _save_games(self, games: List[Dict], date_obj):
        """Save games to JSON and PGN files."""
        date_str = date_obj.strftime('%Y-%m-%d')
        json_dir = Path(JSON_OUTPUT_DIR) / date_str
        pgn_dir = Path(PGN_OUTPUT_DIR) / date_str
        json_dir.mkdir(parents=True, exist_ok=True)
        pgn_dir.mkdir(parents=True, exist_ok=True)
        
        for i, game in enumerate(games):
            # Sanitize opening name for filename
            opening_safe = self._sanitize_filename(game.get('opening', 'unknown'))
            
            # Save JSON
            json_file = json_dir / f"game_{i+1}_{opening_safe}.json"
            with open(json_file, 'w') as f:
                json.dump(game, f, indent=2, default=str)
            
            # Save PGN
            if game.get('pgn'):
                opponent = game.get('white_player') if game.get('player_color') == 'black' else game.get('black_player')
                opponent_safe = self._sanitize_filename(opponent)
                pgn_file = pgn_dir / f"game_{i+1}_{opponent_safe}.pgn"
                with open(pgn_file, 'w') as f:
                    f.write(game['pgn'])
        
        logger.info(f"💾 Saved {len(games)} games to {json_dir}")
    
    def fetch_todays_games(self) -> List[Dict[str, Any]]:
        """Fetch and filter today's games."""
        logger.info("=" * 60)
        logger.info("📥 PHASE 1: Fetching Games")
        logger.info("=" * 60)
        
        # Get current date in IST
        today = datetime.now(self.tz).date()
        logger.info(f"📅 Date: {today} ({TIMEZONE})")
        logger.info(f"👤 Player: {self.username}")
        
        # Fetch games for current month
        year, month = today.year, today.month
        
        try:
            games = self.api.get_monthly_games(year, month)
            self.stats['games_fetched'] = len(games) if games else 0
            logger.info(f"📦 Fetched {self.stats['games_fetched']} total games for {year}/{month:02d}")
        except Exception as e:
            logger.error(f"❌ Failed to fetch games: {e}")
            return []
        
        if not games:
            return []
        
        # Filter to today's Blitz games
        filtered = []
        for game in games:
            parsed = self.parser.parse_game(game)
            if parsed:
                # Check if it's today's game (use end_time attribute)
                game_date = parsed.end_time.date() if parsed.end_time else None
                if game_date == today and parsed.time_class == 'blitz':
                    # Convert to dict for compatibility
                    filtered.append(self._game_to_dict(parsed, game))
        
        self.stats['games_filtered'] = len(filtered)
        logger.info(f"✅ Found {len(filtered)} Blitz games from today")
        
        # Save games
        if filtered:
            self._save_games(filtered, today)
        
        return filtered
    
    def generate_video(self, game: Dict[str, Any]) -> Optional[Path]:
        """Generate video from a game."""
        logger.info("=" * 60)
        logger.info("🎬 PHASE 2: Generating Video")
        logger.info("=" * 60)
        
        # Get PGN file path
        today = datetime.now(self.tz).date()
        date_str = today.strftime('%Y-%m-%d')
        
        # Create PGN from game data
        pgn_dir = Path(PGN_OUTPUT_DIR) / date_str
        pgn_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the PGN file for this game
        opponent = game.get('white_player') if game.get('player_color') == 'black' else game.get('black_player')
        pgn_files = list(pgn_dir.glob(f"*{opponent}*.pgn"))
        
        if not pgn_files:
            # Save PGN if not already saved
            pgn_file = pgn_dir / f"game_{opponent}_{game.get('num_moves', 0)}moves.pgn"
            if 'pgn' in game:
                with open(pgn_file, 'w') as f:
                    f.write(game['pgn'])
                pgn_files = [pgn_file]
            else:
                logger.error("❌ No PGN data available for this game")
                return None
        
        pgn_file = pgn_files[0]
        logger.info(f"📄 Using PGN: {pgn_file.name}")
        
        # Generate video
        output_dir = Path(VIDEO_OUTPUT_DIR) / date_str
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = f"chess_short_{self._sanitize_filename(opponent)}_{datetime.now().strftime('%H%M%S')}"
        
        try:
            if self.use_enhanced:
                # Use enhanced video generator
                self.video_generator.output_dir = output_dir
                result_path = self.video_generator.generate_video(
                    pgn_path=str(pgn_file),
                    output_filename=video_name
                )
                thumbnail_path = Path(result_path).with_suffix('.jpg') if result_path else None
                if thumbnail_path and thumbnail_path.exists():
                    logger.info(f"🖼️ Thumbnail: {thumbnail_path.name}")
            else:
                # Use basic video generator
                self.video_generator.output_dir = output_dir
                result_path = self.video_generator.generate_video(
                    pgn_path=str(pgn_file),
                    output_filename=video_name
                )
            
            if result_path and Path(result_path).exists():
                self.stats['video_generated'] = True
                logger.info(f"✅ Video generated: {result_path}")
                
                # Save game metadata for upload
                json_path = Path(result_path).with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(game, f, indent=2, default=str)
                
                return Path(result_path)
            else:
                logger.error("❌ Video generation failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Video generation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def upload_video(self, video_path: Path, game: Dict[str, Any]) -> Optional[str]:
        """Upload video to YouTube."""
        logger.info("=" * 60)
        logger.info("📤 PHASE 3: Uploading to YouTube")
        logger.info("=" * 60)
        
        if self.dry_run:
            logger.info("🔸 DRY RUN - Skipping upload")
            return "DRY_RUN_VIDEO_ID"
        
        # Calculate publish time (6 PM IST today, or tomorrow if past 6 PM)
        publish_hour, publish_minute = map(int, PUBLISH_TIME.split(':'))
        publish_time = self.uploader.get_scheduled_publish_time(publish_hour, publish_minute)
        
        logger.info(f"📅 Scheduled publish: {publish_time.strftime('%Y-%m-%d %H:%M %Z')}")
        
        # Get game JSON path
        json_path = video_path.with_suffix('.json')
        
        try:
            video_id = self.uploader.upload_chess_video(
                video_path=str(video_path),
                game_json_path=str(json_path),
                publish_time=publish_time
            )
            
            if video_id:
                self.stats['video_uploaded'] = True
                self.stats['video_id'] = video_id
                logger.info(f"✅ Upload successful!")
                logger.info(f"   Video ID: {video_id}")
                logger.info(f"   URL: https://youtube.com/shorts/{video_id}")
                return video_id
            else:
                logger.error("❌ Upload failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            return None
    
    def run(self) -> bool:
        """Run the full daily automation pipeline for 3 winning games per day."""
        logger.info("🚀 Starting Daily Chess Shorts Automation (3 shorts per day)")
        logger.info(f"⏰ Time: {datetime.now(self.tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")

        if self.dry_run:
            logger.info("🔸 DRY RUN MODE - No uploads will be made")

        # Phase 1: Fetch games (today, then fallback to previous day if needed)
        today = datetime.now(self.tz).date()
        games = self.fetch_todays_games()
        if not games:
            # Try previous day
            prev_day = today - timedelta(days=1)
            logger.warning("⚠️ No games found for today, trying previous day...")
            games = self._fetch_games_for_date(prev_day)
            if not games:
                logger.error("❌ No games found for today or previous day. Exiting.")
                return False

        # Filter to wins only
        winning_games = [g for g in games if g.get('player_won')]
        if len(winning_games) < 3:
            # Try to supplement with previous day's wins if not enough
            prev_day = today - timedelta(days=1)
            prev_games = self._fetch_games_for_date(prev_day)
            prev_wins = [g for g in prev_games if g.get('player_won')] if prev_games else []
            winning_games += prev_wins
        if not winning_games:
            logger.error("❌ No winning games found for today or previous day. Exiting.")
            return False

        # Select up to 3 best wins
        scored_games = [(self.selector.score_game(g), g) for g in winning_games]
        scored_games.sort(key=lambda x: x[0], reverse=True)
        top_games = [g for _, g in scored_games[:3]]

        # Schedule times: 5 PM, 6 PM, 7 PM IST
        publish_times = [(17, 0), (18, 0), (19, 0)]
        video_ids = []
        for idx, (game, (hour, minute)) in enumerate(zip(top_games, publish_times), 1):
            logger.info(f"\n=== Processing Game {idx} ===")
            # Generate video with correct game number
            video_path = self.generate_video_with_number(game, idx)
            if not video_path:
                logger.error(f"❌ Video generation failed for Game {idx}")
                continue
            # Schedule upload
            publish_time = self.uploader.get_scheduled_publish_time(hour, minute)
            video_id = self.upload_video_with_time(video_path, game, idx, publish_time)
            if video_id:
                video_ids.append(video_id)
        if not video_ids:
            logger.error("❌ No videos uploaded.")
            return False
        logger.info("=" * 60)
        logger.info("✅ DAILY AUTOMATION COMPLETE! Uploaded videos:")
        for i, vid in enumerate(video_ids, 1):
            logger.info(f"   Game {i}: https://youtube.com/shorts/{vid}")
        return True

    def _fetch_games_for_date(self, date_obj):
        """Fetch and filter games for a specific date."""
        year, month = date_obj.year, date_obj.month
        try:
            games = self.api.get_monthly_games(year, month)
        except Exception as e:
            logger.error(f"❌ Failed to fetch games for {date_obj}: {e}")
            return []
        filtered = []
        for game in games:
            parsed = self.parser.parse_game(game)
            if parsed:
                game_date = parsed.end_time.date() if parsed.end_time else None
                if game_date == date_obj and parsed.time_class == 'blitz':
                    filtered.append(self._game_to_dict(parsed, game))
        return filtered

    def generate_video_with_number(self, game: Dict[str, Any], game_number: int) -> Optional[Path]:
        """Generate video from a game, passing game_number for title/thumbnail."""
        logger.info(f"🎬 Generating Video for Game {game_number}")
        today = datetime.now(self.tz).date()
        date_str = today.strftime('%Y-%m-%d')
        pgn_dir = Path(PGN_OUTPUT_DIR) / date_str
        pgn_dir.mkdir(parents=True, exist_ok=True)
        opponent = game.get('white_player') if game.get('player_color') == 'black' else game.get('black_player')
        pgn_files = list(pgn_dir.glob(f"*{opponent}*.pgn"))
        if not pgn_files:
            pgn_file = pgn_dir / f"game_{opponent}_{game.get('num_moves', 0)}moves.pgn"
            if 'pgn' in game:
                with open(pgn_file, 'w') as f:
                    f.write(game['pgn'])
                pgn_files = [pgn_file]
            else:
                logger.error("❌ No PGN data available for this game")
                return None
        pgn_file = pgn_files[0]
        output_dir = Path(VIDEO_OUTPUT_DIR) / date_str
        output_dir.mkdir(parents=True, exist_ok=True)
        video_name = f"Game_{game_number}"
        try:
            if self.use_enhanced:
                self.video_generator.output_dir = output_dir
                # Pass game_number to generator if supported
                if hasattr(self.video_generator, 'game_number'):
                    self.video_generator.game_number = game_number
                result_path = self.video_generator.generate_video(
                    pgn_path=str(pgn_file),
                    output_filename=video_name
                )
                thumbnail_path = Path(result_path).with_suffix('.jpg') if result_path else None
                if thumbnail_path and thumbnail_path.exists():
                    logger.info(f"🖼️ Thumbnail: {thumbnail_path.name}")
            else:
                self.video_generator.output_dir = output_dir
                result_path = self.video_generator.generate_video(
                    pgn_path=str(pgn_file),
                    output_filename=video_name
                )
            if result_path and Path(result_path).exists():
                logger.info(f"✅ Video generated: {result_path}")
                json_path = Path(result_path).with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(game, f, indent=2, default=str)
                return Path(result_path)
            else:
                logger.error("❌ Video generation failed")
                return None
        except Exception as e:
            logger.error(f"❌ Video generation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def upload_video_with_time(self, video_path: Path, game: Dict[str, Any], game_number: int, publish_time) -> Optional[str]:
        """Upload video to YouTube with custom title/description and scheduled time."""
        logger.info(f"📤 Uploading Game {game_number} to YouTube (scheduled at {publish_time.strftime('%Y-%m-%d %H:%M %Z')})")
        if self.dry_run:
            logger.info("🔸 DRY RUN - Skipping upload")
            return f"DRY_RUN_VIDEO_ID_{game_number}"
        # Use new metadata generator
        metadata = self.uploader.metadata_generator.generate(
            opening=game.get('opening', 'Unknown Opening'),
            game_number=game_number
        )
        metadata.publish_at = publish_time
        video_id = self.uploader.upload_video(
            video_path=str(video_path),
            metadata=metadata,
            notify_subscribers=False
        )
        if video_id:
            logger.info(f"✅ Upload successful! Video ID: {video_id}")
            logger.info(f"   URL: https://youtube.com/shorts/{video_id}")
            return video_id
        else:
            logger.error("❌ Upload failed")
            return None
    
    def fetch_only(self) -> bool:
        """Only fetch and save games."""
        games = self.fetch_todays_games()
        return len(games) > 0
    
    def video_only(self) -> bool:
        """Generate video from the most recent saved game."""
        logger.info("🎬 Video-only mode: Using most recent saved game")
        
        # Find most recent game JSON
        games_dir = Path(JSON_OUTPUT_DIR)
        if not games_dir.exists():
            logger.error("❌ No saved games found")
            return False
        
        # Get latest date folder (directories only, not summary files)
        date_folders = sorted(
            [d for d in games_dir.iterdir() if d.is_dir()],
            reverse=True
        )
        if not date_folders:
            logger.error("❌ No game folders found")
            return False
        
        latest_folder = date_folders[0]
        logger.info(f"📂 Using games from: {latest_folder.name}")
        json_files = list(latest_folder.glob("*.json"))
        
        if not json_files:
            logger.error(f"❌ No games in {latest_folder}")
            return False
        
        # Load games and select best
        games = []
        for jf in json_files:
            with open(jf) as f:
                games.append(json.load(f))
        
        best_game = self.selector.select_best(games)
        
        if not best_game:
            logger.error("❌ Could not select game")
            return False
        
        # Generate video
        video_path = self.generate_video(best_game)
        return video_path is not None

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Daily Chess Shorts Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python daily_automation.py              # Full pipeline with enhanced video
  python daily_automation.py --dry-run    # Test without uploading
  python daily_automation.py --basic      # Use basic video generator (no Stockfish)
  python daily_automation.py --fetch-only # Only fetch games
  python daily_automation.py --video-only # Only generate video
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without uploading to YouTube'
    )
    
    parser.add_argument(
        '--basic',
        action='store_true',
        help='Use basic video generator (no Stockfish analysis, captions, etc.)'
    )
    
    parser.add_argument(
        '--fetch-only',
        action='store_true',
        help='Only fetch games from Chess.com'
    )
    
    parser.add_argument(
        '--video-only',
        action='store_true',
        help='Only generate video from saved games'
    )
    
    args = parser.parse_args()
    
    use_enhanced = not args.basic
    automation = DailyAutomation(dry_run=args.dry_run, use_enhanced=use_enhanced)
    
    try:
        if args.fetch_only:
            success = automation.fetch_only()
        elif args.video_only:
            success = automation.video_only()
        else:
            success = automation.run()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
