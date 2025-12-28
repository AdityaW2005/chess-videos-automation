#!/usr/bin/env python3
"""
Chess.com Blitz Games Fetcher
=============================
Fetches daily Blitz games from Chess.com Public API and saves them as PGN files.

Chess.com API Endpoints Used:
- Monthly Games: https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM}

Author: Chess Shorts Automation Project
Date: December 2024
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Import configuration
try:
    from config import (
        CHESS_COM_USERNAME,
        PGN_OUTPUT_DIR,
        JSON_OUTPUT_DIR,
        API_DELAY_SECONDS,
        GAME_TYPES
    )
except ImportError:
    print("❌ Error: config.py not found. Please create it from config_template.py")
    sys.exit(1)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging with both file and console output."""
    logger = logging.getLogger("ChessGamesFetcher")
    logger.setLevel(logging.DEBUG)
    
    # Create logs directory
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # File handler (detailed logs)
    file_handler = logging.FileHandler(
        log_dir / f"fetch_{date.today().isoformat()}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (user-friendly output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ChessGame:
    """Represents a parsed Chess.com game."""
    game_id: str
    url: str
    pgn: str
    time_class: str  # blitz, rapid, bullet, daily
    time_control: str  # e.g., "600" or "180+2"
    white_player: str
    black_player: str
    white_rating: int
    black_rating: int
    result: str  # "1-0", "0-1", "1/2-1/2"
    end_time: datetime
    player_color: str  # "white" or "black"
    player_won: bool
    player_rating: int
    opponent_rating: int
    opening: Optional[str] = None
    accuracy: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['end_time'] = self.end_time.isoformat()
        return data

# =============================================================================
# CHESS.COM API CLIENT
# =============================================================================

class ChessComAPI:
    """
    Client for Chess.com Public API.
    
    API Documentation: https://www.chess.com/news/view/published-data-api
    Rate Limit: Be respectful, add delays between requests
    """
    
    BASE_URL = "https://api.chess.com/pub"
    
    def __init__(self, username: str):
        self.username = username.lower()
        self.session = requests.Session()
        # Set a descriptive User-Agent (Chess.com recommends this)
        self.session.headers.update({
            'User-Agent': 'ChessYouTubeShortsBot/1.0 (Educational Project)',
            'Accept': 'application/json'
        })
    
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """
        Make a GET request to Chess.com API with error handling.
        
        Args:
            endpoint: API endpoint (without base URL)
            
        Returns:
            JSON response as dict, or None if request failed
        """
        url = f"{self.BASE_URL}{endpoint}"
        logger.debug(f"📡 Requesting: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            
            # Handle different status codes
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"⚠️  Resource not found: {endpoint}")
                return None
            elif response.status_code == 429:
                logger.error("🚫 Rate limited by Chess.com. Please wait and try again.")
                return None
            else:
                logger.error(f"❌ API error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("⏰ Request timed out. Chess.com may be slow.")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("🔌 Connection error. Check your internet connection.")
            return None
        except json.JSONDecodeError:
            logger.error("📄 Invalid JSON response from Chess.com")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            return None
    
    def verify_user_exists(self) -> bool:
        """Check if the username exists on Chess.com."""
        logger.info(f"🔍 Verifying user '{self.username}' exists...")
        data = self._make_request(f"/player/{self.username}")
        
        if data:
            logger.info(f"✅ User verified: {data.get('username', self.username)}")
            return True
        else:
            logger.error(f"❌ User '{self.username}' not found on Chess.com")
            return False
    
    def get_monthly_games(self, year: int, month: int) -> Optional[List[Dict]]:
        """
        Fetch all games for a specific month.
        
        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            
        Returns:
            List of game dictionaries, or None if failed
        """
        # Chess.com API expects 2-digit month
        month_str = str(month).zfill(2)
        endpoint = f"/player/{self.username}/games/{year}/{month_str}"
        
        logger.info(f"📅 Fetching games for {year}-{month_str}...")
        data = self._make_request(endpoint)
        
        if data and 'games' in data:
            games = data['games']
            logger.info(f"📊 Found {len(games)} total games in {year}-{month_str}")
            return games
        
        return []
    
    def get_todays_games(self, target_date: Optional[date] = None) -> List[Dict]:
        """
        Fetch games played on a specific date (default: today).
        
        Chess.com API doesn't have a daily endpoint, so we fetch the whole month
        and filter by date.
        
        Args:
            target_date: Date to fetch games for (default: today)
            
        Returns:
            List of games played on the target date
        """
        if target_date is None:
            target_date = date.today()
        
        logger.info(f"📆 Fetching games for {target_date.isoformat()}...")
        
        # Fetch the month's games
        monthly_games = self.get_monthly_games(target_date.year, target_date.month)
        
        if not monthly_games:
            logger.info(f"📭 No games found for {target_date.strftime('%B %Y')}")
            return []
        
        # Filter games by date
        todays_games = []
        for game in monthly_games:
            # end_time is Unix timestamp
            game_timestamp = game.get('end_time', 0)
            game_date = datetime.fromtimestamp(game_timestamp).date()
            
            if game_date == target_date:
                todays_games.append(game)
        
        logger.info(f"🎯 Found {len(todays_games)} games on {target_date.isoformat()}")
        return todays_games

# =============================================================================
# GAME PARSER
# =============================================================================

class GameParser:
    """Parses raw Chess.com game data into structured ChessGame objects."""
    
    def __init__(self, username: str):
        self.username = username.lower()
    
    def parse_game(self, raw_game: Dict) -> Optional[ChessGame]:
        """
        Parse a raw game dictionary into a ChessGame object.
        
        Args:
            raw_game: Raw game data from Chess.com API
            
        Returns:
            ChessGame object, or None if parsing failed
        """
        try:
            # Extract basic info
            url = raw_game.get('url', '')
            game_id = url.split('/')[-1] if url else str(raw_game.get('end_time', 'unknown'))
            
            # Extract player info
            white = raw_game.get('white', {})
            black = raw_game.get('black', {})
            
            white_player = white.get('username', 'Unknown').lower()
            black_player = black.get('username', 'Unknown').lower()
            white_rating = white.get('rating', 0)
            black_rating = black.get('rating', 0)
            white_result = white.get('result', '')
            
            # Determine result in standard notation
            result = self._determine_result(white_result)
            
            # Determine player's color and if they won
            if white_player == self.username:
                player_color = "white"
                player_won = white_result == "win"
                player_rating = white_rating
                opponent_rating = black_rating
            else:
                player_color = "black"
                player_won = white_result != "win" and result != "1/2-1/2"
                player_rating = black_rating
                opponent_rating = white_rating
            
            # Parse time info
            time_class = raw_game.get('time_class', 'unknown')
            time_control = raw_game.get('time_control', 'unknown')
            end_time = datetime.fromtimestamp(raw_game.get('end_time', 0))
            
            # Extract PGN
            pgn = raw_game.get('pgn', '')
            
            # Try to extract opening from PGN
            opening = self._extract_opening(pgn)
            
            # Extract accuracy if available
            accuracy = None
            if 'accuracies' in raw_game:
                accuracy = raw_game['accuracies']
            
            return ChessGame(
                game_id=game_id,
                url=url,
                pgn=pgn,
                time_class=time_class,
                time_control=time_control,
                white_player=white_player,
                black_player=black_player,
                white_rating=white_rating,
                black_rating=black_rating,
                result=result,
                end_time=end_time,
                player_color=player_color,
                player_won=player_won,
                player_rating=player_rating,
                opponent_rating=opponent_rating,
                opening=opening,
                accuracy=accuracy
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to parse game: {str(e)}")
            return None
    
    def _determine_result(self, white_result: str) -> str:
        """Convert Chess.com result format to standard notation."""
        win_results = ['win']
        draw_results = ['agreed', 'stalemate', 'insufficient', 'repetition', 
                       '50move', 'timevsinsufficient']
        
        if white_result in win_results:
            return "1-0"
        elif white_result in draw_results:
            return "1/2-1/2"
        else:
            return "0-1"
    
    def _extract_opening(self, pgn: str) -> Optional[str]:
        """Extract opening name from PGN header."""
        import re
        match = re.search(r'\[ECOUrl ".*?/([^"]+)"\]', pgn)
        if match:
            opening = match.group(1).replace('-', ' ').title()
            return opening
        return None

# =============================================================================
# GAME FILTER
# =============================================================================

class GameFilter:
    """Filters games based on type and quality criteria."""
    
    @staticmethod
    def filter_by_type(games: List[ChessGame], 
                       include_blitz: bool = True,
                       include_rapid: bool = False,
                       include_bullet: bool = False,
                       include_daily: bool = False) -> List[ChessGame]:
        """
        Filter games by time control type.
        
        Args:
            games: List of ChessGame objects
            include_*: Boolean flags for each game type
            
        Returns:
            Filtered list of games
        """
        allowed_types = set()
        if include_blitz:
            allowed_types.add('blitz')
        if include_rapid:
            allowed_types.add('rapid')
        if include_bullet:
            allowed_types.add('bullet')
        if include_daily:
            allowed_types.add('daily')
        
        filtered = [g for g in games if g.time_class in allowed_types]
        logger.info(f"🎮 Filtered to {len(filtered)} {'/'.join(allowed_types)} games")
        return filtered
    
    @staticmethod
    def sort_by_interest(games: List[ChessGame]) -> List[ChessGame]:
        """
        Sort games by how "interesting" they might be for content.
        
        Criteria:
        1. Wins are more interesting than losses
        2. Higher rated opponents are more interesting
        3. Games with accuracy data are preferred
        
        Returns:
            Sorted list (most interesting first)
        """
        def interest_score(game: ChessGame) -> int:
            score = 0
            
            # Wins are great content
            if game.player_won:
                score += 100
            
            # Beating higher rated opponents is impressive
            rating_diff = game.opponent_rating - game.player_rating
            if rating_diff > 0:
                score += min(rating_diff, 200)  # Cap bonus at 200
            
            # Games with accuracy are more analyzable
            if game.accuracy:
                score += 50
            
            return score
        
        sorted_games = sorted(games, key=interest_score, reverse=True)
        
        if sorted_games:
            top_game = sorted_games[0]
            logger.info(f"⭐ Most interesting game: vs {top_game.opponent_rating} "
                       f"({'WIN' if top_game.player_won else 'LOSS'})")
        
        return sorted_games

# =============================================================================
# FILE STORAGE
# =============================================================================

class GameStorage:
    """Handles saving games to disk as PGN and JSON files."""
    
    def __init__(self, pgn_dir: str, json_dir: str):
        self.pgn_dir = Path(pgn_dir)
        self.json_dir = Path(json_dir)
        
        # Create directories
        self.pgn_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)
    
    def save_game(self, game: ChessGame, game_date: date) -> tuple[Optional[str], Optional[str]]:
        """
        Save a game as both PGN and JSON files.
        
        File naming convention:
        - {date}_{time_class}_{game_id}.pgn
        - {date}_{time_class}_{game_id}.json
        
        Args:
            game: ChessGame object to save
            game_date: Date of the game
            
        Returns:
            Tuple of (pgn_path, json_path) or (None, None) if failed
        """
        try:
            # Create date-based subdirectory
            date_str = game_date.isoformat()
            pgn_date_dir = self.pgn_dir / date_str
            json_date_dir = self.json_dir / date_str
            pgn_date_dir.mkdir(exist_ok=True)
            json_date_dir.mkdir(exist_ok=True)
            
            # Generate filename
            result_str = "win" if game.player_won else "loss"
            filename = f"{date_str}_{game.time_class}_{result_str}_{game.game_id}"
            
            # Save PGN
            pgn_path = pgn_date_dir / f"{filename}.pgn"
            with open(pgn_path, 'w', encoding='utf-8') as f:
                f.write(game.pgn)
            logger.debug(f"💾 Saved PGN: {pgn_path.name}")
            
            # Save JSON (for metadata and future processing)
            json_path = json_date_dir / f"{filename}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(game.to_dict(), f, indent=2)
            logger.debug(f"💾 Saved JSON: {json_path.name}")
            
            return str(pgn_path), str(json_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save game {game.game_id}: {str(e)}")
            return None, None
    
    def save_daily_summary(self, games: List[ChessGame], game_date: date) -> Optional[str]:
        """
        Save a summary of all games for a day.
        
        Args:
            games: List of ChessGame objects
            game_date: Date of the games
            
        Returns:
            Path to summary file, or None if failed
        """
        try:
            summary = {
                "date": game_date.isoformat(),
                "total_games": len(games),
                "wins": sum(1 for g in games if g.player_won),
                "losses": sum(1 for g in games if not g.player_won),
                "game_types": {},
                "games": [g.to_dict() for g in games]
            }
            
            # Count by game type
            for game in games:
                game_type = game.time_class
                if game_type not in summary["game_types"]:
                    summary["game_types"][game_type] = 0
                summary["game_types"][game_type] += 1
            
            summary_path = self.json_dir / f"{game_date.isoformat()}_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"📋 Saved daily summary: {summary_path.name}")
            return str(summary_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save summary: {str(e)}")
            return None

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def fetch_and_save_todays_blitz_games(
    username: str = CHESS_COM_USERNAME,
    target_date: Optional[date] = None
) -> List[ChessGame]:
    """
    Main function to fetch, parse, filter, and save today's Blitz games.
    
    Args:
        username: Chess.com username
        target_date: Date to fetch games for (default: today)
        
    Returns:
        List of ChessGame objects that were saved
    """
    if target_date is None:
        target_date = date.today()
    
    logger.info("=" * 60)
    logger.info("🏁 CHESS.COM BLITZ GAMES FETCHER")
    logger.info("=" * 60)
    logger.info(f"👤 Username: {username}")
    logger.info(f"📅 Date: {target_date.isoformat()}")
    logger.info("=" * 60)
    
    # Initialize components
    api = ChessComAPI(username)
    parser = GameParser(username)
    storage = GameStorage(PGN_OUTPUT_DIR, JSON_OUTPUT_DIR)
    
    # Step 1: Verify user exists
    if not api.verify_user_exists():
        logger.error("❌ Cannot proceed without valid username")
        return []
    
    time.sleep(API_DELAY_SECONDS)
    
    # Step 2: Fetch today's games
    raw_games = api.get_todays_games(target_date)
    
    if not raw_games:
        logger.info("📭 No games played today!")
        logger.info("💡 Tip: Play some Blitz games on Chess.com first")
        return []
    
    # Step 3: Parse all games
    logger.info(f"\n🔄 Parsing {len(raw_games)} games...")
    parsed_games = []
    for raw_game in raw_games:
        game = parser.parse_game(raw_game)
        if game:
            parsed_games.append(game)
    
    logger.info(f"✅ Successfully parsed {len(parsed_games)} games")
    
    # Step 4: Filter for Blitz only (or other types based on config)
    blitz_games = GameFilter.filter_by_type(
        parsed_games,
        include_blitz=GAME_TYPES.get('blitz', True),
        include_rapid=GAME_TYPES.get('rapid', False),
        include_bullet=GAME_TYPES.get('bullet', False),
        include_daily=GAME_TYPES.get('daily', False)
    )
    
    if not blitz_games:
        logger.info("📭 No Blitz games found for today!")
        logger.info("💡 You played other game types:")
        for game in parsed_games:
            logger.info(f"   - {game.time_class}: vs {game.opponent_rating}")
        return []
    
    # Step 5: Sort by interest level
    sorted_games = GameFilter.sort_by_interest(blitz_games)
    
    # Step 6: Save all games
    logger.info(f"\n💾 Saving {len(sorted_games)} Blitz games...")
    saved_games = []
    for game in sorted_games:
        pgn_path, json_path = storage.save_game(game, target_date)
        if pgn_path:
            saved_games.append(game)
    
    # Step 7: Save daily summary
    storage.save_daily_summary(sorted_games, target_date)
    
    # Step 8: Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"   Total games today: {len(raw_games)}")
    logger.info(f"   Blitz games: {len(blitz_games)}")
    logger.info(f"   Wins: {sum(1 for g in blitz_games if g.player_won)}")
    logger.info(f"   Losses: {sum(1 for g in blitz_games if not g.player_won)}")
    logger.info(f"   Files saved to: {PGN_OUTPUT_DIR}")
    logger.info("=" * 60)
    
    # Print game details
    if sorted_games:
        logger.info("\n🎮 GAMES SAVED (sorted by interest):")
        for i, game in enumerate(sorted_games, 1):
            result_emoji = "✅" if game.player_won else "❌"
            logger.info(
                f"   {i}. {result_emoji} vs {game.opponent_rating} "
                f"({game.player_color}) - {game.opening or 'Unknown opening'}"
            )
    
    return saved_games


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line interface for the game fetcher."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fetch Chess.com Blitz games for YouTube Shorts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_chess_games.py                    # Fetch today's games
  python fetch_chess_games.py --date 2024-12-25  # Fetch specific date
  python fetch_chess_games.py --user hikaru      # Different username
        """
    )
    
    parser.add_argument(
        '--user', '-u',
        default=CHESS_COM_USERNAME,
        help='Chess.com username (default: from config.py)'
    )
    
    parser.add_argument(
        '--date', '-d',
        type=str,
        default=None,
        help='Date to fetch games for (YYYY-MM-DD format, default: today)'
    )
    
    args = parser.parse_args()
    
    # Parse date if provided
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            logger.error(f"❌ Invalid date format: {args.date}")
            logger.error("   Use YYYY-MM-DD format (e.g., 2024-12-25)")
            sys.exit(1)
    
    # Run the fetcher
    games = fetch_and_save_todays_blitz_games(
        username=args.user,
        target_date=target_date
    )
    
    # Exit with appropriate code
    if games:
        logger.info("\n🎬 Ready to generate video from these games!")
        sys.exit(0)
    else:
        logger.info("\n💤 No games to process today")
        sys.exit(0)  # Not an error, just no games


if __name__ == "__main__":
    main()
