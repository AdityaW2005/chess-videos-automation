#!/usr/bin/env python3
"""
Chess Video Generator for YouTube Shorts
=========================================
Generates vertical 9:16 videos from PGN files with animated board,
move annotations, and smooth transitions.

Dependencies:
- python-chess: PGN parsing and board SVG generation
- Pillow (PIL): Image processing and text overlays
- OpenCV: Video encoding and transitions
- cairosvg: SVG to PNG conversion

Author: Chess Shorts Automation Project
"""

import os
import io
import re
import sys
import json
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import chess
import chess.pgn
import chess.svg
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cairosvg

# Import configuration
try:
    from config import VIDEO_SETTINGS, VIDEO_OUTPUT_DIR
except ImportError:
    VIDEO_SETTINGS = {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "max_duration_seconds": 59
    }
    VIDEO_OUTPUT_DIR = "./videos"

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("VideoGenerator")

# =============================================================================
# CONSTANTS
# =============================================================================

# Video dimensions for YouTube Shorts (9:16 vertical)
VIDEO_WIDTH = VIDEO_SETTINGS.get("width", 1080)
VIDEO_HEIGHT = VIDEO_SETTINGS.get("height", 1920)
FPS = VIDEO_SETTINGS.get("fps", 30)

# Board rendering settings
BOARD_SIZE = 900  # Square board size in pixels
BOARD_MARGIN_TOP = 350  # Space above board for header
BOARD_MARGIN_BOTTOM = 670  # Space below board for move list

# Color scheme (dark theme for better visibility)
COLORS = {
    "background": (18, 18, 18),        # Dark background
    "board_light": "#f0d9b5",           # Light squares
    "board_dark": "#b58863",            # Dark squares
    "text_primary": (255, 255, 255),    # White text
    "text_secondary": (180, 180, 180),  # Gray text
    "accent": (129, 212, 250),          # Light blue accent
    "win": (76, 175, 80),               # Green for wins
    "loss": (244, 67, 54),              # Red for losses
    "highlight_last": "#aaa23a",        # Yellow for last move
    "highlight_check": "#ff0000"        # Red for check
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GameInfo:
    """Extracted game information from PGN."""
    white_player: str
    black_player: str
    white_rating: int
    black_rating: int
    result: str
    opening: str
    date: str
    time_control: str
    player_color: str  # "white" or "black"
    player_won: bool

# =============================================================================
# FONT MANAGEMENT
# =============================================================================

class FontManager:
    """Manages fonts for text rendering with fallbacks."""
    
    # Common font paths on different systems
    FONT_PATHS = {
        "macos": [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSText.ttf",
            "/Library/Fonts/Arial.ttf",
        ],
        "linux": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ],
        "windows": [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
    }
    
    _cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
    
    @classmethod
    def get_font(cls, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Get a font at the specified size with caching."""
        cache_key = (f"{'bold' if bold else 'regular'}", size)
        
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        font = cls._load_system_font(size)
        cls._cache[cache_key] = font
        return font
    
    @classmethod
    def _load_system_font(cls, size: int) -> ImageFont.FreeTypeFont:
        """Try to load a system font, falling back to default if needed."""
        import platform
        system = platform.system().lower()
        
        if system == "darwin":
            paths = cls.FONT_PATHS["macos"]
        elif system == "linux":
            paths = cls.FONT_PATHS["linux"]
        else:
            paths = cls.FONT_PATHS["windows"]
        
        for path in paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        
        # Fallback to default
        logger.warning("Using default font (custom fonts not found)")
        return ImageFont.load_default()

# =============================================================================
# CHESS BOARD RENDERER
# =============================================================================

class ChessBoardRenderer:
    """
    Renders chess positions as images using python-chess SVG output.
    """
    
    def __init__(self, board_size: int = BOARD_SIZE):
        self.board_size = board_size
    
    def render_position(
        self,
        board: chess.Board,
        last_move: Optional[chess.Move] = None,
        flipped: bool = False
    ) -> Image.Image:
        """
        Render a chess position as a PIL Image.
        
        Args:
            board: Current board position
            last_move: Last move played (for highlighting)
            flipped: If True, render from black's perspective
            
        Returns:
            PIL Image of the board
        """
        # Determine squares to highlight
        fill_squares = {}
        
        if last_move:
            # Highlight last move squares
            fill_squares[last_move.from_square] = COLORS["highlight_last"]
            fill_squares[last_move.to_square] = COLORS["highlight_last"]
        
        # Check if king is in check
        if board.is_check():
            king_square = board.king(board.turn)
            if king_square is not None:
                fill_squares[king_square] = COLORS["highlight_check"]
        
        # Generate SVG
        svg_data = chess.svg.board(
            board,
            flipped=flipped,
            fill=fill_squares,
            size=self.board_size,
            colors={
                "square light": COLORS["board_light"],
                "square dark": COLORS["board_dark"],
            }
        )
        
        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(
            bytestring=svg_data.encode('utf-8'),
            output_width=self.board_size,
            output_height=self.board_size
        )
        
        # Load as PIL Image
        image = Image.open(io.BytesIO(png_data))
        return image.convert('RGBA')

# =============================================================================
# VIDEO FRAME COMPOSER
# =============================================================================

class FrameComposer:
    """
    Composes complete video frames with board, annotations, and overlays.
    """
    
    def __init__(
        self,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
        board_size: int = BOARD_SIZE
    ):
        self.width = width
        self.height = height
        self.board_size = board_size
        self.board_renderer = ChessBoardRenderer(board_size)
        
        # Calculate board position (centered horizontally)
        self.board_x = (width - board_size) // 2
        self.board_y = BOARD_MARGIN_TOP
    
    def create_frame(
        self,
        board: chess.Board,
        game_info: GameInfo,
        move_number: int,
        total_moves: int,
        last_move: Optional[chess.Move] = None,
        san_move: Optional[str] = None,
        move_list: List[str] = None
    ) -> np.ndarray:
        """
        Create a complete video frame.
        
        Args:
            board: Current position
            game_info: Game metadata
            move_number: Current move number
            total_moves: Total moves in game
            last_move: Last move for highlighting
            san_move: SAN notation of last move
            move_list: List of all moves up to current position
            
        Returns:
            OpenCV-compatible numpy array (BGR)
        """
        # Create base image with dark background
        frame = Image.new('RGB', (self.width, self.height), COLORS["background"])
        draw = ImageDraw.Draw(frame)
        
        # Flip board if player is black
        flipped = game_info.player_color == "black"
        
        # Render chess board
        board_image = self.board_renderer.render_position(board, last_move, flipped)
        frame.paste(board_image, (self.board_x, self.board_y), board_image)
        
        # Draw header section
        self._draw_header(draw, game_info)
        
        # Draw player info bars
        self._draw_player_bars(draw, game_info, flipped)
        
        # Draw current move indicator
        if san_move:
            self._draw_current_move(draw, san_move, move_number, board.turn)
        
        # Draw move list (scrolling)
        if move_list:
            self._draw_move_list(draw, move_list, move_number)
        
        # Draw progress bar
        self._draw_progress_bar(draw, move_number, total_moves)
        
        # Convert PIL to OpenCV format (RGB to BGR)
        frame_array = np.array(frame)
        return cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
    
    def _draw_header(self, draw: ImageDraw.Draw, game_info: GameInfo):
        """Draw the header with opening name and game info."""
        # Opening name
        font_large = FontManager.get_font(48, bold=True)
        font_small = FontManager.get_font(32)
        
        # Truncate opening name if too long
        opening = game_info.opening[:35] + "..." if len(game_info.opening) > 35 else game_info.opening
        
        draw.text(
            (self.width // 2, 60),
            opening,
            font=font_large,
            fill=COLORS["text_primary"],
            anchor="mm"
        )
        
        # Time control
        draw.text(
            (self.width // 2, 120),
            f"⏱️ {game_info.time_control}",
            font=font_small,
            fill=COLORS["text_secondary"],
            anchor="mm"
        )
        
        # Result badge
        result_color = COLORS["win"] if game_info.player_won else COLORS["loss"]
        result_text = "WIN" if game_info.player_won else "LOSS"
        
        # Draw result badge
        badge_y = 180
        badge_padding = 20
        
        draw.rounded_rectangle(
            [(self.width // 2 - 60, badge_y - 20), 
             (self.width // 2 + 60, badge_y + 25)],
            radius=10,
            fill=result_color
        )
        draw.text(
            (self.width // 2, badge_y),
            result_text,
            font=font_small,
            fill=(255, 255, 255),
            anchor="mm"
        )
    
    def _draw_player_bars(
        self, 
        draw: ImageDraw.Draw, 
        game_info: GameInfo,
        flipped: bool
    ):
        """Draw player information bars above and below the board."""
        font = FontManager.get_font(36)
        font_rating = FontManager.get_font(28)
        
        # Top player (opponent if white, us if black when flipped)
        if flipped:
            top_player = game_info.white_player
            top_rating = game_info.white_rating
            bottom_player = game_info.black_player
            bottom_rating = game_info.black_rating
        else:
            top_player = game_info.black_player
            top_rating = game_info.black_rating
            bottom_player = game_info.white_player
            bottom_rating = game_info.white_rating
        
        # Top bar position
        bar_y_top = self.board_y - 60
        bar_y_bottom = self.board_y + self.board_size + 20
        
        # Top player
        draw.text(
            (self.board_x + 10, bar_y_top),
            f"⬛ {top_player}",
            font=font,
            fill=COLORS["text_primary"]
        )
        draw.text(
            (self.board_x + self.board_size - 10, bar_y_top),
            f"({top_rating})",
            font=font_rating,
            fill=COLORS["text_secondary"],
            anchor="ra"
        )
        
        # Bottom player
        draw.text(
            (self.board_x + 10, bar_y_bottom),
            f"⬜ {bottom_player}",
            font=font,
            fill=COLORS["text_primary"]
        )
        draw.text(
            (self.board_x + self.board_size - 10, bar_y_bottom),
            f"({bottom_rating})",
            font=font_rating,
            fill=COLORS["text_secondary"],
            anchor="ra"
        )
    
    def _draw_current_move(
        self, 
        draw: ImageDraw.Draw, 
        san_move: str,
        move_number: int,
        is_white_turn: bool
    ):
        """Draw the current move prominently."""
        font_large = FontManager.get_font(72, bold=True)
        font_number = FontManager.get_font(36)
        
        # Position below player bar
        y_pos = self.board_y + self.board_size + 100
        
        # Move number
        move_num_display = f"{(move_number + 1) // 2 + 1}." if not is_white_turn else f"{(move_number + 2) // 2}."
        
        draw.text(
            (self.width // 2 - 100, y_pos),
            move_num_display,
            font=font_number,
            fill=COLORS["text_secondary"],
            anchor="rm"
        )
        
        # Move in large text
        draw.text(
            (self.width // 2, y_pos),
            san_move,
            font=font_large,
            fill=COLORS["accent"],
            anchor="mm"
        )
    
    def _draw_move_list(
        self, 
        draw: ImageDraw.Draw, 
        move_list: List[str],
        current_move: int
    ):
        """Draw a scrolling move list."""
        font = FontManager.get_font(28)
        y_start = self.board_y + self.board_size + 180
        x_left = 100
        x_right = self.width // 2 + 50
        line_height = 40
        
        # Show last 8 moves (4 full moves)
        start_idx = max(0, current_move - 7)
        visible_moves = move_list[start_idx:current_move + 1]
        
        for i, move in enumerate(visible_moves):
            actual_idx = start_idx + i
            move_num = (actual_idx // 2) + 1
            is_white = actual_idx % 2 == 0
            
            y = y_start + (i // 2) * line_height
            
            if is_white:
                # Draw move number and white's move
                text = f"{move_num}. {move}"
                x = x_left
            else:
                # Draw black's move
                text = move
                x = x_right
            
            # Highlight current move
            color = COLORS["accent"] if actual_idx == current_move else COLORS["text_secondary"]
            
            draw.text((x, y), text, font=font, fill=color)
    
    def _draw_progress_bar(
        self, 
        draw: ImageDraw.Draw, 
        current: int, 
        total: int
    ):
        """Draw a progress bar at the bottom."""
        bar_height = 8
        bar_y = self.height - 60
        margin = 80
        bar_width = self.width - (margin * 2)
        
        # Background bar
        draw.rounded_rectangle(
            [(margin, bar_y), (margin + bar_width, bar_y + bar_height)],
            radius=4,
            fill=(60, 60, 60)
        )
        
        # Progress fill
        if total > 0:
            progress_width = int(bar_width * (current + 1) / total)
            draw.rounded_rectangle(
                [(margin, bar_y), (margin + progress_width, bar_y + bar_height)],
                radius=4,
                fill=COLORS["accent"]
            )
        
        # Move counter text
        font = FontManager.get_font(24)
        draw.text(
            (self.width // 2, bar_y + 30),
            f"Move {current + 1} of {total}",
            font=font,
            fill=COLORS["text_secondary"],
            anchor="mm"
        )

# =============================================================================
# PGN PARSER
# =============================================================================

class PGNParser:
    """Parses PGN files and extracts game information."""
    
    def __init__(self, player_username: str):
        self.player_username = player_username.lower()
    
    def parse_file(self, pgn_path: str) -> Tuple[Optional[chess.pgn.Game], Optional[GameInfo]]:
        """
        Parse a PGN file and extract game information.
        
        Args:
            pgn_path: Path to PGN file
            
        Returns:
            Tuple of (chess.pgn.Game, GameInfo) or (None, None) if failed
        """
        try:
            with open(pgn_path, 'r') as pgn_file:
                game = chess.pgn.read_game(pgn_file)
            
            if not game:
                logger.error(f"Failed to parse PGN: {pgn_path}")
                return None, None
            
            # Extract headers
            headers = game.headers
            
            white_player = headers.get("White", "Unknown")
            black_player = headers.get("Black", "Unknown")
            white_rating = self._parse_rating(headers.get("WhiteElo", "0"))
            black_rating = self._parse_rating(headers.get("BlackElo", "0"))
            result = headers.get("Result", "*")
            
            # Determine player color and win status
            if white_player.lower() == self.player_username:
                player_color = "white"
                player_won = result == "1-0"
            else:
                player_color = "black"
                player_won = result == "0-1"
            
            # Extract opening name from ECOUrl or ECO
            opening = self._extract_opening(headers)
            
            # Parse time control
            time_control = self._format_time_control(headers.get("TimeControl", "?"))
            
            game_info = GameInfo(
                white_player=white_player,
                black_player=black_player,
                white_rating=white_rating,
                black_rating=black_rating,
                result=result,
                opening=opening,
                date=headers.get("Date", ""),
                time_control=time_control,
                player_color=player_color,
                player_won=player_won
            )
            
            return game, game_info
            
        except Exception as e:
            logger.error(f"Error parsing PGN: {e}")
            return None, None
    
    def _parse_rating(self, rating_str: str) -> int:
        """Parse rating string to int."""
        try:
            return int(rating_str.replace("?", "0"))
        except ValueError:
            return 0
    
    def _extract_opening(self, headers: Dict) -> str:
        """Extract opening name from PGN headers."""
        eco_url = headers.get("ECOUrl", "")
        if eco_url:
            # Extract opening name from URL
            match = re.search(r'/openings?/([^"]+)', eco_url, re.IGNORECASE)
            if match:
                opening = match.group(1)
                # Clean up the opening name
                opening = opening.replace("-", " ").replace("/", " ")
                # Remove move sequences
                opening = re.sub(r'\d+\.?\s*\w+\d*', '', opening)
                return opening.strip().title()
        
        # Fallback to ECO code
        eco = headers.get("ECO", "")
        if eco:
            return f"ECO: {eco}"
        
        return "Unknown Opening"
    
    def _format_time_control(self, tc: str) -> str:
        """Format time control string for display."""
        if "+" in tc:
            parts = tc.split("+")
            try:
                minutes = int(parts[0]) // 60
                increment = int(parts[1])
                return f"{minutes}+{increment} Blitz"
            except ValueError:
                return tc
        else:
            try:
                minutes = int(tc) // 60
                return f"{minutes} min Blitz"
            except ValueError:
                return tc

# =============================================================================
# VIDEO GENERATOR
# =============================================================================

class ChessVideoGenerator:
    """
    Main class for generating chess videos from PGN files.
    """
    
    def __init__(
        self,
        player_username: str,
        output_dir: str = VIDEO_OUTPUT_DIR,
        fps: int = FPS,
        seconds_per_move: float = 1.5,
        transition_frames: int = 5
    ):
        """
        Initialize the video generator.
        
        Args:
            player_username: Chess.com username
            output_dir: Directory to save videos
            fps: Frames per second
            seconds_per_move: How long to show each position
            transition_frames: Number of frames for fade transition
        """
        self.player_username = player_username
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.seconds_per_move = seconds_per_move
        self.transition_frames = transition_frames
        
        self.parser = PGNParser(player_username)
        self.composer = FrameComposer()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_video(
        self,
        pgn_path: str,
        output_filename: Optional[str] = None,
        max_moves: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate a video from a PGN file.
        
        Args:
            pgn_path: Path to PGN file
            output_filename: Custom output filename (without extension)
            max_moves: Limit the number of moves to include
            
        Returns:
            Path to generated video, or None if failed
        """
        logger.info(f"🎬 Starting video generation from: {pgn_path}")
        
        # Parse PGN
        game, game_info = self.parser.parse_file(pgn_path)
        if not game or not game_info:
            return None
        
        logger.info(f"📋 {game_info.white_player} vs {game_info.black_player}")
        logger.info(f"📋 Opening: {game_info.opening}")
        logger.info(f"📋 Result: {'WIN' if game_info.player_won else 'LOSS'}")
        
        # Get all moves
        board = game.board()
        moves = list(game.mainline_moves())
        
        if max_moves:
            moves = moves[:max_moves]
        
        total_moves = len(moves)
        logger.info(f"📋 Total moves: {total_moves}")
        
        # Calculate if video will exceed 60 seconds
        estimated_duration = total_moves * self.seconds_per_move
        if estimated_duration > 59:
            logger.warning(f"⚠️ Video would be {estimated_duration:.1f}s, limiting to 59s")
            max_allowed_moves = int(59 / self.seconds_per_move)
            moves = moves[:max_allowed_moves]
            total_moves = len(moves)
            logger.info(f"📋 Limited to {total_moves} moves")
        
        # Generate output path
        if output_filename:
            output_path = self.output_dir / f"{output_filename}.mp4"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_str = "win" if game_info.player_won else "loss"
            output_path = self.output_dir / f"chess_{result_str}_{timestamp}.mp4"
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (VIDEO_WIDTH, VIDEO_HEIGHT)
        )
        
        if not video_writer.isOpened():
            logger.error("❌ Failed to open video writer")
            return None
        
        try:
            # Generate frames
            logger.info("🎞️ Generating frames...")
            
            # Initial position (starting position)
            move_list = []
            frames_per_position = int(self.fps * self.seconds_per_move)
            
            # Show starting position briefly
            initial_frame = self.composer.create_frame(
                board=board,
                game_info=game_info,
                move_number=0,
                total_moves=total_moves,
                move_list=[]
            )
            
            for _ in range(frames_per_position // 2):
                video_writer.write(initial_frame)
            
            # Process each move
            previous_frame = initial_frame
            
            for move_idx, move in enumerate(moves):
                # Get SAN notation before making the move
                san = board.san(move)
                move_list.append(san)
                
                # Make the move
                board.push(move)
                
                # Create frame for this position
                current_frame = self.composer.create_frame(
                    board=board,
                    game_info=game_info,
                    move_number=move_idx,
                    total_moves=total_moves,
                    last_move=move,
                    san_move=san,
                    move_list=move_list.copy()
                )
                
                # Add transition frames (fade)
                for t in range(self.transition_frames):
                    alpha = t / self.transition_frames
                    blended = cv2.addWeighted(
                        previous_frame, 1 - alpha,
                        current_frame, alpha,
                        0
                    )
                    video_writer.write(blended)
                
                # Add static frames for this position
                static_frames = frames_per_position - self.transition_frames
                for _ in range(static_frames):
                    video_writer.write(current_frame)
                
                previous_frame = current_frame
                
                # Progress logging
                if (move_idx + 1) % 10 == 0:
                    logger.info(f"   Processed {move_idx + 1}/{total_moves} moves")
            
            # Hold final position a bit longer
            for _ in range(frames_per_position):
                video_writer.write(current_frame)
            
            logger.info(f"✅ Video generation complete!")
            
        finally:
            video_writer.release()
        
        # Verify output
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"💾 Saved: {output_path}")
            logger.info(f"📁 Size: {file_size:.2f} MB")
            return str(output_path)
        else:
            logger.error("❌ Failed to save video")
            return None

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line interface for video generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate YouTube Shorts from chess PGN files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_generator.py games/pgn/2025-12-28/game.pgn
  python video_generator.py game.pgn --output my_video
  python video_generator.py game.pgn --max-moves 30 --speed 1.0
        """
    )
    
    parser.add_argument(
        'pgn_file',
        help='Path to PGN file'
    )
    
    parser.add_argument(
        '--user', '-u',
        default=None,
        help='Chess.com username (default: from config.py)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output filename (without .mp4 extension)'
    )
    
    parser.add_argument(
        '--max-moves', '-m',
        type=int,
        default=None,
        help='Maximum number of moves to include'
    )
    
    parser.add_argument(
        '--speed', '-s',
        type=float,
        default=1.5,
        help='Seconds per move (default: 1.5)'
    )
    
    args = parser.parse_args()
    
    # Get username from config or args
    username = args.user
    if not username:
        try:
            from config import CHESS_COM_USERNAME
            username = CHESS_COM_USERNAME
        except ImportError:
            logger.error("❌ Username not provided and config.py not found")
            sys.exit(1)
    
    # Create generator and run
    generator = ChessVideoGenerator(
        player_username=username,
        seconds_per_move=args.speed
    )
    
    result = generator.generate_video(
        pgn_path=args.pgn_file,
        output_filename=args.output,
        max_moves=args.max_moves
    )
    
    if result:
        logger.info(f"\n🎬 Video ready: {result}")
        sys.exit(0)
    else:
        logger.error("\n❌ Video generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
