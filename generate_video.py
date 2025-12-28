#!/usr/bin/env python3
"""
Chess Video Generator for YouTube Shorts (PIL-based, no Cairo required)
========================================================================
Generates vertical 9:16 videos from PGN files with animated board,
move annotations, and smooth transitions.

This version uses PIL/Pillow only - no Cairo dependency needed.

Dependencies:
- python-chess: PGN parsing
- Pillow (PIL): Image processing and rendering
- OpenCV: Video encoding

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
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import configuration
try:
    from config import VIDEO_SETTINGS, VIDEO_OUTPUT_DIR, CHESS_COM_USERNAME
except ImportError:
    VIDEO_SETTINGS = {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "max_duration_seconds": 59
    }
    VIDEO_OUTPUT_DIR = "./videos"
    CHESS_COM_USERNAME = "unknown"

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
SQUARE_SIZE = BOARD_SIZE // 8  # Size of each square
BOARD_MARGIN_TOP = 350  # Space above board for header
BOARD_MARGIN_BOTTOM = 670  # Space below board for move list

# Color scheme (dark theme for better visibility)
COLORS = {
    "background": (18, 18, 18),        # Dark background
    "board_light": (240, 217, 181),    # Light squares #f0d9b5
    "board_dark": (181, 136, 99),      # Dark squares #b58863
    "text_primary": (255, 255, 255),   # White text
    "text_secondary": (180, 180, 180), # Gray text
    "accent": (129, 212, 250),         # Light blue accent
    "win": (76, 175, 80),              # Green for wins
    "loss": (244, 67, 54),             # Red for losses
    "highlight_last": (170, 162, 58),  # Yellow for last move
    "highlight_check": (255, 0, 0, 100), # Red for check (with alpha)
    "coordinate": (100, 100, 100),     # Coordinate text color
}

# Unicode chess pieces (used for rendering)
PIECE_UNICODE = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
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
            "/System/Library/Fonts/SFNS.ttf",
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
    
    # Chess piece fonts (optional)
    CHESS_FONT_PATHS = [
        "/Library/Fonts/Chess.ttf",
        "~/.fonts/Chess.ttf"
    ]
    
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
        try:
            return ImageFont.load_default()
        except:
            return None

# =============================================================================
# PURE PIL CHESS BOARD RENDERER
# =============================================================================

class PurePixelBoardRenderer:
    """
    Renders chess positions as images using pure PIL (no Cairo/SVG).
    Uses Unicode chess symbols for pieces.
    """
    
    def __init__(self, board_size: int = BOARD_SIZE):
        self.board_size = board_size
        self.square_size = board_size // 8
        
        # Load chess piece font or use system font with Unicode
        self.piece_font = self._load_piece_font()
    
    def _load_piece_font(self) -> ImageFont.FreeTypeFont:
        """Load a font that supports chess Unicode symbols."""
        # Try to find a font with good chess symbol support
        font_paths = [
            "/System/Library/Fonts/Apple Symbols.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        
        piece_size = int(self.square_size * 0.75)
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, piece_size)
                    return font
                except Exception:
                    continue
        
        logger.warning("Chess symbol font not found, using default")
        return ImageFont.load_default()
    
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
        # Create board image
        img = Image.new('RGBA', (self.board_size, self.board_size))
        draw = ImageDraw.Draw(img)
        
        # Draw squares
        for rank in range(8):
            for file in range(8):
                # Calculate position
                if flipped:
                    x = (7 - file) * self.square_size
                    y = rank * self.square_size
                else:
                    x = file * self.square_size
                    y = (7 - rank) * self.square_size
                
                # Determine square color
                is_light = (rank + file) % 2 == 1
                color = COLORS["board_light"] if is_light else COLORS["board_dark"]
                
                # Check if this square should be highlighted
                square = chess.square(file, rank)
                if last_move:
                    if square == last_move.from_square or square == last_move.to_square:
                        # Blend highlight color with square color
                        highlight = COLORS["highlight_last"]
                        color = (
                            (color[0] + highlight[0]) // 2,
                            (color[1] + highlight[1]) // 2,
                            (color[2] + highlight[2]) // 2
                        )
                
                # Check for check highlight
                if board.is_check():
                    king_square = board.king(board.turn)
                    if square == king_square:
                        color = (255, 100, 100)  # Red tint for check
                
                # Draw square
                draw.rectangle(
                    [x, y, x + self.square_size, y + self.square_size],
                    fill=color
                )
                
                # Draw piece if present
                piece = board.piece_at(square)
                if piece:
                    self._draw_piece(draw, piece, x, y)
        
        # Draw coordinates
        self._draw_coordinates(draw, flipped)
        
        # Draw border
        draw.rectangle(
            [0, 0, self.board_size - 1, self.board_size - 1],
            outline=(50, 50, 50),
            width=2
        )
        
        return img
    
    def _draw_piece(self, draw: ImageDraw.Draw, piece: chess.Piece, x: int, y: int):
        """Draw a chess piece at the given position."""
        # Get Unicode symbol
        symbol = piece.unicode_symbol()
        
        # Calculate centered position
        bbox = draw.textbbox((0, 0), symbol, font=self.piece_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = x + (self.square_size - text_width) // 2
        text_y = y + (self.square_size - text_height) // 2 - bbox[1]
        
        # Draw piece with outline for visibility
        outline_color = (50, 50, 50) if piece.color == chess.WHITE else (220, 220, 220)
        
        # Draw outline
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text(
                (text_x + dx, text_y + dy),
                symbol,
                font=self.piece_font,
                fill=outline_color
            )
        
        # Draw piece
        piece_color = (255, 255, 255) if piece.color == chess.WHITE else (30, 30, 30)
        draw.text(
            (text_x, text_y),
            symbol,
            font=self.piece_font,
            fill=piece_color
        )
    
    def _draw_coordinates(self, draw: ImageDraw.Draw, flipped: bool):
        """Draw file and rank coordinates."""
        coord_font = FontManager.get_font(16)
        if not coord_font:
            return
            
        files = 'abcdefgh'
        ranks = '12345678'
        
        if flipped:
            files = files[::-1]
            ranks = ranks[::-1]
        
        # Draw file letters (a-h) at bottom
        for i, f in enumerate(files):
            x = i * self.square_size + self.square_size - 12
            y = self.board_size - 18
            # Use contrasting color based on square
            is_light = i % 2 == 0 if not flipped else i % 2 == 1
            color = COLORS["board_dark"] if is_light else COLORS["board_light"]
            draw.text((x, y), f, font=coord_font, fill=color)
        
        # Draw rank numbers (1-8) on left
        for i, r in enumerate(ranks[::-1]):
            x = 4
            y = i * self.square_size + 4
            is_light = i % 2 == 0 if not flipped else i % 2 == 1
            color = COLORS["board_dark"] if is_light else COLORS["board_light"]
            draw.text((x, y), r, font=coord_font, fill=color)

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
        self.board_renderer = PurePixelBoardRenderer(board_size)
        
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
        font_large = FontManager.get_font(48)
        font_small = FontManager.get_font(32)
        
        if not font_large or not font_small:
            return
        
        # Truncate opening name if too long
        opening = game_info.opening[:35] + "..." if len(game_info.opening) > 35 else game_info.opening
        
        # Get text size for centering
        bbox = draw.textbbox((0, 0), opening, font=font_large)
        text_width = bbox[2] - bbox[0]
        
        draw.text(
            ((self.width - text_width) // 2, 40),
            opening,
            font=font_large,
            fill=COLORS["text_primary"]
        )
        
        # Time control
        time_text = f"⏱ {game_info.time_control}"
        bbox = draw.textbbox((0, 0), time_text, font=font_small)
        text_width = bbox[2] - bbox[0]
        
        draw.text(
            ((self.width - text_width) // 2, 100),
            time_text,
            font=font_small,
            fill=COLORS["text_secondary"]
        )
        
        # Result badge
        result_color = COLORS["win"] if game_info.player_won else COLORS["loss"]
        result_text = "WIN" if game_info.player_won else "LOSS"
        
        # Draw result badge
        badge_y = 160
        badge_width = 120
        badge_height = 45
        
        draw.rounded_rectangle(
            [(self.width // 2 - badge_width // 2, badge_y), 
             (self.width // 2 + badge_width // 2, badge_y + badge_height)],
            radius=10,
            fill=result_color
        )
        
        font_badge = FontManager.get_font(28)
        if font_badge:
            bbox = draw.textbbox((0, 0), result_text, font=font_badge)
            text_width = bbox[2] - bbox[0]
            draw.text(
                ((self.width - text_width) // 2, badge_y + 8),
                result_text,
                font=font_badge,
                fill=(255, 255, 255)
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
        
        if not font or not font_rating:
            return
        
        # Top player (opponent if white, us if black when flipped)
        if flipped:
            top_player = game_info.white_player
            top_rating = game_info.white_rating
            top_symbol = "⬜"
            bottom_player = game_info.black_player
            bottom_rating = game_info.black_rating
            bottom_symbol = "⬛"
        else:
            top_player = game_info.black_player
            top_rating = game_info.black_rating
            top_symbol = "⬛"
            bottom_player = game_info.white_player
            bottom_rating = game_info.white_rating
            bottom_symbol = "⬜"
        
        # Top bar position
        bar_y_top = self.board_y - 60
        bar_y_bottom = self.board_y + self.board_size + 20
        
        # Truncate player names if too long
        top_player = top_player[:15] + "..." if len(top_player) > 15 else top_player
        bottom_player = bottom_player[:15] + "..." if len(bottom_player) > 15 else bottom_player
        
        # Top player
        draw.text(
            (self.board_x + 10, bar_y_top),
            f"{top_symbol} {top_player}",
            font=font,
            fill=COLORS["text_primary"]
        )
        draw.text(
            (self.board_x + self.board_size - 80, bar_y_top),
            f"({top_rating})",
            font=font_rating,
            fill=COLORS["text_secondary"]
        )
        
        # Bottom player
        draw.text(
            (self.board_x + 10, bar_y_bottom),
            f"{bottom_symbol} {bottom_player}",
            font=font,
            fill=COLORS["text_primary"]
        )
        draw.text(
            (self.board_x + self.board_size - 80, bar_y_bottom),
            f"({bottom_rating})",
            font=font_rating,
            fill=COLORS["text_secondary"]
        )
    
    def _draw_current_move(
        self, 
        draw: ImageDraw.Draw, 
        san_move: str,
        move_number: int,
        is_white_turn: bool
    ):
        """Draw the current move prominently."""
        font_large = FontManager.get_font(72)
        font_number = FontManager.get_font(36)
        
        if not font_large or not font_number:
            return
        
        # Position below player bar
        y_pos = self.board_y + self.board_size + 100
        
        # Move number
        full_move = (move_number // 2) + 1
        if move_number % 2 == 0:
            move_num_display = f"{full_move}."
        else:
            move_num_display = f"{full_move}..."
        
        # Draw move number
        draw.text(
            (self.width // 2 - 150, y_pos - 15),
            move_num_display,
            font=font_number,
            fill=COLORS["text_secondary"]
        )
        
        # Move in large text
        bbox = draw.textbbox((0, 0), san_move, font=font_large)
        text_width = bbox[2] - bbox[0]
        
        draw.text(
            ((self.width - text_width) // 2 + 30, y_pos - 25),
            san_move,
            font=font_large,
            fill=COLORS["accent"]
        )
    
    def _draw_move_list(
        self, 
        draw: ImageDraw.Draw, 
        move_list: List[str],
        current_move: int
    ):
        """Draw a scrolling move list."""
        font = FontManager.get_font(28)
        if not font:
            return
            
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
        if font:
            text = f"Move {current + 1} of {total}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            
            draw.text(
                ((self.width - text_width) // 2, bar_y + 20),
                text,
                font=font,
                fill=COLORS["text_secondary"]
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
                # Remove move sequences (like 2.Bb2 Nc6 etc)
                opening = re.sub(r'\d+\.?\s*[A-Za-z]+\d*', '', opening)
                opening = re.sub(r'\s+', ' ', opening).strip()
                # Limit length
                if len(opening) > 40:
                    opening = opening[:40]
                return opening.title()
        
        # Fallback to ECO code
        eco = headers.get("ECO", "")
        if eco:
            return f"ECO: {eco}"
        
        return "Chess Game"
    
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
                seconds = int(tc)
                minutes = seconds // 60
                if minutes < 3:
                    return f"{minutes} min Bullet"
                elif minutes <= 10:
                    return f"{minutes} min Blitz"
                else:
                    return f"{minutes} min Rapid"
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
        estimated_duration = total_moves * self.seconds_per_move + 2  # +2 for intro/outro
        if estimated_duration > 59:
            logger.warning(f"⚠️ Video would be {estimated_duration:.1f}s, limiting to 59s")
            max_allowed_moves = int((59 - 2) / self.seconds_per_move)
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
            duration = (total_moves * self.seconds_per_move) + 2
            logger.info(f"💾 Saved: {output_path}")
            logger.info(f"📁 Size: {file_size:.2f} MB")
            logger.info(f"⏱️ Duration: ~{duration:.1f} seconds")
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
  python generate_video.py games/pgn/2025-12-28/game.pgn
  python generate_video.py game.pgn --output my_video
  python generate_video.py game.pgn --max-moves 30 --speed 1.0
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
