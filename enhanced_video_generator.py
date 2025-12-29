#!/usr/bin/env python3
"""
Enhanced Chess Video Generator
==============================
Adds professional features to chess videos:
1. Stockfish evaluation bar overlay
2. Auto-generated captions for critical moves
3. Background music support
4. Thumbnail generation

All using FREE Python libraries!

Author: Chess Shorts Automation Project
"""

import os
import sys
import json
import logging
import math
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from io import BytesIO

import chess
import chess.pgn
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Video processing
import cv2

# Stockfish integration
try:
    from stockfish import Stockfish
    STOCKFISH_AVAILABLE = True
except ImportError:
    STOCKFISH_AVAILABLE = False

# Audio processing (optional)
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# Stockfish path (macOS Homebrew)
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

# Video dimensions (YouTube Shorts)
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30

# Colors
COLORS = {
    'white_square': (240, 217, 181),
    'black_square': (181, 136, 99),
    'highlight_from': (255, 255, 0, 128),
    'highlight_to': (255, 255, 0, 180),
    'eval_white': (255, 255, 255),
    'eval_black': (50, 50, 50),
    'eval_border': (100, 100, 100),
    'brilliant': (38, 166, 154),      # Teal - brilliant move
    'great': (118, 190, 67),          # Green - great move
    'best': (150, 190, 67),           # Light green - best move
    'good': (200, 200, 100),          # Yellow-green - good
    'inaccuracy': (240, 180, 50),     # Orange - inaccuracy
    'mistake': (230, 120, 50),        # Dark orange - mistake
    'blunder': (200, 50, 50),         # Red - blunder
    'caption_bg': (0, 0, 0, 200),
    'caption_text': (255, 255, 255),
}

# Evaluation thresholds (in centipawns)
EVAL_THRESHOLDS = {
    'brilliant': 300,    # Finds only move that saves/wins
    'great': 150,        # Much better than alternatives
    'best': 50,          # Best move
    'good': 0,           # Decent move
    'inaccuracy': -50,   # Small mistake
    'mistake': -150,     # Significant error
    'blunder': -300,     # Game-changing error
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("EnhancedVideoGen")

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MoveAnalysis:
    """Analysis of a single move."""
    move_number: int
    move_san: str
    move_uci: str
    eval_before: float  # Evaluation before move (centipawns)
    eval_after: float   # Evaluation after move
    eval_change: float  # Change in evaluation
    is_player_move: bool
    move_quality: str   # brilliant, great, best, good, inaccuracy, mistake, blunder
    is_critical: bool   # Should we show caption?
    caption: str        # Caption text to display

@dataclass
class GameAnalysis:
    """Full game analysis."""
    moves: List[MoveAnalysis]
    player_color: str
    player_won: bool
    critical_moments: List[int]  # Move indices with captions
    average_centipawn_loss: float
    blunders: int
    mistakes: int
    brilliancies: int

# =============================================================================
# STOCKFISH ANALYZER
# =============================================================================

class StockfishAnalyzer:
    """
    Analyzes chess positions using Stockfish engine.
    """
    
    def __init__(self, stockfish_path: str = STOCKFISH_PATH, depth: int = 12):
        """
        Initialize Stockfish analyzer.
        
        Args:
            stockfish_path: Path to Stockfish binary
            depth: Analysis depth (higher = slower but more accurate)
        """
        self.depth = depth
        self.stockfish = None
        
        if not STOCKFISH_AVAILABLE:
            logger.warning("⚠️ Stockfish Python package not installed")
            return
            
        if not os.path.exists(stockfish_path):
            logger.warning(f"⚠️ Stockfish not found at {stockfish_path}")
            return
        
        try:
            self.stockfish = Stockfish(
                path=stockfish_path,
                depth=depth,
                parameters={
                    "Threads": 2,
                    "Hash": 128,
                    "Minimum Thinking Time": 10
                }
            )
            logger.info(f"✅ Stockfish initialized (depth={depth})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Stockfish: {e}")
            self.stockfish = None
    
    def is_available(self) -> bool:
        """Check if Stockfish is available."""
        return self.stockfish is not None
    
    def evaluate_position(self, fen: str) -> float:
        """
        Evaluate a position.
        
        Args:
            fen: FEN string of position
            
        Returns:
            Evaluation in centipawns (positive = white advantage)
        """
        if not self.stockfish:
            return 0.0
        
        try:
            self.stockfish.set_fen_position(fen)
            evaluation = self.stockfish.get_evaluation()
            
            if evaluation["type"] == "cp":
                return evaluation["value"]
            elif evaluation["type"] == "mate":
                # Convert mate score to large centipawn value
                mate_moves = evaluation["value"]
                if mate_moves > 0:
                    return 10000 - (mate_moves * 10)  # White mates
                else:
                    return -10000 - (mate_moves * 10)  # Black mates
            return 0.0
        except Exception as e:
            logger.debug(f"Eval error: {e}")
            return 0.0
    
    def get_best_move(self, fen: str) -> Optional[str]:
        """Get the best move for a position."""
        if not self.stockfish:
            return None
        
        try:
            self.stockfish.set_fen_position(fen)
            return self.stockfish.get_best_move()
        except:
            return None
    
    def analyze_move(
        self,
        fen_before: str,
        move_uci: str,
        fen_after: str,
        is_white_move: bool
    ) -> Tuple[float, float, str]:
        """
        Analyze a move and determine its quality.
        
        Returns:
            (eval_before, eval_after, quality)
        """
        eval_before = self.evaluate_position(fen_before)
        eval_after = self.evaluate_position(fen_after)
        
        # Calculate evaluation change from moving player's perspective
        if is_white_move:
            eval_change = eval_after - eval_before
        else:
            eval_change = -(eval_after - eval_before)
        
        # Determine move quality
        best_move = self.get_best_move(fen_before)
        is_best = best_move == move_uci
        
        if is_best and eval_change > EVAL_THRESHOLDS['brilliant']:
            quality = 'brilliant'
        elif is_best and eval_change >= EVAL_THRESHOLDS['great']:
            quality = 'great'
        elif is_best or eval_change >= EVAL_THRESHOLDS['best']:
            quality = 'best'
        elif eval_change >= EVAL_THRESHOLDS['good']:
            quality = 'good'
        elif eval_change >= EVAL_THRESHOLDS['inaccuracy']:
            quality = 'inaccuracy'
        elif eval_change >= EVAL_THRESHOLDS['mistake']:
            quality = 'mistake'
        else:
            quality = 'blunder'
        
        return eval_before, eval_after, quality

# =============================================================================
# GAME ANALYZER
# =============================================================================

class GameAnalyzer:
    """
    Analyzes entire games for critical moments and move quality.
    """
    
    def __init__(self, stockfish_analyzer: Optional[StockfishAnalyzer] = None):
        self.stockfish = stockfish_analyzer or StockfishAnalyzer()
    
    def analyze_game(
        self,
        game: chess.pgn.Game,
        player_username: str
    ) -> GameAnalysis:
        """
        Analyze a complete game.
        
        Args:
            game: Parsed PGN game
            player_username: Username of the player
            
        Returns:
            GameAnalysis with all move evaluations
        """
        # Determine player color
        white_player = game.headers.get("White", "").lower()
        black_player = game.headers.get("Black", "").lower()
        player_color = "white" if player_username.lower() == white_player else "black"
        
        # Determine if player won
        result = game.headers.get("Result", "*")
        if player_color == "white":
            player_won = result == "1-0"
        else:
            player_won = result == "0-1"
        
        # Analyze each move
        board = game.board()
        moves_analysis = []
        critical_moments = []
        total_centipawn_loss = 0
        player_moves = 0
        blunders = 0
        mistakes = 0
        brilliancies = 0
        
        prev_eval = 0.0
        
        for i, move in enumerate(game.mainline_moves()):
            move_number = (i // 2) + 1
            is_white_move = board.turn == chess.WHITE
            is_player_move = (is_white_move and player_color == "white") or \
                           (not is_white_move and player_color == "black")
            
            # Get FEN before move
            fen_before = board.fen()
            
            # Make the move
            san = board.san(move)
            uci = move.uci()
            board.push(move)
            
            # Get FEN after move
            fen_after = board.fen()
            
            # Analyze with Stockfish
            if self.stockfish.is_available():
                eval_before, eval_after, quality = self.stockfish.analyze_move(
                    fen_before, uci, fen_after, is_white_move
                )
            else:
                # Fallback: no analysis
                eval_before = prev_eval
                eval_after = prev_eval
                quality = 'good'
            
            eval_change = eval_after - eval_before
            if not is_white_move:
                eval_change = -eval_change
            
            # Generate caption for critical moves
            caption = ""
            is_critical = False
            
            if quality == 'brilliant':
                caption = f"💎 BRILLIANT! {san}"
                is_critical = True
                brilliancies += 1
            elif quality == 'blunder':
                caption = f"❌ BLUNDER! {san}"
                is_critical = True
                blunders += 1
            elif quality == 'mistake':
                caption = f"⚠️ Mistake: {san}"
                is_critical = True
                mistakes += 1
            elif quality == 'great' and is_player_move:
                caption = f"🔥 Great move! {san}"
                is_critical = True
            
            # Check for special situations
            if board.is_check():
                if not caption:
                    caption = f"♟️ Check! {san}"
                else:
                    caption += " CHECK!"
                is_critical = True
            
            if board.is_checkmate():
                caption = f"👑 CHECKMATE! {san}"
                is_critical = True
            
            # Track critical moments
            if is_critical:
                critical_moments.append(i)
            
            # Track centipawn loss for player moves
            if is_player_move and eval_change < 0:
                total_centipawn_loss += abs(eval_change)
                player_moves += 1
            elif is_player_move:
                player_moves += 1
            
            moves_analysis.append(MoveAnalysis(
                move_number=move_number,
                move_san=san,
                move_uci=uci,
                eval_before=eval_before,
                eval_after=eval_after,
                eval_change=eval_change,
                is_player_move=is_player_move,
                move_quality=quality,
                is_critical=is_critical,
                caption=caption
            ))
            
            prev_eval = eval_after
        
        # Calculate average centipawn loss
        avg_cpl = total_centipawn_loss / max(player_moves, 1)
        
        return GameAnalysis(
            moves=moves_analysis,
            player_color=player_color,
            player_won=player_won,
            critical_moments=critical_moments,
            average_centipawn_loss=avg_cpl,
            blunders=blunders,
            mistakes=mistakes,
            brilliancies=brilliancies
        )

# =============================================================================
# EVALUATION BAR RENDERER
# =============================================================================

class EvalBarRenderer:
    """
    Renders the Stockfish evaluation bar.
    """
    
    def __init__(
        self,
        bar_width: int = 40,
        bar_height: int = 800,
        max_eval: float = 1000  # Centipawns for full bar
    ):
        self.bar_width = bar_width
        self.bar_height = bar_height
        self.max_eval = max_eval
    
    def render(self, evaluation: float) -> Image.Image:
        """
        Render evaluation bar.
        
        Args:
            evaluation: Centipawn evaluation (positive = white advantage)
            
        Returns:
            PIL Image of the evaluation bar
        """
        # Create bar image
        bar = Image.new('RGBA', (self.bar_width, self.bar_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bar)
        
        # Draw border
        draw.rectangle(
            [0, 0, self.bar_width - 1, self.bar_height - 1],
            outline=COLORS['eval_border'],
            width=2
        )
        
        # Calculate white portion (0.5 = equal, 1.0 = white winning)
        normalized = (evaluation / self.max_eval) / 2 + 0.5
        normalized = max(0.05, min(0.95, normalized))  # Clamp
        
        white_height = int(self.bar_height * (1 - normalized))
        
        # Draw black portion (top)
        draw.rectangle(
            [2, 2, self.bar_width - 3, white_height],
            fill=COLORS['eval_black']
        )
        
        # Draw white portion (bottom)
        draw.rectangle(
            [2, white_height, self.bar_width - 3, self.bar_height - 3],
            fill=COLORS['eval_white']
        )
        
        # Add evaluation text
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()
        
        # Format evaluation
        if abs(evaluation) >= 9000:
            if evaluation > 0:
                eval_text = "M" + str(int((10000 - abs(evaluation)) / 10))
            else:
                eval_text = "-M" + str(int((10000 - abs(evaluation)) / 10))
        else:
            eval_text = f"{evaluation/100:+.1f}"
        
        # Draw text at center
        text_bbox = draw.textbbox((0, 0), eval_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (self.bar_width - text_width) // 2
        
        # Choose text color based on position
        if normalized > 0.5:
            text_y = self.bar_height // 4
            text_color = COLORS['eval_white']
        else:
            text_y = 3 * self.bar_height // 4
            text_color = COLORS['eval_black']
        
        # Add background for text
        draw.rectangle(
            [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + 16],
            fill=(128, 128, 128, 200)
        )
        draw.text((text_x, text_y), eval_text, fill=text_color, font=font)
        
        return bar

# =============================================================================
# CAPTION RENDERER
# =============================================================================

class CaptionRenderer:
    """
    Renders captions for critical moves.
    """
    
    def __init__(self, video_width: int = VIDEO_WIDTH, video_height: int = VIDEO_HEIGHT):
        self.video_width = video_width
        self.video_height = video_height
        
        # Try to load fonts
        self.fonts = {}
        try:
            self.fonts['large'] = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
            self.fonts['medium'] = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
            self.fonts['small'] = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            self.fonts['large'] = ImageFont.load_default()
            self.fonts['medium'] = ImageFont.load_default()
            self.fonts['small'] = ImageFont.load_default()
    
    def render_caption(
        self,
        text: str,
        move_quality: str = 'good',
        position: str = 'bottom'  # 'top', 'center', 'bottom'
    ) -> Image.Image:
        """
        Render a caption overlay.
        
        Args:
            text: Caption text
            move_quality: Quality for color coding
            position: Where to place caption
            
        Returns:
            PIL Image (RGBA) with transparent background
        """
        # Create transparent image
        caption_img = Image.new('RGBA', (self.video_width, 120), (0, 0, 0, 0))
        draw = ImageDraw.Draw(caption_img)
        
        # Get quality color
        quality_colors = {
            'brilliant': COLORS['brilliant'],
            'great': COLORS['great'],
            'best': COLORS['best'],
            'good': COLORS['good'],
            'inaccuracy': COLORS['inaccuracy'],
            'mistake': COLORS['mistake'],
            'blunder': COLORS['blunder'],
        }
        accent_color = quality_colors.get(move_quality, COLORS['good'])
        
        # Calculate text size
        font = self.fonts['large']
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Center text
        x = (self.video_width - text_width) // 2
        y = (120 - text_height) // 2
        
        # Draw background pill
        padding = 20
        pill_rect = [
            x - padding,
            y - padding // 2,
            x + text_width + padding,
            y + text_height + padding // 2
        ]
        
        # Draw rounded rectangle background
        draw.rounded_rectangle(
            pill_rect,
            radius=15,
            fill=(0, 0, 0, 200)
        )
        
        # Draw accent bar on left
        draw.rectangle(
            [pill_rect[0], pill_rect[1], pill_rect[0] + 5, pill_rect[3]],
            fill=accent_color
        )
        
        # Draw text
        draw.text((x, y), text, fill=COLORS['caption_text'], font=font)
        
        return caption_img

# =============================================================================
# THUMBNAIL GENERATOR
# =============================================================================

class ThumbnailGenerator:
    """
    Generates eye-catching thumbnails for YouTube.
    """
    
    def __init__(self, width: int = 1280, height: int = 720):
        """YouTube thumbnail dimensions."""
        self.width = width
        self.height = height
        
        # Load fonts
        try:
            self.title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
            self.subtitle_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        except:
            self.title_font = ImageFont.load_default()
            self.subtitle_font = ImageFont.load_default()
    
    def generate(
        self,
        board: chess.Board,
        title: str,
        subtitle: str = "",
        player_won: bool = True,
        highlight_square: Optional[str] = None
    ) -> Image.Image:
        """
        Generate a thumbnail.
        
        Args:
            board: Chess board position
            title: Main title text
            subtitle: Secondary text
            player_won: Whether player won (affects color scheme)
            highlight_square: Square to highlight (e.g., "e4")
            
        Returns:
            PIL Image thumbnail
        """
        # Create base image with gradient background
        thumbnail = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(thumbnail)
        
        # Draw gradient background
        if player_won:
            color1, color2 = (20, 60, 20), (40, 100, 40)  # Green for win
        else:
            color1, color2 = (60, 20, 20), (100, 40, 40)  # Red for loss
        
        for y in range(self.height):
            ratio = y / self.height
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            draw.line([(0, y), (self.width, y)], fill=(r, g, b))
        
        # Render chess board (smaller, on the left)
        board_size = 400
        board_img = self._render_mini_board(board, board_size, highlight_square)
        
        # Position board on left side
        board_x = 50
        board_y = (self.height - board_size) // 2
        thumbnail.paste(board_img, (board_x, board_y))
        
        # Add title on right side
        text_x = board_x + board_size + 50
        text_y = self.height // 3
        
        # Draw title with shadow
        shadow_offset = 3
        draw.text((text_x + shadow_offset, text_y + shadow_offset), title, 
                  fill=(0, 0, 0), font=self.title_font)
        draw.text((text_x, text_y), title, fill=(255, 255, 255), font=self.title_font)
        
        # Draw subtitle
        if subtitle:
            sub_y = text_y + 100
            draw.text((text_x + shadow_offset, sub_y + shadow_offset), subtitle,
                     fill=(0, 0, 0), font=self.subtitle_font)
            draw.text((text_x, sub_y), subtitle, fill=(200, 200, 200), font=self.subtitle_font)
        
        # Add result badge
        badge_text = "WIN! 🏆" if player_won else "LOSS 📚"
        badge_color = (50, 200, 50) if player_won else (200, 50, 50)
        
        badge_bbox = draw.textbbox((0, 0), badge_text, font=self.subtitle_font)
        badge_width = badge_bbox[2] - badge_bbox[0]
        badge_x = self.width - badge_width - 50
        badge_y = 50
        
        # Draw badge background
        draw.rounded_rectangle(
            [badge_x - 20, badge_y - 10, badge_x + badge_width + 20, badge_y + 60],
            radius=10,
            fill=badge_color
        )
        draw.text((badge_x, badge_y), badge_text, fill=(255, 255, 255), font=self.subtitle_font)
        
        return thumbnail
    
    def _render_mini_board(
        self,
        board: chess.Board,
        size: int,
        highlight_square: Optional[str] = None
    ) -> Image.Image:
        """Render a small chess board for thumbnail."""
        square_size = size // 8
        img = Image.new('RGB', (size, size))
        draw = ImageDraw.Draw(img)
        
        # Unicode pieces
        piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        try:
            piece_font = ImageFont.truetype("/System/Library/Fonts/Apple Symbols.ttf", square_size - 8)
        except:
            piece_font = ImageFont.load_default()
        
        for row in range(8):
            for col in range(8):
                x = col * square_size
                y = row * square_size
                
                # Square color
                is_light = (row + col) % 2 == 0
                color = COLORS['white_square'] if is_light else COLORS['black_square']
                
                # Highlight square
                square_name = chess.square_name(chess.square(col, 7 - row))
                if highlight_square and square_name == highlight_square:
                    color = (255, 255, 100)
                
                draw.rectangle([x, y, x + square_size, y + square_size], fill=color)
                
                # Draw piece
                square = chess.square(col, 7 - row)
                piece = board.piece_at(square)
                if piece:
                    symbol = piece_symbols.get(piece.symbol(), '')
                    # Center piece
                    bbox = draw.textbbox((0, 0), symbol, font=piece_font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    px = x + (square_size - tw) // 2
                    py = y + (square_size - th) // 2 - 5
                    
                    # Draw shadow
                    draw.text((px + 2, py + 2), symbol, fill=(50, 50, 50), font=piece_font)
                    draw.text((px, py), symbol, fill=(255, 255, 255) if piece.color else (0, 0, 0), font=piece_font)
        
        return img

# =============================================================================
# BACKGROUND MUSIC HANDLER
# =============================================================================

class BackgroundMusicHandler:
    """
    Handles adding background music to videos.
    """
    
    # Royalty-free music sources (URLs to download)
    MUSIC_SOURCES = {
        'epic': 'https://www.soundjay.com/free-music/sounds/epic-battle-153797.mp3',
        'calm': 'https://www.soundjay.com/free-music/sounds/piano-meditation-138752.mp3',
        'tense': 'https://www.soundjay.com/free-music/sounds/suspense-action-142823.mp3',
    }
    
    def __init__(self, music_dir: Path = Path("./music")):
        self.music_dir = music_dir
        self.music_dir.mkdir(exist_ok=True)
    
    def add_music_to_video(
        self,
        video_path: str,
        output_path: str,
        music_file: Optional[str] = None,
        volume: float = 0.3
    ) -> bool:
        """
        Add background music to a video.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            music_file: Path to music file (optional)
            volume: Music volume (0.0 - 1.0)
            
        Returns:
            True if successful
        """
        if not MOVIEPY_AVAILABLE:
            logger.warning("⚠️ moviepy not available, skipping music")
            return False
        
        try:
            # Load video
            video = VideoFileClip(video_path)
            
            if music_file and os.path.exists(music_file):
                # Load and process audio
                audio = AudioFileClip(music_file)
                
                # Loop audio if shorter than video
                if audio.duration < video.duration:
                    loops_needed = int(video.duration / audio.duration) + 1
                    audio = audio.loop(loops_needed)
                
                # Trim to video length
                audio = audio.subclip(0, video.duration)
                
                # Reduce volume
                audio = audio.volumex(volume)
                
                # Add fade out at end
                audio = audio.audio_fadeout(2)
                
                # Set audio to video
                video = video.set_audio(audio)
            
            # Write output
            video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=FPS
            )
            
            video.close()
            if music_file:
                audio.close()
            
            logger.info(f"✅ Added background music to video")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add music: {e}")
            return False

# =============================================================================
# ENHANCED VIDEO GENERATOR
# =============================================================================

class EnhancedVideoGenerator:
    """
    Main class that generates enhanced chess videos with all features.
    """
    
    def __init__(
        self,
        player_username: str,
        output_dir: Path = Path("./videos"),
        enable_stockfish: bool = True,
        enable_music: bool = False,
        stockfish_depth: int = 12
    ):
        self.player_username = player_username.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        if enable_stockfish:
            self.stockfish = StockfishAnalyzer(depth=stockfish_depth)
        else:
            self.stockfish = StockfishAnalyzer(stockfish_path="/nonexistent")
        
        self.game_analyzer = GameAnalyzer(self.stockfish)
        self.eval_bar = EvalBarRenderer()
        self.caption_renderer = CaptionRenderer()
        self.thumbnail_gen = ThumbnailGenerator()
        self.music_handler = BackgroundMusicHandler()
        
        # Video settings
        self.fps = FPS
        self.seconds_per_move = 1.5
        
        logger.info(f"🎬 Enhanced Video Generator initialized")
        logger.info(f"   Stockfish: {'✅' if self.stockfish.is_available() else '❌'}")
        logger.info(f"   Output: {self.output_dir}")
    
    def generate_video(
        self,
        pgn_path: str,
        output_filename: Optional[str] = None,
        max_moves: Optional[int] = None,
        add_music: bool = False,
        music_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate an enhanced video from a PGN file.
        
        Args:
            pgn_path: Path to PGN file
            output_filename: Custom output filename
            max_moves: Maximum moves to include
            add_music: Whether to add background music
            music_file: Path to music file
            
        Returns:
            Path to generated video
        """
        logger.info(f"🎬 Starting enhanced video generation")
        logger.info(f"   PGN: {pgn_path}")
        
        # Parse PGN
        try:
            with open(pgn_path) as f:
                game = chess.pgn.read_game(f)
            if not game:
                logger.error("❌ Failed to parse PGN")
                return None
        except Exception as e:
            logger.error(f"❌ PGN error: {e}")
            return None
        
        # Analyze game
        logger.info("🔍 Analyzing game with Stockfish...")
        analysis = self.game_analyzer.analyze_game(game, self.player_username)
        
        logger.info(f"📊 Analysis complete:")
        logger.info(f"   Brilliancies: {analysis.brilliancies}")
        logger.info(f"   Blunders: {analysis.blunders}")
        logger.info(f"   Avg CPL: {analysis.average_centipawn_loss:.1f}")
        
        # Generate frames
        logger.info("🎞️ Generating enhanced frames...")
        frames = self._generate_frames(game, analysis, max_moves)
        
        if not frames:
            logger.error("❌ No frames generated")
            return None
        
        # Create video
        if output_filename:
            video_path = self.output_dir / f"{output_filename}.mp4"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result = "win" if analysis.player_won else "loss"
            video_path = self.output_dir / f"enhanced_{result}_{timestamp}.mp4"
        
        logger.info(f"🎥 Encoding video...")
        self._encode_video(frames, str(video_path))
        
        # Add music if requested
        if add_music and music_file:
            temp_path = str(video_path).replace('.mp4', '_temp.mp4')
            os.rename(video_path, temp_path)
            self.music_handler.add_music_to_video(temp_path, str(video_path), music_file)
            os.remove(temp_path)
        
        # Generate thumbnail
        thumbnail_path = video_path.with_suffix('.jpg')
        self._generate_thumbnail(game, analysis, str(thumbnail_path))
        
        logger.info(f"✅ Enhanced video complete!")
        logger.info(f"   Video: {video_path}")
        logger.info(f"   Thumbnail: {thumbnail_path}")
        
        return str(video_path)
    
    def _generate_frames(
        self,
        game: chess.pgn.Game,
        analysis: GameAnalysis,
        max_moves: Optional[int] = None
    ) -> List[np.ndarray]:
        """Generate all video frames with evaluation bar and captions."""
        frames = []
        board = game.board()
        moves = list(game.mainline_moves())
        
        if max_moves:
            moves = moves[:max_moves]
        
        # Limit to 59 seconds
        max_possible_moves = int((59 - 2) / self.seconds_per_move)
        if len(moves) > max_possible_moves:
            moves = moves[:max_possible_moves]
            logger.info(f"📋 Limited to {len(moves)} moves for 59s video")
        
        frames_per_move = int(self.fps * self.seconds_per_move)
        
        # Generate initial position frames
        for _ in range(self.fps):  # 1 second intro
            frame = self._render_frame(board, 0, None, analysis.moves[0] if analysis.moves else None)
            frames.append(frame)
        
        # Generate move frames
        for i, move in enumerate(moves):
            if i >= len(analysis.moves):
                break
            
            move_analysis = analysis.moves[i]
            
            # Make the move
            board.push(move)
            
            # Render frames for this move
            for f in range(frames_per_move):
                # Show caption for first half of frames if critical move
                show_caption = f < frames_per_move // 2 and move_analysis.is_critical
                
                frame = self._render_frame(
                    board,
                    move_analysis.eval_after,
                    move_analysis if show_caption else None,
                    move_analysis
                )
                frames.append(frame)
            
            if (i + 1) % 10 == 0:
                logger.info(f"   Processed {i + 1}/{len(moves)} moves")
        
        # Add ending frames
        for _ in range(self.fps):  # 1 second outro
            frame = self._render_frame(board, analysis.moves[-1].eval_after if analysis.moves else 0, None, None)
            frames.append(frame)
        
        return frames
    
    def _render_frame(
        self,
        board: chess.Board,
        evaluation: float,
        caption_analysis: Optional[MoveAnalysis],
        move_analysis: Optional[MoveAnalysis]
    ) -> np.ndarray:
        """Render a single frame with all overlays."""
        # Create base frame
        frame = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), (30, 30, 30))
        
        # Render chess board
        board_size = 900
        board_img = self._render_board(board, board_size)
        
        # Position board
        board_x = (VIDEO_WIDTH - board_size - 60) // 2  # Leave room for eval bar
        board_y = 400
        frame.paste(board_img, (board_x, board_y))
        
        # Render evaluation bar
        eval_bar = self.eval_bar.render(evaluation)
        eval_x = board_x + board_size + 20
        eval_y = board_y + (board_size - self.eval_bar.bar_height) // 2
        frame.paste(eval_bar, (eval_x, eval_y), eval_bar)
        
        # Add move quality indicator
        if move_analysis:
            quality_img = self._render_quality_indicator(move_analysis)
            frame.paste(quality_img, (board_x, board_y - 80), quality_img)
        
        # Add caption if critical move
        if caption_analysis and caption_analysis.is_critical:
            caption_img = self.caption_renderer.render_caption(
                caption_analysis.caption,
                caption_analysis.move_quality
            )
            caption_y = board_y + board_size + 50
            frame.paste(caption_img, (0, caption_y), caption_img)
        
        # Add player info bar at top
        info_bar = self._render_info_bar(board)
        frame.paste(info_bar, (0, 50), info_bar)
        
        # Convert to numpy array for OpenCV
        return np.array(frame)[:, :, ::-1]  # RGB to BGR
    
    def _render_board(self, board: chess.Board, size: int) -> Image.Image:
        """Render the chess board."""
        square_size = size // 8
        img = Image.new('RGB', (size, size))
        draw = ImageDraw.Draw(img)
        
        piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        try:
            piece_font = ImageFont.truetype("/System/Library/Fonts/Apple Symbols.ttf", square_size - 15)
        except:
            piece_font = ImageFont.load_default()
        
        for row in range(8):
            for col in range(8):
                x = col * square_size
                y = row * square_size
                
                is_light = (row + col) % 2 == 0
                color = COLORS['white_square'] if is_light else COLORS['black_square']
                
                draw.rectangle([x, y, x + square_size, y + square_size], fill=color)
                
                square = chess.square(col, 7 - row)
                piece = board.piece_at(square)
                if piece:
                    symbol = piece_symbols.get(piece.symbol(), '')
                    bbox = draw.textbbox((0, 0), symbol, font=piece_font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    px = x + (square_size - tw) // 2
                    py = y + (square_size - th) // 2 - 8
                    
                    draw.text((px + 2, py + 2), symbol, fill=(50, 50, 50), font=piece_font)
                    piece_color = (255, 255, 255) if piece.color else (30, 30, 30)
                    draw.text((px, py), symbol, fill=piece_color, font=piece_font)
        
        return img
    
    def _render_quality_indicator(self, move_analysis: MoveAnalysis) -> Image.Image:
        """Render move quality indicator."""
        img = Image.new('RGBA', (VIDEO_WIDTH, 70), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        quality_colors = {
            'brilliant': COLORS['brilliant'],
            'great': COLORS['great'],
            'best': COLORS['best'],
            'good': COLORS['good'],
            'inaccuracy': COLORS['inaccuracy'],
            'mistake': COLORS['mistake'],
            'blunder': COLORS['blunder'],
        }
        
        quality_text = {
            'brilliant': '💎 BRILLIANT',
            'great': '🔥 GREAT',
            'best': '✅ BEST',
            'good': '👍 GOOD',
            'inaccuracy': '⚠️ INACCURACY',
            'mistake': '❌ MISTAKE',
            'blunder': '💀 BLUNDER',
        }
        
        color = quality_colors.get(move_analysis.move_quality, COLORS['good'])
        text = quality_text.get(move_analysis.move_quality, '')
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        except:
            font = ImageFont.load_default()
        
        # Draw indicator
        text_full = f"{text}  {move_analysis.move_san}"
        bbox = draw.textbbox((0, 0), text_full, font=font)
        text_width = bbox[2] - bbox[0]
        x = (VIDEO_WIDTH - text_width) // 2
        
        draw.rounded_rectangle(
            [x - 20, 10, x + text_width + 20, 60],
            radius=10,
            fill=(*color, 200)
        )
        draw.text((x, 15), text_full, fill=(255, 255, 255), font=font)
        
        return img
    
    def _render_info_bar(self, board: chess.Board) -> Image.Image:
        """Render top info bar with turn indicator."""
        img = Image.new('RGBA', (VIDEO_WIDTH, 80), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        except:
            font = ImageFont.load_default()
        
        # Turn indicator
        turn_text = "White to move" if board.turn else "Black to move"
        turn_color = (255, 255, 255) if board.turn else (100, 100, 100)
        
        draw.rounded_rectangle([VIDEO_WIDTH // 2 - 100, 10, VIDEO_WIDTH // 2 + 100, 60],
                               radius=10, fill=(50, 50, 50, 200))
        
        bbox = draw.textbbox((0, 0), turn_text, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text((VIDEO_WIDTH // 2 - text_width // 2, 20), turn_text, fill=turn_color, font=font)
        
        return img
    
    def _encode_video(self, frames: List[np.ndarray], output_path: str):
        """Encode frames to video file."""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # Get file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        duration = len(frames) / self.fps
        logger.info(f"💾 Saved: {output_path}")
        logger.info(f"📁 Size: {size_mb:.2f} MB")
        logger.info(f"⏱️ Duration: ~{duration:.1f} seconds")
    
    def _generate_thumbnail(
        self,
        game: chess.pgn.Game,
        analysis: GameAnalysis,
        output_path: str
    ):
        """Generate video thumbnail."""
        # Get position at a critical moment or near end
        board = game.board()
        moves = list(game.mainline_moves())
        
        # Find a good position for thumbnail (near critical moment or 2/3 through)
        target_move = len(moves) * 2 // 3
        if analysis.critical_moments:
            target_move = min(analysis.critical_moments[-1], len(moves) - 1)
        
        for i, move in enumerate(moves[:target_move + 1]):
            board.push(move)
        
        # Generate title
        opening = game.headers.get("ECO", "") + " " + game.headers.get("Opening", "Chess Game")
        opening = opening.strip()[:30]
        
        result_text = "WIN!" if analysis.player_won else "LOSS"
        
        thumbnail = self.thumbnail_gen.generate(
            board=board,
            title=result_text,
            subtitle=opening,
            player_won=analysis.player_won
        )
        
        thumbnail.save(output_path, quality=95)
        logger.info(f"🖼️ Thumbnail saved: {output_path}")

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Chess Video Generator")
    parser.add_argument('pgn_file', help='Path to PGN file')
    parser.add_argument('--output', '-o', help='Output filename (without extension)')
    parser.add_argument('--player', '-p', default='ChessBeast_37', help='Player username')
    parser.add_argument('--max-moves', '-m', type=int, help='Maximum moves to include')
    parser.add_argument('--no-stockfish', action='store_true', help='Disable Stockfish analysis')
    parser.add_argument('--depth', '-d', type=int, default=12, help='Stockfish analysis depth')
    parser.add_argument('--music', help='Path to background music file')
    
    args = parser.parse_args()
    
    generator = EnhancedVideoGenerator(
        player_username=args.player,
        enable_stockfish=not args.no_stockfish,
        stockfish_depth=args.depth
    )
    
    result = generator.generate_video(
        pgn_path=args.pgn_file,
        output_filename=args.output,
        max_moves=args.max_moves,
        add_music=bool(args.music),
        music_file=args.music
    )
    
    if result:
        print(f"\n✅ Video generated: {result}")
        return 0
    else:
        print("\n❌ Video generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
