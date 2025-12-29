#!/usr/bin/env python3
"""
Enhanced Chess Video Generator - ChessBeast Edition
=======================================================
Customized video generation with:
1. Evaluation bar for each move
2. Move numbers clearly visible
3. Bold chess pieces
4. "ChessBeast" as player name, "Opponent" for opponent
5. No move quality labels
6. Final result at end (1-0 or 0-1)
7. Background music (dramatic tension)
8. Sound effects (move, capture, check, checkmate)
9. Move arrows showing piece movement
10. Captured pieces display

Author: ChessBeast Chess Shorts Automation
"""

import os
import sys
import logging
import json
import tempfile
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

import chess
import chess.pgn
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Video processing
import cv2

# Audio processing
try:
    from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips
    MOVIEPY_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older moviepy versions
        from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
        MOVIEPY_AVAILABLE = True
    except ImportError:
        MOVIEPY_AVAILABLE = False

# Stockfish integration
try:
    from stockfish import Stockfish
    STOCKFISH_AVAILABLE = True
except ImportError:
    STOCKFISH_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

# Video dimensions (YouTube Shorts)
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30

# Audio paths
AUDIO_DIR = Path(__file__).parent / "audio"
SFX_DIR = AUDIO_DIR / "sfx"
MUSIC_DIR = AUDIO_DIR / "music"

# Colors - Board stays same, pieces more bold
COLORS = {
    'white_square': (240, 217, 181),
    'black_square': (181, 136, 99),
    'last_move_from': (186, 202, 68),     # Highlight last move
    'last_move_to': (246, 246, 105),
    'check_square': (235, 97, 80),         # Red for check
    'eval_white': (255, 255, 255),
    'eval_black': (40, 40, 40),
    'eval_border': (80, 80, 80),
    'background': (28, 28, 28),
    'text_primary': (255, 255, 255),
    'text_secondary': (180, 180, 180),
    'player_bar_bg': (45, 45, 45),
    'win_color': (76, 175, 80),            # Green
    'loss_color': (244, 67, 54),           # Red
    'white_piece': (255, 255, 255),        # White pieces
    'black_piece': (30, 30, 30),           # Black pieces (slightly lighter)
    'white_outline': (60, 60, 60),         # Dark outline for white pieces (visibility)
    'black_outline': (0, 0, 0),            # Subtle outline for black pieces
    'arrow_color': (255, 170, 0, 180),     # Orange arrow with transparency
    'captured_bg': (35, 35, 35),           # Background for captured pieces
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ChessBeastVideo")

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MoveData:
    """Data for a single move."""
    move_number: int
    move_san: str
    move_uci: str
    is_white_move: bool
    eval_after: float
    from_square: int
    to_square: int
    is_check: bool
    is_checkmate: bool
    is_capture: bool = False
    captured_piece: Optional[str] = None

@dataclass  
class GameData:
    """Data for the entire game."""
    moves: List[MoveData]
    player_color: str
    player_won: bool
    result: str  # "1-0", "0-1", "1/2-1/2"
    opening: str
    white_player: str
    black_player: str
    termination: str  # How the game ended: checkmate, timeout, resignation, etc.
    captured_by_white: List[str] = field(default_factory=list)  # Pieces captured by white
    captured_by_black: List[str] = field(default_factory=list)  # Pieces captured by black

# =============================================================================
# STOCKFISH ANALYZER (Simplified)
# =============================================================================

class StockfishAnalyzer:
    """Simple Stockfish evaluator - just gets position evaluation."""
    
    def __init__(self, stockfish_path: str = STOCKFISH_PATH, depth: int = 12):
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
                parameters={"Threads": 2, "Hash": 128}
            )
            logger.info(f"✅ Stockfish initialized (depth={depth})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Stockfish: {e}")
    
    def is_available(self) -> bool:
        return self.stockfish is not None
    
    def evaluate(self, fen: str) -> float:
        """Get evaluation in centipawns. Positive = white advantage."""
        if not self.stockfish:
            return 0.0
        
        try:
            self.stockfish.set_fen_position(fen)
            evaluation = self.stockfish.get_evaluation()
            
            if evaluation["type"] == "cp":
                return evaluation["value"]
            elif evaluation["type"] == "mate":
                mate_moves = evaluation["value"]
                return 10000 - (abs(mate_moves) * 10) if mate_moves > 0 else -10000 + (abs(mate_moves) * 10)
            return 0.0
        except:
            return 0.0

# =============================================================================
# GAME PARSER
# =============================================================================

class GameParser:
    """Parses PGN and extracts game data with evaluations."""
    
    def __init__(self, stockfish: Optional[StockfishAnalyzer] = None):
        self.stockfish = stockfish or StockfishAnalyzer()
    
    def parse(self, pgn_path: str, player_username: str) -> Optional[GameData]:
        """Parse a PGN file and return game data."""
        try:
            with open(pgn_path) as f:
                game = chess.pgn.read_game(f)
            if not game:
                return None
        except Exception as e:
            logger.error(f"❌ PGN error: {e}")
            return None
        
        # Get headers
        white_player = game.headers.get("White", "Unknown")
        black_player = game.headers.get("Black", "Unknown")
        result = game.headers.get("Result", "*")
        termination = game.headers.get("Termination", "")
        
        # Get full opening name from ECOUrl or Opening header
        eco_url = game.headers.get("ECOUrl", "")
        if eco_url and "/openings/" in eco_url:
            # Extract opening name from URL like "...openings/Benoni-Defense-3.e3..."
            opening_part = eco_url.split("/openings/")[-1]
            # Clean up: replace dashes with spaces, take main opening name
            parts = opening_part.split("-")
            # Take words until we hit a number (move notation)
            opening_words = []
            for p in parts:
                if p and not p[0].isdigit():
                    opening_words.append(p)
                else:
                    break
            opening = " ".join(opening_words[:4])  # Max 4 words
        else:
            opening = game.headers.get("Opening", game.headers.get("ECO", "Chess Game"))
        
        # Determine player color and win status
        player_color = "white" if player_username.lower() in white_player.lower() else "black"
        player_won = (result == "1-0" and player_color == "white") or \
                     (result == "0-1" and player_color == "black")
        
        # Parse moves and track captured pieces
        board = game.board()
        moves_data = []
        captured_by_white = []  # Pieces white has captured (black pieces)
        captured_by_black = []  # Pieces black has captured (white pieces)
        
        # Piece values for sorting captured pieces display
        piece_order = {'q': 0, 'r': 1, 'b': 2, 'n': 3, 'p': 4}
        
        for i, move in enumerate(game.mainline_moves()):
            move_number = (i // 2) + 1
            is_white_move = board.turn == chess.WHITE
            
            from_sq = move.from_square
            to_sq = move.to_square
            san = board.san(move)
            uci = move.uci()
            
            # Check for capture BEFORE making the move
            is_capture = board.is_capture(move)
            captured_piece = None
            if is_capture:
                # Get the piece being captured
                target_piece = board.piece_at(to_sq)
                if target_piece:
                    captured_piece = target_piece.symbol().lower()
                    if is_white_move:
                        captured_by_white.append(captured_piece)
                    else:
                        captured_by_black.append(captured_piece.upper())
                # Handle en passant
                elif board.is_en_passant(move):
                    captured_piece = 'p'
                    if is_white_move:
                        captured_by_white.append('p')
                    else:
                        captured_by_black.append('P')
            
            board.push(move)
            
            # Get evaluation after move
            # IMPORTANT: Stockfish returns eval from SIDE TO MOVE's perspective
            # After white moves, it's black's turn, so we need to NEGATE to get white's perspective
            # After black moves, it's white's turn, so eval is already from white's POV
            raw_eval = self.stockfish.evaluate(board.fen()) if self.stockfish.is_available() else 0.0
            
            # Normalize to WHITE's perspective (positive = white winning)
            # After white's move: board.turn is BLACK, so negate the eval
            # After black's move: board.turn is WHITE, so keep as is
            if board.turn == chess.BLACK:
                # It's black's turn, meaning white just moved
                # Stockfish gave eval from black's POV, so negate for white's POV
                eval_after = -raw_eval
            else:
                # It's white's turn, meaning black just moved
                # Stockfish gave eval from white's POV, so keep it
                eval_after = raw_eval
            
            moves_data.append(MoveData(
                move_number=move_number,
                move_san=san,
                move_uci=uci,
                is_white_move=is_white_move,
                eval_after=eval_after,
                from_square=from_sq,
                to_square=to_sq,
                is_check=board.is_check(),
                is_checkmate=board.is_checkmate(),
                is_capture=is_capture,
                captured_piece=captured_piece
            ))
        
        # Parse termination reason
        win_reason = "Game Over"
        if termination:
            term_lower = termination.lower()
            if "checkmate" in term_lower:
                win_reason = "Checkmate"
            elif "timeout" in term_lower:
                win_reason = "Timeout"
            elif "resign" in term_lower:
                win_reason = "Resignation"
            elif "abandon" in term_lower:
                win_reason = "Abandoned"
            elif "stalemate" in term_lower:
                win_reason = "Stalemate"
            elif "repetition" in term_lower:
                win_reason = "Repetition"
            elif "insufficient" in term_lower:
                win_reason = "Insufficient Material"
            elif "agreement" in term_lower:
                win_reason = "Draw by Agreement"
        
        return GameData(
            moves=moves_data,
            player_color=player_color,
            player_won=player_won,
            result=result,
            opening=opening[:35],
            white_player=white_player,
            black_player=black_player,
            termination=win_reason,
            captured_by_white=captured_by_white,
            captured_by_black=captured_by_black
        )

# =============================================================================
# EVALUATION BAR RENDERER
# =============================================================================

class EvalBarRenderer:
    """Renders vertical evaluation bar."""
    
    def __init__(self, width: int = 50, height: int = 800):
        self.width = width
        self.height = height
        self.max_eval = 800  # Centipawns for full bar
        
        try:
            self.font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            self.font = ImageFont.load_default()
    
    def render(self, evaluation: float) -> Image.Image:
        """
        Render evaluation bar like chess.com:
        - BLACK section at TOP of bar
        - WHITE section at BOTTOM of bar
        - Positive eval (+3.0, M2) = WHITE is winning = WHITE section grows UP
        - Negative eval (-3.0, -M2) = BLACK is winning = BLACK section grows DOWN
        """
        bar = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bar)
        
        # Border
        draw.rectangle([0, 0, self.width - 1, self.height - 1], 
                       outline=COLORS['eval_border'], width=2)
        
        # Clamp evaluation to display range
        clamped = max(-self.max_eval, min(self.max_eval, evaluation))
        
        # Calculate where the dividing line is (as percentage from TOP)
        # eval = +800 (white winning) → divider near TOP (black small at top, white big at bottom)
        # eval = 0    → divider at 50% (middle)
        # eval = -800 (black winning) → divider near BOTTOM (black big at top, white small at bottom)
        
        # black_portion = how much of bar is black (from top)
        # When white winning (+), black_portion should be SMALL
        # When black winning (-), black_portion should be LARGE
        black_portion = 0.5 - (clamped / (2 * self.max_eval))
        black_portion = max(0.05, min(0.95, black_portion))
        
        # Calculate the Y coordinate where black ends and white begins
        divider_y = int(self.height * black_portion)
        
        # Draw BLACK section (TOP - from top to divider)
        draw.rectangle([3, 3, self.width - 4, divider_y], fill=COLORS['eval_black'])
        
        # Draw WHITE section (BOTTOM - from divider to bottom)
        draw.rectangle([3, divider_y, self.width - 4, self.height - 4], fill=COLORS['eval_white'])
        
        # Evaluation text
        if abs(evaluation) >= 9000:
            mate_in = max(1, int((10000 - abs(evaluation)) / 10))
            eval_text = f"M{mate_in}" if evaluation > 0 else f"-M{mate_in}"
        else:
            eval_text = f"{evaluation/100:+.1f}"
        
        # Position text in the LARGER section for readability
        bbox = draw.textbbox((0, 0), eval_text, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_x = (self.width - text_width) // 2
        
        if evaluation >= 0:  # White winning or equal - text on white (bottom)
            text_y = self.height - 30
            text_color = COLORS['eval_black']
        else:  # Black winning - text on black (top)
            text_y = 10
            text_color = COLORS['eval_white']
        
        draw.text((text_x, text_y), eval_text, fill=text_color, font=self.font)
        
        return bar

# =============================================================================
# AUDIO MANAGER
# =============================================================================

class AudioManager:
    """Manages sound effects and background music for chess videos."""
    
    def __init__(self):
        self.available = MOVIEPY_AVAILABLE and SFX_DIR.exists()
        
        # Sound effect paths
        self.sfx = {
            'move': SFX_DIR / "move.wav",
            'capture': SFX_DIR / "capture.wav",
            'check': SFX_DIR / "check.wav",
            'checkmate': SFX_DIR / "checkmate.wav",
            'victory': SFX_DIR / "victory.wav"
        }
        
        # Music paths
        self.music = {
            'tension': MUSIC_DIR / "tension.wav",
            'victory': MUSIC_DIR / "victory.wav"
        }
        
        # Check what's available
        self.has_sfx = all(p.exists() for p in self.sfx.values())
        self.has_music = all(p.exists() for p in self.music.values())
        
        if self.available:
            logger.info(f"🎵 Audio: SFX={'✅' if self.has_sfx else '❌'} | Music={'✅' if self.has_music else '❌'}")
    
    def add_audio_to_video(
        self,
        video_path: str,
        moves: List[MoveData],
        fps: int,
        intro_frames: int,
        frames_per_move: int
    ) -> str:
        """Add sound effects and music to video, return path to new video."""
        if not self.available or not MOVIEPY_AVAILABLE:
            logger.warning("⚠️ MoviePy not available, skipping audio")
            return video_path
        
        try:
            # Load video
            video = VideoFileClip(video_path)
            audio_clips = []
            
            # Add background music (looped, lower volume)
            if self.has_music and self.music['tension'].exists():
                music = AudioFileClip(str(self.music['tension']))
                # Loop if video is longer than music
                if video.duration > music.duration:
                    loops_needed = int(video.duration / music.duration) + 1
                    music = music.with_effects([lambda c: c.loop(loops_needed)]) if hasattr(music, 'with_effects') else music.loop(loops_needed)
                # Trim to video duration and reduce volume
                music = music.with_end(video.duration).with_volume_scaled(0.25)
                audio_clips.append(music)
            
            # Add sound effects for each move
            if self.has_sfx:
                for i, move_data in enumerate(moves):
                    # Calculate start time for this move
                    move_start_time = (intro_frames + i * frames_per_move) / fps
                    
                    # Select appropriate sound
                    if move_data.is_checkmate:
                        sfx_path = self.sfx['checkmate']
                    elif move_data.is_check:
                        sfx_path = self.sfx['check']
                    elif move_data.is_capture:
                        sfx_path = self.sfx['capture']
                    else:
                        sfx_path = self.sfx['move']
                    
                    # Add sound effect
                    if sfx_path.exists():
                        sfx = AudioFileClip(str(sfx_path)).with_volume_scaled(0.7)
                        sfx = sfx.with_start(move_start_time)
                        audio_clips.append(sfx)
            
            # Add victory sound at the end (2 seconds before end)
            if self.has_sfx and self.sfx['victory'].exists():
                victory_time = max(0, video.duration - 2.5)
                victory = AudioFileClip(str(self.sfx['victory'])).with_volume_scaled(0.8)
                victory = victory.with_start(victory_time)
                audio_clips.append(victory)
            
            # Compose all audio
            if audio_clips:
                final_audio = CompositeAudioClip(audio_clips)
                video_with_audio = video.with_audio(final_audio)
                
                # Output to new file
                output_path = video_path.replace('.mp4', '_with_audio.mp4')
                video_with_audio.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    logger=None,  # Suppress moviepy output
                    fps=fps
                )
                
                # Clean up
                video.close()
                video_with_audio.close()
                for clip in audio_clips:
                    clip.close()
                
                # Replace original with audio version
                import shutil
                shutil.move(output_path, video_path)
                
                logger.info("🎵 Audio added successfully")
                return video_path
            
            video.close()
            return video_path
            
        except Exception as e:
            logger.warning(f"⚠️ Audio processing failed: {e}")
            import traceback
            traceback.print_exc()
            return video_path

# =============================================================================
# ARROW RENDERER  
# =============================================================================

class ArrowRenderer:
    """Renders move arrows on chess board."""
    
    @staticmethod
    def draw_arrow(
        draw: ImageDraw.Draw,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        color: Tuple[int, int, int, int] = COLORS['arrow_color'],
        width: int = 12
    ):
        """Draw an arrow from one position to another."""
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        # Calculate arrow direction
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        
        if length == 0:
            return
        
        # Normalize direction
        dx /= length
        dy /= length
        
        # Arrow head size
        head_length = 25
        head_width = 18
        
        # Calculate points for arrow head
        # Point where head starts on the line
        head_base_x = x2 - dx * head_length
        head_base_y = y2 - dy * head_length
        
        # Perpendicular direction
        px = -dy
        py = dx
        
        # Arrow head points
        left_x = head_base_x + px * head_width
        left_y = head_base_y + py * head_width
        right_x = head_base_x - px * head_width
        right_y = head_base_y - py * head_width
        
        # Draw arrow body (thick line)
        draw.line([(x1, y1), (head_base_x, head_base_y)], fill=color[:3], width=width)
        
        # Draw arrow head (triangle)
        draw.polygon([(x2, y2), (left_x, left_y), (right_x, right_y)], fill=color[:3])

# =============================================================================
# CHESSBEAST VIDEO GENERATOR
# =============================================================================

class ChessBeastVideoGenerator:
    """
    Main video generator - ChessBeast Edition v2
    
    Features:
    - Evaluation bar with each move
    - Move numbers clearly visible
    - Bold pieces
    - "ChessBeast" as player name
    - "Opponent" for opponent (no rating)
    - No move quality labels
    - Final result at end
    - Background music & sound effects
    - Move arrows
    - Captured pieces display
    """
    
    def __init__(
        self,
        player_username: str = "ChessBeast_37",
        display_name: str = "ChessBeast",
        output_dir: Path = Path("./videos"),
        enable_stockfish: bool = True,
        stockfish_depth: int = 12,
        enable_audio: bool = True
    ):
        self.player_username = player_username.lower()
        self.display_name = display_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if enable_stockfish:
            self.stockfish = StockfishAnalyzer(depth=stockfish_depth)
        else:
            self.stockfish = StockfishAnalyzer(stockfish_path="/nonexistent")
        
        self.parser = GameParser(self.stockfish)
        self.eval_bar = EvalBarRenderer(width=50, height=850)
        self.audio_manager = AudioManager() if enable_audio else None
        
        self.fps = FPS
        self.seconds_per_move = 1.2
        
        # Track captured pieces during rendering
        self.current_captured_white = []  # Captured BY white (black pieces)
        self.current_captured_black = []  # Captured BY black (white pieces)
        
        # Piece symbols for captured display
        self.capture_symbols = {
            'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
            'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙'
        }
        
        # Load fonts
        try:
            self.font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
            self.font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
            self.font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            self.font_move = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 72)
            self.piece_font = ImageFont.truetype("/System/Library/Fonts/Apple Symbols.ttf", 100)
            self.capture_font = ImageFont.truetype("/System/Library/Fonts/Apple Symbols.ttf", 36)
        except:
            self.font_large = ImageFont.load_default()
            self.font_medium = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            self.font_move = ImageFont.load_default()
            self.piece_font = ImageFont.load_default()
            self.capture_font = ImageFont.load_default()
        
        logger.info(f"🎬 ChessBeast Video Generator v2 initialized")
        logger.info(f"   Stockfish: {'✅' if self.stockfish.is_available() else '❌'}")
        logger.info(f"   Audio: {'✅' if self.audio_manager and self.audio_manager.available else '❌'}")
        logger.info(f"   Output: {self.output_dir}")
    
    def generate_video(
        self,
        pgn_path: str,
        output_filename: Optional[str] = None,
        max_moves: Optional[int] = None,
        wins_only: bool = True
    ) -> Optional[str]:
        """Generate video from PGN file."""
        logger.info(f"🎬 Starting video generation: {pgn_path}")
        
        # Parse game
        game_data = self.parser.parse(pgn_path, self.player_username)
        if not game_data:
            logger.error("❌ Failed to parse game")
            return None
        
        # Check if player won (skip if wins_only and not a win)
        if wins_only and not game_data.player_won:
            logger.warning("⚠️ Skipping - not a win")
            return None
        
        logger.info(f"✅ Game: {game_data.opening} - Result: {game_data.result}")
        logger.info(f"   Playing as: {game_data.player_color} | Won: {'Yes' if game_data.player_won else 'No'}")
        
        # Reset captured pieces tracking
        self.current_captured_white = []
        self.current_captured_black = []
        
        # Generate frames
        frames, frames_per_move = self._generate_frames(game_data, max_moves)
        
        if not frames:
            logger.error("❌ No frames generated")
            return None
        
        # Output path
        if output_filename:
            video_path = self.output_dir / f"{output_filename}.mp4"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = self.output_dir / f"chess_short_{self.display_name.lower()}_{timestamp}.mp4"
        
        # Encode video
        self._encode_video(frames, str(video_path))
        
        # Add audio (music + sound effects)
        if self.audio_manager and self.audio_manager.available:
            moves = game_data.moves[:max_moves] if max_moves else game_data.moves
            self.audio_manager.add_audio_to_video(
                str(video_path),
                moves,
                self.fps,
                intro_frames=self.fps,  # 1 second intro
                frames_per_move=frames_per_move
            )
        
        # Generate thumbnail
        thumbnail_path = video_path.with_suffix('.jpg')
        self._generate_thumbnail(game_data, str(thumbnail_path))
        
        # Save metadata
        self._save_metadata(game_data, str(video_path.with_suffix('.json')))
        
        logger.info(f"✅ Video complete: {video_path}")
        logger.info(f"🖼️ Thumbnail: {thumbnail_path}")
        return str(video_path)
    
    def _generate_frames(self, game_data: GameData, max_moves: Optional[int] = None) -> Tuple[List[np.ndarray], int]:
        """Generate all video frames. Returns (frames, frames_per_move)."""
        frames = []
        board = chess.Board()
        
        moves = game_data.moves
        if max_moves:
            moves = moves[:max_moves]
        
        # Adjust seconds per move to fit full game in ~57 seconds
        total_moves = len(moves)
        available_time = 57 - 3  # Leave 3s for intro/outro
        
        if total_moves * self.seconds_per_move > available_time:
            # Speed up to fit all moves
            self.seconds_per_move = available_time / total_moves
            logger.info(f"📋 Showing all {total_moves} moves (speed: {self.seconds_per_move:.2f}s per move)")
        else:
            logger.info(f"📋 Showing all {total_moves} moves")
        
        frames_per_move = int(self.fps * self.seconds_per_move)
        
        # Intro frames (1 second)
        for _ in range(self.fps):
            frame = self._render_frame(board, game_data, None, show_intro=True)
            frames.append(frame)
        
        # Move frames
        last_move_data = None
        for i, move_data in enumerate(moves):
            # Track captures for display
            if move_data.is_capture and move_data.captured_piece:
                if move_data.is_white_move:
                    self.current_captured_white.append(move_data.captured_piece)
                else:
                    self.current_captured_black.append(move_data.captured_piece.upper())
            
            move = chess.Move.from_uci(move_data.move_uci)
            board.push(move)
            
            for _ in range(frames_per_move):
                frame = self._render_frame(board, game_data, move_data)
                frames.append(frame)
            
            last_move_data = move_data
            
            if (i + 1) % 10 == 0:
                logger.info(f"   Processed {i + 1}/{len(moves)} moves")
        
        # Outro frames with result (2 seconds)
        for _ in range(self.fps * 2):
            frame = self._render_frame(board, game_data, last_move_data, show_result=True)
            frames.append(frame)
        
        return frames, frames_per_move
    
    def _render_frame(
        self,
        board: chess.Board,
        game_data: GameData,
        move_data: Optional[MoveData],
        show_intro: bool = False,
        show_result: bool = False
    ) -> np.ndarray:
        """Render a single frame."""
        frame = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), COLORS['background'])
        draw = ImageDraw.Draw(frame)
        
        # Layout
        board_size = 900
        board_x = 40
        board_y = 480
        eval_x = board_x + board_size + 30
        eval_y = board_y + 25
        
        # Flip board if player is black
        flip_board = game_data.player_color == "black"
        
        # === HEADER: Opening name ===
        draw.text((VIDEO_WIDTH // 2, 60), game_data.opening, 
                  fill=COLORS['text_secondary'], font=self.font_medium, anchor="mm")
        
        # === OPPONENT BAR (top) - show what opponent captured (white pieces if we're white) ===
        if game_data.player_color == "white":
            opponent_captures = self.current_captured_black  # Black captured white pieces
        else:
            opponent_captures = self.current_captured_white  # White captured black pieces (shown as white)
        self._draw_player_bar(draw, "Opponent", is_top=True, is_player=False, captured=opponent_captures)
        
        # === CHESS BOARD with ARROW ===
        board_img = self._render_board(board, board_size, move_data, flip_board)
        frame.paste(board_img, (board_x, board_y))
        
        # === PLAYER BAR (bottom) - show what we captured ===
        if game_data.player_color == "white":
            player_captures = self.current_captured_white  # White captured black pieces
        else:
            player_captures = self.current_captured_black  # Black captured white pieces (shown as black)
        self._draw_player_bar(draw, self.display_name, is_top=False, is_player=True, captured=player_captures)
        
        # === EVALUATION BAR ===
        if move_data:
            eval_bar = self.eval_bar.render(move_data.eval_after)
            frame.paste(eval_bar, (eval_x, eval_y), eval_bar)
        else:
            eval_bar = self.eval_bar.render(0)
            frame.paste(eval_bar, (eval_x, eval_y), eval_bar)
        
        # === MOVE NUMBER & NOTATION (large, clear) ===
        if move_data and not show_intro and not show_result:
            if move_data.is_white_move:
                move_text = f"{move_data.move_number}. {move_data.move_san}"
            else:
                move_text = f"{move_data.move_number}...{move_data.move_san}"
            draw.text((VIDEO_WIDTH // 2, 1580), move_text,
                      fill=COLORS['text_primary'], font=self.font_move, anchor="mm")
        
        # === INTRO OVERLAY ===
        if show_intro:
            # Semi-transparent overlay
            overlay = Image.new('RGBA', (VIDEO_WIDTH, 200), (0, 0, 0, 200))
            frame_rgba = frame.convert('RGBA')
            frame_rgba.paste(overlay, (0, VIDEO_HEIGHT // 2 - 100), overlay)
            frame = frame_rgba.convert('RGB')
            draw = ImageDraw.Draw(frame)
            
            draw.text((VIDEO_WIDTH // 2, VIDEO_HEIGHT // 2 - 30), "♟️ Chess Short",
                      fill=COLORS['text_primary'], font=self.font_large, anchor="mm")
            draw.text((VIDEO_WIDTH // 2, VIDEO_HEIGHT // 2 + 40), f"by {self.display_name}",
                      fill=COLORS['text_secondary'], font=self.font_medium, anchor="mm")
        
        # === RESULT OVERLAY (at end) ===
        if show_result:
            # Darken background
            overlay = Image.new('RGBA', (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 180))
            frame = Image.alpha_composite(frame.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(frame)
            
            # Result badge
            result_color = COLORS['win_color'] if game_data.player_won else COLORS['loss_color']
            badge_y = VIDEO_HEIGHT // 2 - 80
            
            draw.rounded_rectangle([VIDEO_WIDTH // 2 - 200, badge_y - 60, 
                                   VIDEO_WIDTH // 2 + 200, badge_y + 60],
                                  radius=20, fill=result_color)
            
            result_text = "VICTORY!" if game_data.player_won else "DEFEAT"
            draw.text((VIDEO_WIDTH // 2, badge_y), result_text,
                      fill=COLORS['text_primary'], font=self.font_large, anchor="mm")
            
            # Score (1-0, 0-1, etc.)
            draw.text((VIDEO_WIDTH // 2, badge_y + 120), game_data.result,
                      fill=COLORS['text_primary'], font=self.font_move, anchor="mm")
            
            # How the game ended (Checkmate, Timeout, etc.)
            draw.text((VIDEO_WIDTH // 2, badge_y + 200), f"by {game_data.termination}",
                      fill=COLORS['text_secondary'], font=self.font_medium, anchor="mm")
        
        return np.array(frame)[:, :, ::-1]  # RGB to BGR
    
    def _draw_player_bar(self, draw: ImageDraw.Draw, name: str, is_top: bool, is_player: bool, captured: List[str] = None):
        """Draw player name bar with captured pieces."""
        if is_top:
            y = 140
        else:
            y = 1420
        
        # Background
        draw.rounded_rectangle([40, y, VIDEO_WIDTH - 120, y + 70], radius=10, 
                               fill=COLORS['player_bar_bg'])
        
        # Player icon (circle)
        icon_x = 70
        icon_color = COLORS['win_color'] if is_player else COLORS['text_secondary']
        draw.ellipse([icon_x, y + 15, icon_x + 40, y + 55], fill=icon_color)
        
        # Name (no rating shown)
        name_x = icon_x + 60
        draw.text((name_x, y + 35), name, fill=COLORS['text_primary'], 
                  font=self.font_medium, anchor="lm")
        
        # Draw captured pieces (right side of bar)
        if captured:
            # Sort by piece value (queens first)
            piece_order = {'q': 0, 'Q': 0, 'r': 1, 'R': 1, 'b': 2, 'B': 2, 'n': 3, 'N': 3, 'p': 4, 'P': 4}
            sorted_captures = sorted(captured, key=lambda x: piece_order.get(x, 5))
            
            # Draw captured pieces
            cap_x = VIDEO_WIDTH - 180
            for piece in sorted_captures[-8:]:  # Show last 8 max
                symbol = self.capture_symbols.get(piece, piece)
                draw.text((cap_x, y + 35), symbol, fill=COLORS['text_secondary'], 
                         font=self.capture_font, anchor="mm")
                cap_x -= 30
    
    def _render_board(
        self,
        board: chess.Board,
        size: int,
        move_data: Optional[MoveData],
        flip: bool = False
    ) -> Image.Image:
        """Render chess board with bold pieces and move arrow."""
        square_size = size // 8
        img = Image.new('RGB', (size, size))
        draw = ImageDraw.Draw(img)
        
        # Chess piece symbols
        piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        for row in range(8):
            for col in range(8):
                # Handle board flip
                display_row = 7 - row if flip else row
                display_col = 7 - col if flip else col
                
                x = col * square_size
                y = row * square_size
                
                # Square color (original colors kept)
                is_light = (display_row + display_col) % 2 == 0
                color = COLORS['white_square'] if is_light else COLORS['black_square']
                
                # Highlight last move
                square = chess.square(display_col, 7 - display_row)
                if move_data:
                    if square == move_data.from_square:
                        color = COLORS['last_move_from']
                    elif square == move_data.to_square:
                        color = COLORS['last_move_to']
                
                # Highlight check
                if board.is_check():
                    king_square = board.king(board.turn)
                    if square == king_square:
                        color = COLORS['check_square']
                
                draw.rectangle([x, y, x + square_size, y + square_size], fill=color)
                
                # Draw piece
                piece = board.piece_at(square)
                if piece:
                    symbol = piece_symbols.get(piece.symbol(), '')
                    bbox = draw.textbbox((0, 0), symbol, font=self.piece_font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    px = x + (square_size - tw) // 2
                    py = y + (square_size - th) // 2 - 10
                    
                    if piece.color:  # White pieces
                        # Dark outline for white pieces (makes them visible on light squares)
                        outline_color = COLORS['white_outline']
                        for ox, oy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                            draw.text((px + ox, py + oy), symbol, fill=outline_color, font=self.piece_font)
                        # White piece fill
                        draw.text((px, py), symbol, fill=COLORS['white_piece'], font=self.piece_font)
                    else:  # Black pieces
                        # Very subtle outline for black pieces
                        outline_color = COLORS['black_outline']
                        for ox, oy in [(-1, 0), (1, 0)]:
                            draw.text((px + ox, py + oy), symbol, fill=outline_color, font=self.piece_font)
                        # Black piece fill
                        draw.text((px, py), symbol, fill=COLORS['black_piece'], font=self.piece_font)
        
        # === DRAW MOVE ARROW ===
        if move_data:
            from_file = chess.square_file(move_data.from_square)
            from_rank = chess.square_rank(move_data.from_square)
            to_file = chess.square_file(move_data.to_square)
            to_rank = chess.square_rank(move_data.to_square)
            
            # Convert to pixel coordinates (accounting for flip)
            if flip:
                from_x = (7 - from_file) * square_size + square_size // 2
                from_y = from_rank * square_size + square_size // 2
                to_x = (7 - to_file) * square_size + square_size // 2
                to_y = to_rank * square_size + square_size // 2
            else:
                from_x = from_file * square_size + square_size // 2
                from_y = (7 - from_rank) * square_size + square_size // 2
                to_x = to_file * square_size + square_size // 2
                to_y = (7 - to_rank) * square_size + square_size // 2
            
            ArrowRenderer.draw_arrow(draw, (from_x, from_y), (to_x, to_y))
        
        return img
    
    def _encode_video(self, frames: List[np.ndarray], output_path: str):
        """Encode frames to MP4."""
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        duration = len(frames) / self.fps
        logger.info(f"💾 Size: {size_mb:.2f} MB | Duration: {duration:.1f}s")
    
    def _generate_thumbnail(self, game_data: GameData, output_path: str):
        """Generate video thumbnail."""
        thumb = Image.new('RGB', (1280, 720), COLORS['background'])
        draw = ImageDraw.Draw(thumb)
        
        # Green background for WIN
        draw.rectangle([0, 0, 1280, 720], fill=(30, 60, 30))
        
        # Title
        draw.text((640, 200), "♟️ CHESS WIN", fill=COLORS['text_primary'], 
                  font=self.font_large, anchor="mm")
        draw.text((640, 300), game_data.result, fill=COLORS['win_color'],
                  font=self.font_move, anchor="mm")
        draw.text((640, 420), game_data.opening, fill=COLORS['text_secondary'],
                  font=self.font_medium, anchor="mm")
        draw.text((640, 520), f"by {self.display_name}", fill=COLORS['text_primary'],
                  font=self.font_medium, anchor="mm")
        
        thumb.save(output_path, quality=95)
    
    def _save_metadata(self, game_data: GameData, output_path: str):
        """Save video metadata for YouTube upload."""
        metadata = {
            "title": f"♟️ Chess Win: {game_data.opening} | {game_data.result}",
            "description": f"Watch this chess victory by {self.display_name}!\n\n"
                          f"Opening: {game_data.opening}\n"
                          f"Result: {game_data.result}\n\n"
                          f"#chess #shorts #chessgame #checkmate",
            "tags": ["chess", "shorts", "chess game", "checkmate", "blitz chess", 
                    game_data.opening.split()[0] if game_data.opening else "chess"],
            "result": game_data.result,
            "opening": game_data.opening,
            "player_color": game_data.player_color,
            "player_won": game_data.player_won
        }
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================

EnhancedVideoGenerator = ChessBeastVideoGenerator

# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ChessBeast Video Generator")
    parser.add_argument('pgn_file', help='Path to PGN file')
    parser.add_argument('--output', '-o', help='Output filename')
    parser.add_argument('--player', '-p', default='ChessBeast_37', help='Chess.com username')
    parser.add_argument('--name', '-n', default='ChessBeast', help='Display name in video')
    parser.add_argument('--max-moves', '-m', type=int, help='Max moves')
    parser.add_argument('--depth', '-d', type=int, default=12, help='Stockfish depth')
    parser.add_argument('--include-losses', action='store_true', help='Include lost games too')
    
    args = parser.parse_args()
    
    generator = ChessBeastVideoGenerator(
        player_username=args.player,
        display_name=args.name,
        stockfish_depth=args.depth
    )
    
    result = generator.generate_video(
        pgn_path=args.pgn_file,
        output_filename=args.output,
        max_moves=args.max_moves,
        wins_only=not args.include_losses
    )
    
    if result:
        print(f"\n✅ Video: {result}")
        return 0
    print("\n❌ Failed (maybe not a win?)")
    return 1

if __name__ == "__main__":
    sys.exit(main())
