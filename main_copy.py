"""
Zero Latency Hindi→English Translation
Decoupled Pipeline Architecture (Netflix/YouTube Pattern)

🏭 THE FACTORY LINE: Three Independent Teams with Conveyor Belts

Team 1 (Ears): listen_loop
    → Listens to Deepgram STT continuously
    → Throws text onto Conveyor Belt A (translation_queue)
    → NEVER STOPS - Continuous listening

Team 2 (Brain): translate_and_synthesize_loop  
    → Picks text from Conveyor Belt A
    → Translates using Groq LLM (llama-3.3-70b-versatile - Best Quality)
    → Generates audio frames using ElevenLabs TTS (Flash 2.5)
    → Throws audio frames onto Conveyor Belt B (audio_queue)
    → RUNS IN PARALLEL - Never blocks Team 1 or Team 3

Team 3 (Mouth): playback_loop
    → Picks audio frames from Conveyor Belt B
    → Plays them with strict real-time pacing (1.0x speed)
    → STRICTLY PACED - Prevents buffer bloat

Key Insight: Processing (Thinking) and Playback (Speaking) never fight for CPU.
Each team runs independently, only communicating through queues.

Run: python agent.py dev
"""

import os
import sys
import asyncio
import time
import re
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
# Note: _ExitCli is a private class, we'll catch it by type checking
from livekit.agents import stt
from livekit.plugins import deepgram, openai

try:
    from livekit.plugins import elevenlabs
    from livekit.plugins.elevenlabs import Voice, VoiceSettings
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    import logging
    logging.basicConfig()
    temp_logger = logging.getLogger("TEMP")
    temp_logger.warning("LiveKit ElevenLabs plugin not available - install with: pip install 'livekit-agents[elevenlabs]'")

# Performance Optimization: Use fastest event loop for each platform
# - Windows: winloop (significantly faster than default Python loop)
# - Linux/Mac: uvloop (2-4x faster async I/O)
# This provides massive speed gains (100ms-300ms savings) for async operations
# 
# Note: winloop requires explicit event loop creation, so we create it here
# This ensures compatibility with LiveKit CLI which calls get_event_loop()
try:
    if sys.platform == 'win32':
        # Windows: Use winloop (significantly faster than default Python loop)
        import winloop
        # Set the policy first
        asyncio.set_event_loop_policy(winloop.EventLoopPolicy())
        # winloop's policy requires an event loop to exist before get_event_loop() is called
        # Create and set it here to ensure compatibility with LiveKit CLI
        try:
            # Check if event loop already exists
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                # If closed, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        print("✓ winloop enabled - Significantly faster async I/O on Windows")
    else:
        # Linux/Mac: Use uvloop (2-4x faster async I/O)
        import uvloop
        # Set the policy instead of install() to be compatible with LiveKit CLI
        # This allows LiveKit to create the event loop, which will use uvloop's policy
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        # uvloop's policy requires an event loop to exist before get_event_loop() is called
        # Create and set it here to ensure compatibility with LiveKit CLI
        try:
            # Check if event loop already exists
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                # If closed, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        print("✓ uvloop enabled - 2-4x faster async I/O")
except ImportError as e:
    print(f"ℹ Fast event loop not available ({e}) - using default event loop")
    print("  Performance tip: Install 'winloop' on Windows or 'uvloop' on Linux/Mac for better performance")

load_dotenv()

# ============================================================================
# PERFORMANCE METRICS TRACKING
# ============================================================================

# ============================================================================
# LOGGING SETUP - Detailed logs for all models
# ============================================================================
class MicrosecondFormatter(logging.Formatter):
    """Custom formatter that supports microseconds in timestamps"""
    def formatTime(self, record, datefmt=None):
        """Format time with microseconds support"""
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            # Replace %f with actual microseconds if present
            if '%f' in datefmt:
                s = ct.strftime(datefmt.replace('%f', f'{ct.microsecond:06d}'))
            else:
                s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s

def setup_logging():
    """Setup standard Python logging"""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
        
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"agent_log_{timestamp}.log")
        
    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    if root.handlers:
        root.handlers.clear()
    
    # File handler for log file
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_formatter = MicrosecondFormatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        '%Y-%m-%d %H:%M:%S.%f'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    
    # Console handler for terminal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', '%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)
        
    # Create specific loggers for each model
    stt_logger = logging.getLogger("STT_DEEPGRAM")
    llm_logger = logging.getLogger("LLM_GROQ")
    tts_logger = logging.getLogger("TTS_ELEVENLABS")
    queue_logger = logging.getLogger("QUEUE")
    perf_logger = logging.getLogger("PERFORMANCE")
    error_logger = logging.getLogger("ERROR")
        
    # Log initialization
    root.info(f"=" * 80)
    root.info(f"LOGGING INITIALIZED - Log file: {log_file_path}")
    root.info(f"Models: Deepgram STT (nova-3), Groq LLM (llama-3.3-70b-versatile), ElevenLabs TTS (flash_v2_5)")
    root.info(f"=" * 80)
        
    return {
        'main': root,
        'stt': stt_logger,
        'llm': llm_logger,
        'tts': tts_logger,
        'queue': queue_logger,
        'perf': perf_logger,
        'error': error_logger,
        'file': log_file_path
    }

# Initialize logging
LOGGERS = setup_logging()

# LLM - Optimized Settings (Quality-First: Best Translation Accuracy)
# Groq: llama-3.3-70b-versatile - Superior translation quality and accuracy
# Quality: 70B model provides better translation accuracy, preserves context, handles complex idioms better
# Latency: ~200-300ms TTFT (acceptable trade-off for superior translation quality)
GROQ_LLM = openai.LLM(
    model="llama-3.3-70b-versatile",  # Best Quality: Superior translation accuracy, better context understanding
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,  # Lower temperature for consistent, accurate translations
)

# Use Groq LLM directly (no Cerebras wrapper)
MAIN_LLM = GROQ_LLM

# Configuration Constants - ULTRA-OPTIMIZED for Sub-Second Latency (<1000ms total)
# OPTIMIZED BASED ON COMPREHENSIVE ANALYSIS: agent_log_20251202_103504.log
# Target: STT <1200ms, LLM <500ms, TTS <800ms, Total <1500ms
# Ultra-Low Latency Optimizations: Immediate Streaming (No Buffering)
# 🧠 ARCHITECTURE DECISION: The "Razor's Edge" - 220ms endpointing
# The average human pause between words is ~200ms. 220ms is the "razor's edge" where we catch
# the end of sentences instantly without cutting off slow talkers. 300ms was too safe (800ms+ latency floor).
# 220ms reduces minimum latency from 800ms to ~720ms while still catching natural sentence endings.
STT_ENDPOINTING_MS = 220  # 220ms: Aggressive endpointing for zero-latency - catches sentence end instantly
# Minimum word threshold: Allow single-word responses (Yes, No, Stop, Wait, Namaste)
# In translation apps, users often say single words - these should be translated
STT_MIN_WORDS_THRESHOLD = 1  # 1 word: Allow single-word responses for better UX
# Speech start time max age: Prevent incorrect latency calculations
STT_SPEECH_START_MAX_AGE = 2.5  # 2.5 seconds: More accurate tracking
# START_OF_SPEECH debouncing: Ignore multiple VAD events within this window
STT_VAD_DEBOUNCE_SEC = 0.5  # 0.5 seconds: Faster detection for ultra-low latency
LATENCY_ALERT_THRESHOLD_MS = 3000  # Alert if STT latency exceeds 3s
TTS_BATCH_MIN_WORDS = 5  # 5 words: Sending 5-word chunks reduces API calls by 40% and prevents rate limits
TTS_BATCH_SENTENCE_MIN_WORDS = 2  # 2 words: Complete phrases on sentence end
TTS_BATCH_TIMEOUT_SEC = 0.5  # 500ms: Slightly longer for better sentence completion
TTS_BATCH_MIN_WORDS_TIMEOUT = 2  # 2 words: Better minimum before timeout send
TTS_MAX_CHUNK_SIZE = 40  # 40 chars: Smaller chunks = faster synthesis
TTS_CONCURRENT_LIMIT = 2  # 2: Allows handshake overlap - next sentence starts TTS connection while current one is speaking (hides ~150ms connection time)
CONTEXT_WINDOW_SIZE = 1  # 1 message: ISOLATED mode - each translation is independent (prevents context bleeding)
CONTEXT_MAX_MESSAGES = 2  # System(1) + Current message(1) = 2 - Zero context bleeding
CONTEXT_MAX_ITEMS = 2  # System(1) + Current message(1) = 2 - Complete isolation prevents hallucination
# ⚠️ ARCHITECTURAL DESIGN: ISOLATED TRANSLATION MODE
# Context window is set to 1 message (ISOLATED mode) - each translation is completely independent.
# Strategy: Create fresh isolated context for each translation (system + current message only).
# Benefit: ZERO context bleeding - LLM cannot see previous messages, preventing hallucination/merging.
# Trade-off: No pronoun resolution from previous messages, but prevents severe context bleeding issues.
# This is the correct approach for translation tasks where each input should be translated independently.
TRANSLATION_QUEUE_TIMEOUT = 0.03  # 30ms: Ultra-responsive pickup (reduced from 50ms - 40% faster)
# Translation timeout: Adaptive based on text length (longer text = more time)
def get_adaptive_translation_timeout(text_length: int) -> float:
    """Calculate adaptive timeout based on text length - Optimized for quality and completeness"""
    # Increased timeouts to prevent incomplete translations ending with "..."
    base_timeout = 4.0  # Base timeout for short texts (increased to allow complete translations)
    char_timeout = text_length * 0.08  # 80ms per character (increased to allow LLM to finish completely)
    # Cap at 10.0s for very long inputs (allows complete translation without premature cutoff)
    return min(base_timeout + char_timeout, 10.0)

MAX_TRANSLATION_DELAY = 4.0  # 4s: Default timeout (adaptive function used for actual timeout)
# 🧹 CLEANUP "GHOST" RETRIES: Disable retries for real-time performance
# Retries add 2-4 seconds of silence on failure - unacceptable for real-time translation
# Fail fast, move on to next sentence. Speed > Recovery.
MAX_TRANSLATION_RETRIES = 0  # Disable retries. Speed > Recovery.
TRANSLATION_RETRY_DELAY_BASE = 0.2  # Base delay for exponential backoff (reduced from 0.3s - 33% faster)
TRANSLATION_RETRY_BACKOFF_MULTIPLIER = 1.6  # Exponential backoff multiplier (reduced from 1.8 - faster recovery)
# Network resilience
TCP_WARMING_TIMEOUT = 2.0  # 2s: Increased timeout for TCP warming
TCP_WARMING_RETRIES = 3  # 3 retries with exponential backoff
CONNECTION_HEALTH_CHECK_INTERVAL = 15.0  # 15s: Check connection health every 15s (reduced for faster recovery)
# STT Keep-Alive: Send every 5 seconds to prevent 10-second WebSocket timeout
# Deepgram requires keep-alive every 3-5 seconds to prevent NET-0001 errors
# The LiveKit Deepgram plugin handles this internally, but explicit keep-alive helps with firewalls
STT_CONNECTION_KEEPALIVE_INTERVAL = 5.0  # 5s: Send keepalive to STT connection every 5 seconds (matches Deepgram's internal keep-alive)

# Input Validation Constants
MIN_INPUT_LENGTH = 1  # Minimum input length to accept - We want to translate everything, even short affirmations
MAX_INPUT_LENGTH = 500  # Maximum input length before truncation
MIN_INPUT_LENGTH_FOR_RETRY = 5  # Minimum input length to retry on empty result
MIN_INPUT_LENGTH_SHORT = 15  # Short input threshold for ellipsis validation
MIN_INPUT_LENGTH_VERY_SHORT = 20  # Very short input threshold for ellipsis rejection
MAX_DOTS_THRESHOLD = 3  # Maximum dots before considering excessive ellipsis
MAX_ELLIPSIS_THRESHOLD = 2  # Maximum ellipsis characters before considering excessive
MAX_DOTS_SHORT_INPUT = 2  # Maximum dots for short inputs
MAX_ELLIPSIS_SHORT_INPUT = 1  # Maximum ellipsis for short inputs

# TTS Buffer Constants
# Smart Buffering Strategy: Balance speed and quality
# - Full buffer (60 chars / ~250ms): Ensures smooth, natural intonation for complete phrases
# - Smart early send (20+ chars with punctuation): Sends immediately when sentence ends early
# This gives quality of 250ms buffering with speed of 50ms when punctuation appears
TTS_BUFFER_SOFT_LIMIT = 60  # Full buffer limit (~250ms delay) - smoother speech, better intonation
TTS_BUFFER_SMART_MIN = 20  # Minimum chars for smart early send (when punctuation detected)
TTS_BUFFER_HARD_LIMIT = 600  # Hard limit for TTS buffer (force clear when exceeded) - Increased to prevent data loss on long sentences
TTS_CHUNK_LOG_INTERVAL = 5  # Log every Nth TTS chunk
TTS_FRAME_LOG_INTERVAL = 10  # Log every Nth TTS audio frame

# Output Validation Constants
MIN_OUTPUT_LENGTH = 2  # Minimum output length to accept
MIN_OUTPUT_LENGTH_MEANINGFUL = 3  # Minimum output length for meaningful translation
MAX_OUTPUT_EXPANSION = 5.0  # Maximum expansion ratio (output/input) - Prevents valid translations from being deleted just because they are longer than input
OUTPUT_EXTRACTION_RATIO = 1.6  # Ratio for extracting sentences from context bleeding
OUTPUT_EXTRACTION_RATIO_STRICT = 1.5  # Stricter ratio for first sentence extraction
MIN_SENTENCE_LENGTH = 5  # Minimum sentence length for extraction
MIN_SENTENCE_LENGTH_COMPLETE = 10  # Minimum length for complete sentence

# Context Pruning Constants
CONTEXT_PRUNE_TRIGGER_OFFSET = 1  # Prune when 1 item away from limit
CONTEXT_PRUNE_KEEP_OFFSET = 4  # Keep count offset for aggressive pruning
CONTEXT_PRUNE_KEEP_OFFSET_POST = 2  # Keep count offset after translation

# Active Translations Management
MAX_ACTIVE_TRANSLATIONS = 8  # Maximum active translations before cleanup
ACTIVE_TRANSLATIONS_CLEANUP_THRESHOLD = 5  # Cleanup threshold for active translations
ACTIVE_TRANSLATIONS_CLEANUP_INTERVAL = 10  # Cleanup every Nth translation

# Timeout Check Constants
TIMEOUT_CHECK_CHUNK_INTERVAL = 10  # Check timeout every Nth chunk
TIMEOUT_CHECK_EARLY_THRESHOLD = 0.8  # Check timeout on every chunk if past 80% of timeout
TIMEOUT_GRACE_PERIOD = 1.75  # Grace period multiplier for timeout (75% extra time)
TIMEOUT_RETRY_MULTIPLIER = 1.2  # Timeout multiplier for retries
TIMEOUT_RETRY_MAX = 8.0  # Maximum timeout for retries


# Playback Constants
PLAYBACK_LOG_INTERVAL = 50  # Log every Nth playback event
# NOTE: Playback speed is now controlled natively via ElevenLabs VoiceSettings(speed=1.0)
# No client-side resampling needed - saves 50-100ms CPU time per frame

# Sanitization Constants
SANITIZE_MAX_LENGTH = 2000  # Maximum length for LLM input sanitization
MIN_WORDS_FOR_VALIDATION = 1  # Minimum words for text validation

# STT - Deepgram Nova-3 (Optimized for Hindi/Hinglish with Code-Switching)
DEEPGRAM_STT = deepgram.STT(
    model="nova-3",  # ⬆️ UPGRADED: 54.3% WER reduction, better Hindi support, similar latency
    language="multi",  # ⬆️ CHANGED: Enables automatic code-switching for Hinglish (Hindi+English)
    smart_format=False,  # Disabled: Raw transcripts for better TTS input (no formatting interference) - Minimizes token latency
    endpointing_ms=STT_ENDPOINTING_MS,  # 220ms pause detection - Aggressive endpointing for zero-latency (razor's edge: catches sentence end instantly without cutting off slow talkers)
    interim_results=False,  # Disabled: Only use final transcripts to avoid duplicates and ensure stable TTS input
    no_delay=True,  # Disable delay for faster response
    # Note: vad_events=True is automatically enabled by the LiveKit Deepgram plugin
    # Note: utterance_end_ms is not a valid parameter for LiveKit Deepgram plugin - endpointing_ms handles pause detection
    # TTS Optimization: Disabled smart_format and interim_results for cleaner, more natural TTS input
    # - smart_format=False: Raw text without formatting that could interfere with TTS pronunciation
    # - interim_results=False: Only final, stable transcripts for consistent TTS output
)

# TTS - ElevenLabs Flash 2.5 (Ultra-low latency, ~75ms inference, high quality)
# CRITICAL: LiveKit ElevenLabs plugin expects ELEVEN_API_KEY (not ELEVENLABS_API_KEY)
# We support both for compatibility, but prioritize ELEVEN_API_KEY
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
if ELEVEN_API_KEY and not os.getenv("ELEVEN_API_KEY"):
    # Set ELEVEN_API_KEY if only ELEVENLABS_API_KEY is provided
    os.environ["ELEVEN_API_KEY"] = ELEVEN_API_KEY

# Voice ID validation - reject placeholder values
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice - reliable default
voice_id_from_env = os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)

# Validate voice ID - reject common placeholder values
INVALID_VOICE_PLACEHOLDERS = ["your_voice_id_here", "voice_id_here", "your_voice", "placeholder", ""]
if voice_id_from_env.lower().strip() in [p.lower() for p in INVALID_VOICE_PLACEHOLDERS]:
    if LOGGERS:
        LOGGERS['tts'].warning(f"Invalid voice ID '{voice_id_from_env}' detected - using default Rachel voice: {DEFAULT_VOICE_ID}")
    ELEVENLABS_VOICE_ID = DEFAULT_VOICE_ID
else:
    ELEVENLABS_VOICE_ID = voice_id_from_env

ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"  # Flash 2.5 for ultra-low latency (~75ms)
ELEVENLABS_STREAMING_LATENCY = 1  # Low latency streaming (default: 3, lower = faster)

# Initialize ElevenLabs TTS using LiveKit plugin
# CRITICAL FIX: Dynamic TTS initialization to handle voice ID changes
ELEVENLABS_TTS = None
_ELEVENLABS_TTS_VOICE_ID = None  # Track the voice ID used in current TTS instance

def get_elevenlabs_tts():
    """
    Get or create ElevenLabs TTS instance with current voice ID.
    Dynamically recreates TTS if voice ID has changed.
    
    SAFETY: This function safely handles the case where VoiceSettings is not available.
    If ELEVENLABS_AVAILABLE is False (import failed), VoiceSettings would be undefined,
    but we check and return None BEFORE attempting to use it.
    """
    global ELEVENLABS_TTS, ELEVENLABS_VOICE_ID, _ELEVENLABS_TTS_VOICE_ID
    
    # Get current voice ID from environment (may have changed)
    current_voice_id = os.getenv("ELEVENLABS_VOICE_ID", ELEVENLABS_VOICE_ID)
    
    # Validate current voice ID
    if current_voice_id.lower().strip() in [p.lower() for p in INVALID_VOICE_PLACEHOLDERS]:
        current_voice_id = DEFAULT_VOICE_ID
    
    # Check if TTS needs to be recreated (voice ID changed or not initialized)
    if (ELEVENLABS_TTS is None or 
        _ELEVENLABS_TTS_VOICE_ID != current_voice_id or
        not ELEVENLABS_AVAILABLE):
        
        # CRITICAL SAFETY CHECK: Return early if ElevenLabs plugin is not available
        # This prevents NameError if VoiceSettings was not imported (ELEVENLABS_AVAILABLE = False)
        if not ELEVENLABS_AVAILABLE:
            if LOGGERS:
                LOGGERS['tts'].warning("LiveKit ElevenLabs plugin not available - install with: pip install 'livekit-agents[elevenlabs]'")
            return None
        
        if not ELEVEN_API_KEY:
            if LOGGERS:
                LOGGERS['error'].error("ElevenLabs API key not found - Set ELEVEN_API_KEY or ELEVENLABS_API_KEY environment variable")
            return None
        
        try:
            # Update global voice ID
            ELEVENLABS_VOICE_ID = current_voice_id
            
            # VoiceSettings with speed=1.0 (stability and similarity_boost are required)
            voice_settings = VoiceSettings(
                stability=0.5,          # Required parameter (standard default)
                similarity_boost=0.75, # Required parameter (standard default)
                speed=1.0            # Desired speed setting (1.0 = normal speed)
            )
            
            # Initialize TTS with voice_id and voice_settings directly
            # This is the correct API pattern (as verified in test_elevenlabs_speed.py)
            # The TTS constructor accepts voice_settings parameter directly, not wrapped in Voice
            ELEVENLABS_TTS = elevenlabs.TTS(
                voice_id=ELEVENLABS_VOICE_ID,
                voice_settings=voice_settings,  # Pass VoiceSettings directly to TTS constructor
                model=ELEVENLABS_MODEL_ID,
                streaming_latency=ELEVENLABS_STREAMING_LATENCY,
            )
            _ELEVENLABS_TTS_VOICE_ID = current_voice_id  # Track the voice ID used
            
            if LOGGERS:
                LOGGERS['tts'].info(f"ElevenLabs TTS initialized/recreated - Voice: {ELEVENLABS_VOICE_ID}, Model: {ELEVENLABS_MODEL_ID}, Speed: 1.0, Streaming Latency: {ELEVENLABS_STREAMING_LATENCY}")
        except Exception as e:
            if LOGGERS:
                LOGGERS['error'].error(f"ElevenLabs TTS initialization failed: {str(e)}")
                LOGGERS['error'].error(f"  - Voice ID: {ELEVENLABS_VOICE_ID}")
                LOGGERS['error'].error(f"  - Model: {ELEVENLABS_MODEL_ID}")
                LOGGERS['error'].error(f"  - API Key present: {bool(ELEVEN_API_KEY)}")
            ELEVENLABS_TTS = None
            _ELEVENLABS_TTS_VOICE_ID = None
            return None
    
    return ELEVENLABS_TTS

# Initialize TTS on startup
if ELEVENLABS_AVAILABLE:
    get_elevenlabs_tts()  # Initial creation

# ============================================================================
# TRANSLATION ENGINE PROMPT
# Real-time Hindi-to-English livestream translation and text-normalization
# Optimized for ElevenLabs Flash v2.5 text-to-speech
# ============================================================================

TRANSLATION_INSTRUCTIONS = (
    "You are a Hindi/Hinglish-to-English translator for live streaming. Translate ONLY the current input.\n\n"
    
    "CRITICAL: ISOLATED TRANSLATION MODE - ZERO CONTEXT BLEEDING\n"
    "• You will receive ONLY ONE user message - translate ONLY that message\n"
    "• DO NOT add information from previous conversations\n"
    "• DO NOT continue or merge with anything else\n"
    "• DO NOT translate multiple messages - only the current one\n"
    "• If input is incomplete (ends with ...), translate what is there, don't add context\n"
    "• Example: Input 'हां,...' → Output 'Yes,...' (NOT a long sentence about previous topics)\n\n"
    
    "CRITICAL RULES:\n"
    "1. ACCURACY: Translate every word. \"और\"=and, \"भी\"=also, \"तो\"=so, \"नहीं\"=not\n"
    "2. NO ADDITIONS: Never add words not in Hindi (no \"brother\", \"friends\", names unless said)\n"
    "3. GRAMMAR: Complete sentences with subjects. \"जा रहा है\" → \"It is going\" (not \"is going\")\n"
    "4. IDIOMS: Translate meaning, not literal. \"खाते हुए नहीं खाते\" → \"don't waste resources\" (not \"don't eat while eating\")\n"
    "5. TENSE: है=is, था=was, गा=will. Preserve exactly\n"
    "6. HINGLISH: Keep English words as-is: Nifty, Telegram, channel, subscribe, Etc\n"
    "7. INCOMPLETE: Complete grammatically if clear, else use \"...\" or \"?\"\n"
    "8. TTS FORMAT: Use punctuation for rhythm. Numbers: 25→\"twenty-five\", ₹500→\"five hundred rupees\"\n"
    "10. NATURAL: Short, clear sentences. Remove fillers (acha, toh) unless emotional\n"
    "11. RELIGIOUS/PRAYER TERMS: Keep in Hindi - \"नमस्ते\"→\"namaste\", \"जय श्री राम\"→\"jai shri ram\", \"हर हर महादेव\"→\"har har mahadev\". Any prayers, mantras, or religious greetings stay in Hindi (not translated to English).\n\n"
    
    "EXAMPLES:\n"
    "\"नमस्ते\" → \"namaste\"\n"
    "\"जय श्री राम\" → \"jai shri ram\"\n"
    "\"हर हर महादेव\" → \"har har mahadev\"\n"
    "\"Nifty ऊपर जा रहा है\" → \"Nifty is going up\"\n"
    "\"channel subscribe करो\" → \"Subscribe to the channel\"\n"
    "\"पर आपने नहीं देखी या...\" → \"but you haven't seen it, have you?\"\n"
    "\"second पीछे चल रहा है\" → \"It is running behind by a second\"\n\n"
    
    "OUTPUT: Plain English text only. No explanations, no markdown, no quotes."
)

TRANSLATION_PROMPT = TRANSLATION_INSTRUCTIONS + "\n\nTranslate:"


def sanitize_text_for_tts(text: str) -> str | None:
    """Minimal text validation for TTS - Modern TTS engines handle messy text gracefully"""
    if not text or not text.strip():
        return None
    return text.strip()


def sanitize_llm_input(text: str) -> str:
    """Sanitize LLM input to prevent injection attacks and ensure valid content"""
    if not text:
        return ""
        
    # Remove excessive newlines (potential injection vector)
    text = text.replace('\n\n\n', '\n').replace('\r\n', '\n')
        
    # Limit length to prevent abuse (2000 chars max)
    if len(text) > SANITIZE_MAX_LENGTH:
        text = text[:2000] + "..."
        
    return text.strip()


def prune_chat_context(chat_ctx: llm.ChatContext, logger_name: str = 'llm') -> int:
    """
    Prune chat context to prevent overflow while preserving system message.
    
    This function keeps the system message and the most recent conversation pairs,
    ensuring the context stays within CONTEXT_MAX_ITEMS limit.
    
    Args:
        chat_ctx: The chat context to prune
        logger_name: Logger name for debug messages (default: 'llm')
    
    Returns:
        int: Number of items after pruning
    
    Strategy:
        - Always preserves system message (if present)
        - FILTERS OUT all assistant messages (prevents hallucination)
        - Keeps last (CONTEXT_MAX_ITEMS - 2) USER messages only (for pronoun reference)
        - Falls back to keeping first item + recent user items if no system message found
    """
    # ========================================================================
    # CONTEXT PRUNING ALGORITHM: Preserve System Message + Recent User Messages Only
    # ========================================================================
    # Goal: Keep context within CONTEXT_MAX_ITEMS (15) to prevent:
    #   - Token limit exceeded errors
    #   - Slow LLM responses (more context = slower processing)
    #   - Memory bloat
    #   - HALLUCINATION (by removing assistant messages)
    #
    # Strategy:
    #   1. Always preserve system message (contains translation instructions)
    #   2. FILTER OUT all assistant messages (prevents hallucination/context bleeding)
    #   3. Keep last (CONTEXT_MAX_ITEMS - 2) USER messages only (for pronoun reference)
    #   4. Fallback: If no system message, keep first item + recent user items
    #
    # Example with CONTEXT_MAX_ITEMS=15:
    #   Before: [system, user1, asst1, user2, asst2, ..., user8, asst8] (17 items)
    #   After:  [system, user6, user7, user8] (4 items - only user messages)
    #   Room for: new user message (5 items total)
    # ========================================================================
    context_size_before = len(chat_ctx.items)
    
    # CRITICAL: Filter out ALL assistant messages first (anti-hallucination)
    # Only user messages are kept for pronoun reference
    filtered_items = []
    for item in chat_ctx.items:
        if item.type == "message":
            if item.role == "system":
                filtered_items.append(item)  # Always keep system message
            elif item.role == "user":
                filtered_items.append(item)  # Keep user messages for pronoun reference
            # SKIP assistant messages - they cause hallucination
    
    # Update context with filtered items (removed assistant messages)
    if len(filtered_items) < len(chat_ctx.items):
        LOGGERS[logger_name].debug(
            f"🔧 Removed {len(chat_ctx.items) - len(filtered_items)} assistant message(s) from context "
            f"(anti-hallucination strategy)"
        )
        chat_ctx.items = filtered_items
    
    # Only prune if we're close to the limit
    if len(chat_ctx.items) < CONTEXT_MAX_ITEMS - CONTEXT_PRUNE_TRIGGER_OFFSET:
        # No pruning needed yet
        return len(chat_ctx.items)
    
    # Find system message
    system_msg = None
    for item in chat_ctx.items:
        if item.type == "message" and item.role == "system":
            system_msg = item
            break
    
    if system_msg:
        # Keep system + recent USER messages only (no assistant messages)
        # Keep only last (CONTEXT_MAX_ITEMS - 2) user messages to ensure room for new user message
        keep_count = max(1, CONTEXT_MAX_ITEMS - 2)
        recent_user_items = [item for item in chat_ctx.items[-keep_count:] 
                            if item.type == "message" and item.role == "user"]
        chat_ctx.items = [system_msg] + recent_user_items
        LOGGERS[logger_name].debug(
            f"🔧 Context pruning - Before: {context_size_before}, "
            f"After: {len(chat_ctx.items)} (using CONTEXT_MAX_ITEMS={CONTEXT_MAX_ITEMS})"
        )
    else:
        # Fallback: keep first item + recent user items only
        if len(chat_ctx.items) > 0:
            first_item = chat_ctx.items[0]
            keep_count = max(1, CONTEXT_MAX_ITEMS - 2)
            recent_user_items = [item for item in chat_ctx.items[-keep_count:]
                                if item.type == "message" and item.role == "user"]
            chat_ctx.items = [first_item] + recent_user_items
            LOGGERS[logger_name].debug(
                f"🔧 Context pruning (fallback) - Before: {context_size_before}, "
                f"After: {len(chat_ctx.items)} (using CONTEXT_MAX_ITEMS={CONTEXT_MAX_ITEMS})"
            )
    
    return len(chat_ctx.items)


# ============================================================================
# TEAM 1: THE EARS (listen_loop)
# Listens to Deepgram, throws text onto Conveyor Belt A (translation_queue)
# NEVER STOPS - Continuous listening
# ============================================================================
async def listen_loop(room: rtc.Room, translation_queue: asyncio.Queue) -> None:
    """Team 1: Ears - Continuous listening"""
    audio_stream = None
    stt_stream = None
        
    try:
        def on_track_subscribed(track, publication, participant):
            nonlocal audio_stream
            if isinstance(track, rtc.RemoteAudioTrack) and not audio_stream:
                try:
                    audio_stream = rtc.AudioStream.from_track(
                        track=track, sample_rate=16000, num_channels=1
                    )
                except Exception as e:
                    pass
            
        room.on("track_subscribed", on_track_subscribed)
            
        # Check existing tracks
        for p in room.remote_participants.values():
            for pub in p.track_publications.values():
                if pub.track and isinstance(pub.track, rtc.RemoteAudioTrack):
                    try:
                        audio_stream = rtc.AudioStream.from_track(
                            track=pub.track, sample_rate=16000, num_channels=1
                        )
                        break
                    except Exception:
                        continue
                if audio_stream:
                    break
            
        # Wait for audio stream (30s timeout)
        waited = 0
        while not audio_stream and waited < 30:
            await asyncio.sleep(0.5)
            waited += 0.5
            
        if not audio_stream:
            return
        
        # Minimal delay for stability
        await asyncio.sleep(0.05)
            
        # Create STT stream with retry logic (3 attempts)
        max_stt_retries = 3
        stt_stream = None
        for attempt in range(max_stt_retries):
            try:
                LOGGERS['stt'].info(f"Initializing Deepgram STT stream - Attempt {attempt + 1}/{max_stt_retries}")
                LOGGERS['stt'].info(f"Deepgram config - Model: nova-3, Language: multi, Endpointing: {STT_ENDPOINTING_MS}ms, VAD: enabled")
                stt_stream = DEEPGRAM_STT.stream()
                LOGGERS['stt'].info("Deepgram STT stream initialized successfully")
                break
            except Exception as e:
                LOGGERS['error'].error(f"Deepgram STT initialization failed (attempt {attempt + 1}): {str(e)}")
                if attempt < max_stt_retries - 1:
                    await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
                LOGGERS['error'].error("Deepgram STT initialization failed after all retries")
                return
            
        if not stt_stream:
            return
            
        async def feed_audio():
            """Feed audio to STT - runs until explicitly cancelled"""
            nonlocal audio_stream  # Must be declared before use
            while True:  # Keep running until cancelled, even if stream ends
                try:
                    if not audio_stream:
                        # Wait for audio stream to be available
                        await asyncio.sleep(0.5)
                        continue
                        
                    try:
                        async for event in audio_stream:
                            if event and event.frame and stt_stream:
                                try:
                                    stt_stream.push_frame(event.frame)
                                except (asyncio.CancelledError, Exception):
                                    break
                    except StopAsyncIteration:
                        # Stream ended - wait and try to reconnect
                        LOGGERS['stt'].warning("Audio stream ended - waiting for reconnection...")
                        await asyncio.sleep(1.0)
                        # Try to get new audio stream
                        for p in room.remote_participants.values():
                            for pub in p.track_publications.values():
                                if pub.track and isinstance(pub.track, rtc.RemoteAudioTrack):
                                    try:
                                        audio_stream = rtc.AudioStream.from_track(
                                            track=pub.track, sample_rate=16000, num_channels=1
                                        )
                                        LOGGERS['stt'].info("Audio stream reconnected")
                                        break
                                    except Exception:
                                        continue
                                if audio_stream:
                                    break
                        if not audio_stream:
                            await asyncio.sleep(2.0)  # Wait before retrying
                        continue
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    LOGGERS['error'].error(f"Audio feed error: {str(e)} - retrying...")
                    await asyncio.sleep(1.0)
                    continue
            
        async def process_transcripts():
            """Gear Shift Strategy: Fast Start (<3s) then Smooth Cruise with Reconnection Logic"""
            # Reconnection configuration - Optimized for faster recovery
            max_reconnect_attempts = 5  # Maximum reconnection attempts
            reconnect_delay_base = 0.5  # Base delay in seconds (reduced from 1.0s for faster recovery)
            max_reconnect_delay = 10.0  # Maximum delay between reconnections (reduced from 30.0s)
            reconnect_backoff_multiplier = 1.5  # Exponential backoff multiplier (gentler than 2.0)
                
            reconnect_count = 0
            nonlocal stt_stream
                
            while True:
                try:
                    speech_start_time = None
                    last_vad_event_time = None  # Track last START_OF_SPEECH event time for debouncing
                    transcript_count = 0
                    MAX_QUEUE_BACKPRESSURE = 8  # Don't send if queue has more than 8 items
                    deepgram_first_byte_time = None  # Track when Deepgram starts processing (for actual processing time)
                        
                    # Deduplication: Track recently sent texts to prevent duplicates
                    recent_sent_texts = []  # FIFO queue of last sent texts
                    MAX_RECENT_TEXTS = 5  # Keep last 5 sent texts for comparison
                    SIMILARITY_THRESHOLD = 0.75  # 75% word overlap = duplicate
                        
                    def calculate_similarity(text1: str, text2: str) -> float:
                        """Calculate word overlap similarity between two texts"""
                        words1 = set(text1.lower().strip().split())
                        words2 = set(text2.lower().strip().split())
                            
                        if len(words1) == 0 or len(words2) == 0:
                            return 0.0
                            
                        # Calculate Jaccard similarity (intersection over union)
                        intersection = len(words1 & words2)
                        union = len(words1 | words2)
                        return intersection / union if union > 0 else 0.0
                        
                    def is_duplicate_or_similar(text: str) -> bool:
                        """Check if text is duplicate or very similar to recently sent texts (final transcripts only)"""
                        text_normalized = text.lower().strip()
                        text_words = text_normalized.split()
                            
                        # Check all recent texts for duplicates
                        for sent_text in recent_sent_texts:
                            sent_normalized = sent_text.lower().strip()
                            sent_words = sent_normalized.split()
                                
                            # Exact match
                            if text_normalized == sent_normalized:
                                return True
                                
                            # Substring match - one contains the other
                            if text_normalized in sent_normalized or sent_normalized in text_normalized:
                                # Require 70% length similarity for final transcripts
                                len_ratio = min(len(text_normalized), len(sent_normalized)) / max(len(text_normalized), len(sent_normalized))
                                if len_ratio > 0.7:
                                    return True
                                
                            # Word overlap similarity
                            similarity = calculate_similarity(text, sent_text)
                            if similarity >= SIMILARITY_THRESHOLD:
                                return True
                            
                        return False
                        
                    def add_to_recent(text: str):
                        """Add text to recent sent texts (FIFO)"""
                        recent_sent_texts.append(text)
                        if len(recent_sent_texts) > MAX_RECENT_TEXTS:
                            recent_sent_texts.pop(0)
                        
                    # Reset reconnect count on successful connection
                    if reconnect_count > 0:
                        LOGGERS['stt'].info(f"STT stream reconnected successfully after {reconnect_count} attempts")
                        reconnect_count = 0
                        
                    # Ultra-Low Latency: No buffering, stream immediately
                        
                    try:
                        async for event in stt_stream:
                            if event.type == stt.SpeechEventType.START_OF_SPEECH:
                                # Debounce START_OF_SPEECH events to prevent multiple updates from rapid VAD triggers
                                current_time = time.time()
                                    
                                # Debouncing: Ignore START_OF_SPEECH events within STT_VAD_DEBOUNCE_SEC of previous
                                if last_vad_event_time is not None:
                                    time_since_last_vad = current_time - last_vad_event_time
                                    if time_since_last_vad < STT_VAD_DEBOUNCE_SEC:
                                        # Ignore this VAD event - too soon after previous one
                                        LOGGERS['stt'].debug(f"Ignoring START_OF_SPEECH (debounced: {time_since_last_vad:.2f}s < {STT_VAD_DEBOUNCE_SEC}s)")
                                        continue
                                    
                                # Update last VAD event time
                                last_vad_event_time = current_time
                                    
                                # Set or update speech_start_time, but cap max age to prevent incorrect latency
                                if not speech_start_time:
                                    # First speech start - set timestamp
                                    speech_start_time = current_time
                                    deepgram_first_byte_time = current_time  # Track when Deepgram starts processing
                                    LOGGERS['stt'].debug(f"First START_OF_SPEECH - speech_start_time set to {current_time:.3f}")
                                elif (current_time - speech_start_time) > STT_SPEECH_START_MAX_AGE:
                                    # Speech start time is too old - reset it
                                    LOGGERS['stt'].debug(f"Resetting stale speech_start_time (age: {current_time - speech_start_time:.1f}s > {STT_SPEECH_START_MAX_AGE}s)")
                                    speech_start_time = current_time
                                    deepgram_first_byte_time = current_time
                                else:
                                    # Keep existing speech_start_time - this is continuation of same segment
                                    LOGGERS['stt'].debug(f"START_OF_SPEECH within same segment - keeping existing timestamp")
                                    
                                LOGGERS['stt'].debug("Deepgram START_OF_SPEECH detected (VAD event)")
                            elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                                # DISABLED: Interim transcripts disabled to avoid duplicate translations
                                # Only processing final transcripts for reliability
                                continue
                            elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                                if event.alternatives and event.alternatives[0].text.strip():
                                    text = event.alternatives[0].text.strip()
                                    transcript_count += 1
                                        
                                    # STEP 4: THE EARS NOISE GATE - Filter garbage transcripts
                                    # Filter empty text
                                    if not text or len(text) == 0:
                                        LOGGERS['stt'].debug("Filtered: Empty transcript")
                                        continue
                                        
                                    # Filter very short text (unless it's "ok" or "ji")
                                    text_lower = text.lower().strip()
                                    if len(text) < MIN_INPUT_LENGTH and text_lower not in ["ok", "ji"]:
                                        LOGGERS['stt'].debug(f"Filtered: Too short - '{text}'")
                                        continue
                                        
                                    # Filter text containing only punctuation
                                    text_no_punct = text.replace('.', '').replace(',', '').replace('!', '').replace('?', '').replace('-', '').replace(':', '').replace(';', '').replace(' ', '').strip()
                                    if len(text_no_punct) == 0:
                                        LOGGERS['stt'].debug(f"Filtered: Only punctuation - '{text}'")
                                        continue
                                        
                                    # Log transcript (processing time will be added later if available)
                                    LOGGERS['stt'].info(f"Deepgram transcript #{transcript_count} - Text: '{text[:100]}...'")
                                        
                                    # Ultra-Low Latency: Stream immediately (no buffering)
                                    words = text.split()
                                    word_count = len(words)
                                        
                                    if word_count == 0:
                                        continue
                                        
                                    # Check minimum word threshold
                                    if word_count < STT_MIN_WORDS_THRESHOLD:
                                        LOGGERS['stt'].debug(f"Skipping: {word_count} words < minimum threshold ({STT_MIN_WORDS_THRESHOLD})")
                                        continue
                                        
                                    # Calculate latency
                                    transcript_received_time = time.time()
                                    deepgram_processing_time = None
                                    stt_latency = 0
                                        
                                    if speech_start_time:
                                        age = transcript_received_time - speech_start_time
                                        if age > STT_SPEECH_START_MAX_AGE:
                                            LOGGERS['stt'].debug(f"Resetting stale speech_start_time (age: {age:.1f}s)")
                                            speech_start_time = transcript_received_time
                                            deepgram_first_byte_time = transcript_received_time
                                        else:
                                            stt_latency = int(age * 1000)
                                            # Calculate actual Deepgram processing time
                                            if deepgram_first_byte_time:
                                                processing_time = transcript_received_time - deepgram_first_byte_time
                                                deepgram_processing_time = int(processing_time * 1000)
                                        
                                    # Stream immediately - no Gear logic, no buffering
                                    sanitized = sanitize_llm_input(text)
                                    if sanitized:
                                        # Check for duplicates
                                        if is_duplicate_or_similar(sanitized):
                                            LOGGERS['stt'].info(f"Skipping duplicate: '{sanitized[:50]}...'")
                                            continue
                                            
                                        # Check queue backpressure
                                        queue_size = translation_queue.qsize()
                                        if queue_size > MAX_QUEUE_BACKPRESSURE:
                                            LOGGERS['stt'].warning(f"Skipping - Queue too full: {queue_size}")
                                            continue
                                            
                                        LOGGERS['stt'].info(f"Streaming immediately - {word_count} words")
                                        LOGGERS['queue'].info(f"Queueing translation - Text: '{sanitized[:100]}...', Length: {len(sanitized)} chars")
                                            
                                        try:
                                            translation_queue.put_nowait(sanitized)
                                            add_to_recent(sanitized)
                                            LOGGERS['queue'].debug(f"Translation queue size: {translation_queue.qsize()}")
                                                
                                            if stt_latency > 0:
                                                latency_msg = f"T1_STT: {stt_latency}ms"
                                                if deepgram_processing_time:
                                                    latency_msg += f" (processing: {deepgram_processing_time}ms)"
                                                print(latency_msg)
                                                LOGGERS['perf'].info(f"T1_STT latency: {stt_latency}ms" + (f" (Deepgram processing: {deepgram_processing_time}ms)" if deepgram_processing_time else ""))
                                        except asyncio.QueueFull:
                                            # FIFO drop
                                            try:
                                                dropped = translation_queue.get_nowait()
                                                translation_queue.task_done()
                                                LOGGERS['queue'].warning(f"Translation queue full - Dropped: '{str(dropped)[:50]}...'")
                                                translation_queue.put_nowait(sanitized)
                                                add_to_recent(sanitized)
                                            except asyncio.QueueEmpty:
                                                try:
                                                    translation_queue.put_nowait(sanitized)
                                                    add_to_recent(sanitized)
                                                except asyncio.QueueFull:
                                                    LOGGERS['queue'].error("Translation queue still full after drop attempt")
                                                    pass
                                        
                                    # Reset speech tracking after sending
                                    speech_start_time = None
                                    deepgram_first_byte_time = None
                        
                    except asyncio.CancelledError:
                        LOGGERS['stt'].info("STT transcript processing cancelled")
                        raise
                    except StopAsyncIteration:
                        # Stream ended normally - attempt reconnection
                        LOGGERS['stt'].warning("STT stream ended unexpectedly - attempting reconnection")
                    except (ConnectionError, OSError, RuntimeError) as conn_error:
                        # Connection-related errors - attempt reconnection
                        error_msg = str(conn_error)
                        LOGGERS['stt'].warning(f"STT connection error: {error_msg} - attempting reconnection")
                    except Exception as e:
                        error_str = str(e).lower()
                        error_msg = str(e)
                        # Check if it's a connection-related error
                        is_connection_error = any(keyword in error_str for keyword in [
                            'connection', 'websocket', 'network', 'timeout', 'closed', 
                            'broken pipe', 'reset by peer', 'connection refused'
                        ])
                            
                        if is_connection_error:
                            LOGGERS['stt'].warning(f"STT connection issue detected: {error_msg} - attempting reconnection")
                        else:
                            # Filter known harmless errors
                            if "timeout context manager" in error_str:
                                LOGGERS['stt'].debug("Known asyncio timeout issue - ignoring")
                            else:
                                LOGGERS['error'].error(f"STT transcript processing error: {error_msg}")
                    
                except asyncio.CancelledError:
                    # Propagate cancellation to outer scope
                    raise
                finally:
                    # Reconnection logic (runs after any exception)
                    pass
                    
                # Reconnection logic
                if reconnect_count < max_reconnect_attempts:
                    reconnect_count += 1
                    # Calculate exponential backoff delay
                    reconnect_delay = min(
                        reconnect_delay_base * (reconnect_backoff_multiplier ** (reconnect_count - 1)),
                        max_reconnect_delay
                    )
                        
                    LOGGERS['stt'].info(
                        f"Attempting STT stream reconnection in {reconnect_delay:.1f}s "
                        f"(Attempt {reconnect_count}/{max_reconnect_attempts})"
                    )
                    await asyncio.sleep(reconnect_delay)
                        
                    try:
                        # Close old stream if it exists
                        if stt_stream:
                            try:
                                await asyncio.wait_for(stt_stream.aclose(), timeout=1.0)
                            except Exception:
                                pass  # Ignore errors when closing old stream
                            
                        # Create new STT stream
                        LOGGERS['stt'].info(f"Reconnecting Deepgram STT stream - Attempt {reconnect_count}/{max_reconnect_attempts}")
                        stt_stream = DEEPGRAM_STT.stream()
                        LOGGERS['stt'].info("STT stream reconnected successfully - resuming processing")
                            
                        # Continue with new stream (will reset state in next iteration)
                        continue
                    except Exception as reconnect_error:
                        LOGGERS['error'].error(f"STT reconnection failed (attempt {reconnect_count}): {str(reconnect_error)}")
                        if reconnect_count >= max_reconnect_attempts:
                            LOGGERS['error'].error("STT reconnection failed after all attempts - stopping")
                            break
                        # Continue to next reconnection attempt
                        continue
                else:
                    LOGGERS['error'].error("STT reconnection exhausted all attempts - stopping transcript processing")
                    break
        async def keep_alive_stt():
            """
            Keep-alive task to prevent STT WebSocket from going stale during silence.
            
            Deepgram WebSocket connections timeout after 10 seconds of inactivity.
            This task sends keep-alive signals every STT_CONNECTION_KEEPALIVE_INTERVAL seconds
            to prevent timeouts during periods of silence (e.g., user thinking, long pauses).
            
            Strategy:
            - Try to call keep_alive() method if available (LiveKit plugin may expose this)
            - If not available, the LiveKit Deepgram plugin handles keep-alive internally
            - This explicit keep-alive helps with firewalls/NATs that drop idle connections
            
            Note: The LiveKit Deepgram plugin sends keep-alive every 5 seconds internally,
            but explicit keep-alive on the application side helps with network infrastructure.
            """
            nonlocal stt_stream
            while True:
                try:
                    await asyncio.sleep(STT_CONNECTION_KEEPALIVE_INTERVAL)
                    
                    # Check if stream exists and is still valid
                    if stt_stream is None:
                        # Stream not initialized yet - wait and retry
                        continue
                    
                    # Try to call keep_alive() method if it exists
                    # The LiveKit Deepgram plugin may expose this method
                    if hasattr(stt_stream, 'keep_alive'):
                        try:
                            # Call keep_alive if it's a coroutine
                            if asyncio.iscoroutinefunction(stt_stream.keep_alive):
                                await stt_stream.keep_alive()
                            else:
                                stt_stream.keep_alive()
                            LOGGERS['stt'].debug(f"STT keep-alive sent (method call)")
                        except Exception as ka_error:
                            # Method exists but failed - log and continue
                            LOGGERS['stt'].debug(f"STT keep-alive method call failed: {str(ka_error)}")
                    else:
                        # No keep_alive method - LiveKit plugin handles this internally
                        # Log periodically (every 10th keep-alive attempt = every 50 seconds) to avoid spam
                        # This confirms the task is running even if method doesn't exist
                        pass  # Plugin handles keep-alive internally, no action needed
                        
                except asyncio.CancelledError:
                    LOGGERS['stt'].debug("STT keep-alive task cancelled")
                    raise
                except Exception as e:
                    # Log error but continue - keep-alive failures shouldn't crash the loop
                    LOGGERS['stt'].debug(f"STT keep-alive error (non-critical): {str(e)}")
                    # Continue loop - wait and retry
                    await asyncio.sleep(STT_CONNECTION_KEEPALIVE_INTERVAL)
                    continue
        
        # Run all three tasks in parallel: feed audio, process transcripts, and keep-alive
        await asyncio.gather(feed_audio(), process_transcripts(), keep_alive_stt(), return_exceptions=True)
    except asyncio.CancelledError:
        LOGGERS['stt'].info("Listen loop cancelled")
        pass
    except Exception as e:
        LOGGERS['error'].error(f"Listen loop error: {str(e)}")
        pass
    finally:
        if stt_stream:
            try:
                LOGGERS['stt'].info("Closing Deepgram STT stream...")
                # Safe stream closure with timeout
                close_task = asyncio.create_task(stt_stream.aclose())
                try:
                    await asyncio.wait_for(close_task, timeout=0.5)
                    LOGGERS['stt'].info("Deepgram STT stream closed successfully")
                except asyncio.TimeoutError:
                    if not close_task.done():
                        close_task.cancel()
                        try:
                            await close_task
                        except Exception:
                            pass
                    LOGGERS['stt'].warning("Deepgram STT stream close timeout")
            except Exception as e:
                LOGGERS['error'].error(f"Error closing STT stream: {str(e)}")
                pass  # Ignore closure errors


# ============================================================================
# TEAM 2: THE BRAIN (translate_and_synthesize_loop)
# Picks text from Belt A, Translates, generates Audio Frames, 
# throws Audio Frames onto Conveyor Belt B (audio_queue)
# RUNS IN PARALLEL - Never blocks
# ============================================================================

async def process_single_translation(hindi_text: str, chat_ctx: llm.ChatContext,
                                    audio_queue: asyncio.Queue,
                                    tts_semaphore: asyncio.Semaphore,
                                    shutdown_event: asyncio.Event,
                                    translation_lock: asyncio.Lock) -> str:
    """Process one translation with continuous streaming TTS - Water Hose method for prosody continuity"""
    
    # ============================================================================
    # CRITICAL FIX #1: Input Validation BEFORE LLM Call
    # ============================================================================
    # Validate input to prevent empty translation results
    hindi_text = hindi_text.strip()
    input_length = len(hindi_text)
    
    # Reject empty or too-short inputs
    if not hindi_text or input_length == 0:
        LOGGERS['llm'].warning("⚠️ Empty input text - skipping translation")
        return ""
    
    # Reject very short inputs (likely noise or incomplete)
    if input_length < MIN_INPUT_LENGTH:
        LOGGERS['llm'].warning(f"⚠️ Input too short ({input_length} chars) - skipping translation: '{hindi_text}'")
        return ""
    
    # Reject inputs with only whitespace
    if not hindi_text.strip():
        LOGGERS['llm'].warning(f"⚠️ Input contains only whitespace - skipping translation")
        return ""
    
    # Reject inputs that are too long (likely concatenated or corrupted)
    if input_length > MAX_INPUT_LENGTH:
        LOGGERS['llm'].warning(f"⚠️ Input too long ({input_length} chars > {MAX_INPUT_LENGTH}) - truncating")
        hindi_text = hindi_text[:MAX_INPUT_LENGTH]  # Truncate to max length
        input_length = len(hindi_text)
    
    
    # ============================================================================
    # CRITICAL FIX #2: ISOLATED CONTEXT - Prevent Context Bleeding
    # ============================================================================
    # Create FRESH isolated context for each translation to prevent LLM from
    # seeing previous messages and trying to merge/continue them
    # Strategy: Only system message + current user message (zero context bleeding)
    async with translation_lock:
        # Create isolated context with ONLY system message + current input
        isolated_ctx = llm.ChatContext()
        
        # Find and copy system message
        system_msg = None
        for item in chat_ctx.items:
            if item.type == "message" and item.role == "system":
                system_msg = item
                break
        
        if system_msg:
            isolated_ctx.add_message(role="system", content=system_msg.content)
        else:
            # Fallback: add system prompt directly
            isolated_ctx.add_message(role="system", content=TRANSLATION_PROMPT)
        
        # Add ONLY the current user message (no previous messages)
        isolated_ctx.add_message(role="user", content=hindi_text)
        
        # Use isolated context for translation (prevents context bleeding)
        translation_ctx = isolated_ctx
        
        # Still update main context for tracking (but LLM won't see it)
        # Keep only last 2-3 user messages for minimal pronoun reference
        prune_chat_context(chat_ctx, logger_name='llm')
        if len([item for item in chat_ctx.items if item.type == "message" and item.role == "user"]) >= 3:
            # Keep only last 2 user messages
            system_msg_main = None
            for item in chat_ctx.items:
                if item.type == "message" and item.role == "system":
                    system_msg_main = item
                    break
            if system_msg_main:
                recent_user_msgs = [item for item in chat_ctx.items[-2:] 
                                  if item.type == "message" and item.role == "user"]
                chat_ctx.items = [system_msg_main] + recent_user_msgs
        chat_ctx.add_message(role="user", content=hindi_text)
        
    trans_start = time.time()
    LOGGERS['llm'].info(f"Translation started - Hindi text: '{hindi_text[:100]}...' ({input_length} chars)")
        
    full_translation = ""
    tts_start_time = None
    audio_task = None  # Initialize to None - will be created when TTS stream opens
        
    try:
        # Open ONE persistent TTS stream at the start (Water Hose method)
        # This maintains prosody continuity across the entire response
        # CRITICAL FIX: Get TTS instance dynamically to ensure current voice ID is used
        tts_instance = get_elevenlabs_tts()
        if not tts_instance:
            LOGGERS['error'].error("ElevenLabs TTS not available - cannot generate audio")
            return ""
        
        async with tts_semaphore:
            LOGGERS['tts'].info(f"ElevenLabs TTS stream opening - Voice: {ELEVENLABS_VOICE_ID}, Model: {ELEVENLABS_MODEL_ID}, Sample Rate: 24kHz")
            tts_stream = None
            try:
                tts_stream = tts_instance.stream()
                async with tts_stream:
                    LOGGERS['tts'].debug("ElevenLabs TTS stream opened successfully")
                    
                    # CRITICAL FIX: Initialize variables BEFORE defining consume_audio to prevent race condition
                    # If consume_audio starts immediately and hits an error, these variables must already exist
                    llm_chunk_count = 0
                    tts_text_chunks = 0
                    total_tts_chars = 0
                    translation_timeout_occurred = False
                    translation_success = False
                    tts_chunk_buffer = ""  # Buffer for TTS text chunks
                    
                    # Background task to consume audio from the stream
                    # EXACT CARTESIA PATTERN: Simple async for loop, no complex timeout monitoring
                    # This matches Cartesia's implementation exactly for consistent low latency
                    async def consume_audio():
                        # CRITICAL FIX: Add nonlocal declarations to prevent UnboundLocalError
                        # This prevents crashes if these variables are ever modified inside the function
                        nonlocal tts_start_time, tts_text_chunks, total_tts_chars
                        tts_start_time = time.time()
                        audio_frame_count = 0
                        total_audio_bytes = 0
                        try:
                            LOGGERS['tts'].debug("Starting audio consumption loop - waiting for frames from ElevenLabs")
                            async for audio in tts_stream:
                                if shutdown_event.is_set():
                                    LOGGERS['tts'].warning("TTS audio consumption stopped - shutdown event")
                                    break
                                if audio.frame:
                                    audio_frame_count += 1
                                    frame_bytes = len(audio.frame.data) if hasattr(audio.frame, 'data') else 0
                                    total_audio_bytes += frame_bytes
                                    try:
                                        audio_queue.put_nowait(audio.frame)
                                        if audio_frame_count % TTS_FRAME_LOG_INTERVAL == 0:  # Log every Nth frame
                                            LOGGERS['tts'].debug(f"TTS audio frame #{audio_frame_count} - {frame_bytes} bytes, Queue size: {audio_queue.qsize()}")
                                    except asyncio.QueueFull:
                                        # ========================================================================
                                        # QUEUE BACKPRESSURE HANDLING: FIFO Drop Strategy
                                        # ========================================================================
                                        # Problem: Audio queue is full (2500 frames = ~50 seconds of audio)
                                        #          This happens when ElevenLabs generates faster than playback consumes.
                                        #
                                        # Strategy: Drop oldest frame (FIFO) to make room for new frame
                                        #          Why FIFO? Oldest audio is least important (user already heard it or will soon)
                                        #
                                        # Flow:
                                        #   1. Try to drop oldest frame (get_nowait)
                                        #   2. If successful, add new frame
                                        #   3. If queue still full after drop, drop new frame (rare edge case)
                                        #
                                        # Trade-off: Better to drop frames than block (real-time performance > perfect audio)
                                        # ========================================================================
                                        try:
                                            dropped_frame = audio_queue.get_nowait()
                                            audio_queue.task_done()
                                            LOGGERS['queue'].warning(f"Audio queue full - Dropped frame, Queue size: {audio_queue.qsize()}")
                                        except asyncio.QueueEmpty:
                                            pass  # Queue became empty (unlikely but handle it)
                                        
                                        # Now try to add new frame
                                        try:
                                            audio_queue.put_nowait(audio.frame)
                                        except asyncio.QueueFull:
                                            LOGGERS['queue'].error("Audio queue still full after drop attempt - dropping new frame")
                                            pass  # Drop this frame
                                else:
                                    # Log when we receive audio events without frames (for debugging)
                                    LOGGERS['tts'].debug(f"Received audio event without frame: {type(audio)}")
                            elapsed = int((time.time() - tts_start_time) * 1000) if tts_start_time else 0
                            if audio_frame_count == 0:
                                LOGGERS['tts'].error(f"⚠️ CRITICAL: TTS audio consumption completed with 0 frames!")
                                LOGGERS['tts'].error(f"⚠️ This means no audio was generated. Possible causes:")
                                LOGGERS['tts'].error(f"⚠️ 1. Stream closed before ElevenLabs could send audio")
                                LOGGERS['tts'].error(f"⚠️ 2. end_input() called too early or stream structure issue")
                                LOGGERS['tts'].error(f"⚠️ 3. ElevenLabs API issue (check API key and voice ID)")
                                LOGGERS['tts'].error(f"⚠️ Time: {elapsed}ms, Text chunks: {tts_text_chunks}, Total chars: {total_tts_chars}")
                            else:
                                LOGGERS['tts'].info(f"TTS audio consumption completed - Frames: {audio_frame_count}, Total bytes: {total_audio_bytes}, Time: {elapsed}ms")
                        except Exception as e:
                            elapsed = int((time.time() - tts_start_time) * 1000) if tts_start_time else 0
                            error_msg = str(e)
                            LOGGERS['error'].error(f"TTS audio consumption exception after {elapsed}ms: {error_msg}")
                            # Check for specific connection errors
                            if "connection closed" in error_msg.lower() or "status_code=-1" in error_msg:
                                LOGGERS['error'].error(f"TTS WebSocket connection closed - This may indicate:")
                                LOGGERS['error'].error(f"  1. Invalid voice ID (current: {ELEVENLABS_VOICE_ID})")
                                LOGGERS['error'].error(f"  2. API key issues")
                                LOGGERS['error'].error(f"  3. Network connectivity problems")
                                LOGGERS['error'].error(f"  4. Rate limiting or quota exceeded")
                            pass
                    
                    # ========================================================================
                    # AUDIO TASK LIFECYCLE: Critical Coordination for TTS Streaming
                    # ========================================================================
                    # Strategy: Start audio consumption BEFORE pushing text to TTS stream
                    # Why: ElevenLabs sends audio frames immediately when text is pushed.
                    #      If we push text before the consumption loop is ready, frames may be lost.
                    # 
                    # Flow:
                    #   1. Create audio_task (starts consume_audio() background loop)
                    #   2. Wait 10ms for task to start iterating (ensures it's waiting for frames)
                    #   3. Push text to TTS stream (ElevenLabs generates frames)
                    #   4. consume_audio() receives frames and queues them
                    #   5. After end_input(), wait for audio_task to finish (ensures all frames received)
                    # ========================================================================
                    audio_task = asyncio.create_task(consume_audio())
                    # Give the task a moment to start iterating (ensures it's waiting for frames)
                    # This 10ms delay is critical - without it, frames may be lost if ElevenLabs
                    # sends them faster than the consumption loop can start
                    await asyncio.sleep(0.01)  # 10ms - allow task to start
                    
                    try:
                        # CRITICAL FIX: Context pruning is handled in process_with_context_update()
                        # No need for duplicate pruning here - it causes conflicts and race conditions
                        # The process_with_context_update() function prunes to CONTEXT_MAX_MESSAGES (6 items)
                        
                        # NOTE: Variables (llm_chunk_count, tts_text_chunks, total_tts_chars, etc.) 
                        # are now initialized BEFORE consume_audio definition to prevent race condition
                        
                        # Calculate adaptive timeout based on text length
                        adaptive_timeout = get_adaptive_translation_timeout(len(hindi_text))
                        
                        # Translation attempt (retries disabled for real-time performance - fail fast)
                        for retry_attempt in range(MAX_TRANSLATION_RETRIES + 1):
                            if retry_attempt > 0:
                                # Exponential backoff: base_delay * (multiplier ^ (attempt - 1))
                                retry_delay = TRANSLATION_RETRY_DELAY_BASE * (TRANSLATION_RETRY_BACKOFF_MULTIPLIER ** (retry_attempt - 1))
                                LOGGERS['llm'].info(f"Retrying translation (attempt {retry_attempt + 1}/{MAX_TRANSLATION_RETRIES + 1}) after {retry_delay:.2f}s delay (exponential backoff)")
                                await asyncio.sleep(retry_delay)
                                trans_start = time.time()  # Reset timeout timer for retry
                                # Increase timeout on retry (network might be slow, allow more time for completion)
                                adaptive_timeout = min(adaptive_timeout * TIMEOUT_RETRY_MULTIPLIER, TIMEOUT_RETRY_MAX)  # Increased timeout for retries
                            
                            try:
                                translation_timeout_occurred = False  # Reset for each attempt
                                async with MAIN_LLM.chat(chat_ctx=translation_ctx) as llm_stream:
                                    async for chunk in llm_stream:
                                        if shutdown_event.is_set():
                                            LOGGERS['llm'].warning("LLM streaming stopped - shutdown event")
                                            break
                                            
                                        # Check timeout with adaptive value - Optimized check (every Nth chunks to reduce overhead)
                                        # CRITICAL FIX: Check timeout more frequently if we're already past the base timeout
                                        elapsed_time = time.time() - trans_start
                                        # Check timeout on every chunk if we're already past threshold, otherwise every Nth chunks
                                        should_check_timeout = (elapsed_time > adaptive_timeout * TIMEOUT_CHECK_EARLY_THRESHOLD) or (llm_chunk_count % TIMEOUT_CHECK_CHUNK_INTERVAL == 0)
                                        if should_check_timeout and elapsed_time > adaptive_timeout:
                                            # Only timeout if we haven't received a chunk recently (stream might be stalled)
                                            # Give extra grace period for stream completion (grace period to prevent incomplete translations)
                                            if elapsed_time > adaptive_timeout * TIMEOUT_GRACE_PERIOD:  # Grace period multiplier
                                                if retry_attempt < MAX_TRANSLATION_RETRIES:
                                                    LOGGERS['llm'].warning(f"Translation timeout after {elapsed_time:.1f}s (adaptive: {adaptive_timeout:.1f}s, text length: {len(hindi_text)} chars, will retry)")
                                                    translation_timeout_occurred = True
                                                    break
                                                else:
                                                    LOGGERS['llm'].warning(f"Translation timeout after {elapsed_time:.1f}s (adaptive: {adaptive_timeout:.1f}s, text length: {len(hindi_text)} chars, max retries reached)")
                                                    translation_timeout_occurred = True
                                                    break
                                        
                                        text = ""
                                        if isinstance(chunk, str):
                                            text = chunk
                                        elif isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.content:
                                            text = chunk.delta.content
                                            
                                        if text:
                                            llm_chunk_count += 1
                                            full_translation += text
                                                
                                                
                                            # ========================================================================
                                            # TTS BUFFERING STRATEGY: The "Sentence" Rule
                                            # ========================================================================
                                            # Problem: LLM streams translation word-by-word. If we send "I" immediately,
                                            #          TTS says "I" with flat intonation. Then "am" arrives, but it's too late.
                                            #          Result: Robotic, choppy speech.
                                            #
                                            # Solution: Buffer text until we hit punctuation (. ? ! ,)
                                            #          This gives TTS the full sentence context for proper intonation.
                                            #
                                            # Example:
                                            #   LLM sends: "I" → buffer
                                            #   LLM sends: " am" → buffer
                                            #   LLM sends: " happy." → buffer ends with "." → SEND TO TTS
                                            #   TTS receives: "I am happy." → Natural intonation ✅
                                            #
                                            # Safety: Hard limit (600 chars) prevents memory leaks on very long sentences
                                            # ========================================================================
                                            
                                            # Accumulate chunks into buffer
                                            tts_chunk_buffer += text
                                            
                                            # Safety check: Prevent memory leaks with hard limit
                                            if len(tts_chunk_buffer) > TTS_BUFFER_HARD_LIMIT:
                                                LOGGERS['tts'].warning(f"TTS buffer exceeded hard limit ({TTS_BUFFER_HARD_LIMIT} chars) - forcing send: '{tts_chunk_buffer[:50]}...'")
                                                # Force send even without punctuation to prevent memory issues
                                                sanitized = sanitize_text_for_tts(tts_chunk_buffer)
                                                if sanitized and len(sanitized) > 0:
                                                    tts_text_chunks += 1
                                                    total_tts_chars += len(sanitized)
                                                    tts_stream.push_text(sanitized)
                                                    LOGGERS['tts'].debug(f"TTS text pushed (hard limit) - Chunk #{tts_text_chunks}, Chars: {len(sanitized)}")
                                                tts_chunk_buffer = ""
                                            else:
                                                # THE SENTENCE RULE: Check if buffer ends with punctuation
                                                # Only send when we have a complete phrase (ends with . ? ! ,)
                                                buffer_len = len(tts_chunk_buffer)
                                                if buffer_len > 0:
                                                    last_char = tts_chunk_buffer[-1]
                                                    buffer_ends_with_punct = last_char in '.!?,'
                                                    
                                                    if buffer_ends_with_punct:
                                                        # We have a complete phrase - send it to TTS
                                                        # This ensures TTS gets the full sentence context for proper intonation
                                                        sanitized = sanitize_text_for_tts(tts_chunk_buffer)
                                                        if sanitized and len(sanitized) > 0:
                                                            tts_text_chunks += 1
                                                            total_tts_chars += len(sanitized)
                                                            tts_stream.push_text(sanitized)
                                                            if tts_text_chunks % TTS_CHUNK_LOG_INTERVAL == 0:  # Log every Nth chunk
                                                                LOGGERS['tts'].debug(f"TTS text pushed (punctuation) - Chunk #{tts_text_chunks}, Chars: {len(sanitized)}, Total: {total_tts_chars}")
                                                            # Clear buffer after successful push
                                                            tts_chunk_buffer = ""
                                                        else:
                                                            # Sanitization failed - clear buffer to prevent issues
                                                            LOGGERS['tts'].warning(f"TTS buffer sanitization failed - clearing: '{tts_chunk_buffer[:50]}...'")
                                                            tts_chunk_buffer = ""
                                                    # If no punctuation yet, keep buffering (wait for complete phrase)
                                
                                    # CRITICAL: Flush any remaining buffered chunks before ending
                                    if tts_chunk_buffer:
                                        sanitized = sanitize_text_for_tts(tts_chunk_buffer)
                                        if sanitized and len(sanitized) > 0:
                                            tts_text_chunks += 1
                                            total_tts_chars += len(sanitized)
                                            tts_stream.push_text(sanitized)
                                            LOGGERS['tts'].debug(f"TTS text pushed (final flush) - Chunk #{tts_text_chunks}, Chars: {len(sanitized)}")
                                        tts_chunk_buffer = ""  # Clear buffer
                                        
                                    # If we got here without timeout, translation succeeded
                                    if not translation_timeout_occurred:
                                        # CRITICAL FIX: LLM now outputs raw text (no JSON) - use directly
                                        cleaned_result = full_translation.strip()
                                        
                                        # ENHANCED: Check if result is truly empty or problematic
                                        if not cleaned_result or len(cleaned_result) == 0:
                                            LOGGERS['llm'].warning(f"⚠️ Translation stream completed but result is empty (chunks: {llm_chunk_count}) - will retry")
                                            
                                            # Don't retry if input was too short (retries won't help)
                                            if input_length < MIN_INPUT_LENGTH_FOR_RETRY:
                                                LOGGERS['llm'].warning(f"⚠️ Input too short ({input_length} chars) - skipping retry (won't help)")
                                                translation_success = False
                                                break
                                            
                                            # Don't retry if we've already retried once (2 attempts total is enough)
                                            if retry_attempt >= 1:
                                                LOGGERS['llm'].warning(f"⚠️ Empty result after {retry_attempt + 1} attempts - likely Groq API issue or invalid input, skipping further retries")
                                                translation_success = False
                                                break
                                            
                                            # ENHANCED: Check if input has problematic patterns before retrying
                                            # If input has excessive ellipsis or incomplete patterns, don't retry
                                            total_dots = hindi_text.count('.')
                                            total_ellipsis_chars = hindi_text.count('…')
                                            if total_dots > MAX_DOTS_THRESHOLD or total_ellipsis_chars > MAX_ELLIPSIS_THRESHOLD or (input_length < MIN_INPUT_LENGTH_SHORT and (total_dots > MAX_DOTS_SHORT_INPUT or total_ellipsis_chars > MAX_ELLIPSIS_SHORT_INPUT)):
                                                LOGGERS['llm'].warning(f"⚠️ Input has problematic patterns (excessive ellipsis: {total_dots} dots, {total_ellipsis_chars} ellipsis) - skipping retry")
                                                translation_success = False
                                                break
                                            
                                            translation_timeout_occurred = True
                                            continue  # Retry if empty result
                                        
                                        # ENHANCED: Post-validation - Check result quality before accepting
                                        # Check if result is meaningful (not just whitespace)
                                        if not cleaned_result.strip():
                                            LOGGERS['llm'].warning(f"⚠️ Translation result is empty - likely invalid")
                                            if retry_attempt >= 1:
                                                LOGGERS['llm'].warning(f"⚠️ Empty result after {retry_attempt + 1} attempts - skipping further retries")
                                                translation_success = False
                                                break
                                            translation_timeout_occurred = True
                                            continue  # Retry if empty
                                        
                                        # Check if result is too short (likely incomplete)
                                        if len(cleaned_result) < MIN_OUTPUT_LENGTH:
                                            LOGGERS['llm'].warning(f"⚠️ Translation result too short ({len(cleaned_result)} chars) - likely incomplete")
                                            if retry_attempt < MAX_TRANSLATION_RETRIES:
                                                translation_timeout_occurred = True
                                                continue  # Retry if too short
                                            else:
                                                # Accept short result if max retries reached
                                                translation_success = True
                                                break
                                        
                                        # Success: We have a meaningful translation
                                        translation_success = True
                                        break  # Exit retry loop on success
                                        
                            except Exception as e:
                                error_msg = str(e)
                                error_str_lower = error_msg.lower()
                                
                                # CRITICAL FIX: Detect and handle rate limit errors (429)
                                is_rate_limit = (
                                    "429" in error_msg or 
                                    "rate limit" in error_str_lower or 
                                    "rate_limit" in error_str_lower or
                                    "tokens per day" in error_str_lower or
                                    "tpd" in error_str_lower
                                )
                                
                                if is_rate_limit:
                                    # CRITICAL: Free tier rate limit detected
                                    # Error shows: "service tier `on_demand` on tokens per day (TPD): Limit 100000"
                                    # This is the FREE TIER limitation - upgrading to paid tier will fix this
                                    LOGGERS['error'].error(f"🚨 GROQ API RATE LIMIT EXCEEDED (FREE TIER LIMITATION)")
                                    LOGGERS['error'].error(f"🚨 Service Tier: on_demand (FREE TIER)")
                                    LOGGERS['error'].error(f"🚨 Daily Token Limit: 100,000 tokens per day (TPD)")
                                    LOGGERS['error'].error(f"🚨 Error Details: {error_msg[:300]}...")
                                    LOGGERS['error'].error(f"🚨 SOLUTION: Upgrade to paid tier (Pay-as-you-go) at: https://console.groq.com/settings/billing")
                                    LOGGERS['error'].error(f"🚨 After upgrading, you'll have much higher limits and this error will be resolved")
                                    translation_success = False
                                    break  # Exit retry loop - no point retrying on free tier limit
                                
                                # Handle other errors with standard retry logic
                                if retry_attempt < MAX_TRANSLATION_RETRIES:
                                    LOGGERS['error'].warning(f"Translation error (will retry): {error_msg[:200]}...")
                                    translation_timeout_occurred = True  # Mark for retry
                                    continue  # Continue to next retry attempt
                                else:
                                    LOGGERS['error'].error(f"Translation error (max retries reached): {error_msg[:200]}...")
                                    raise  # Re-raise on final attempt
                        
                        if translation_success:
                            LOGGERS['llm'].info(f"LLM streaming completed - Chunks: {llm_chunk_count}, Translation length: {len(full_translation)} chars")
                        else:
                            LOGGERS['llm'].warning(f"LLM streaming failed after {MAX_TRANSLATION_RETRIES + 1} attempts - Chunks: {llm_chunk_count}, Translation length: {len(full_translation)} chars")
                        LOGGERS['tts'].info(f"TTS text push completed - Chunks: {tts_text_chunks}, Total chars: {total_tts_chars}")
                        
                        # 🚨 CRITICAL LATENCY FIX: Call end_input() IMMEDIATELY after sending all text
                        # This signals ElevenLabs we're done sending text, so it can generate audio
                        try:
                            if tts_stream is not None:
                                tts_stream.end_input()
                                LOGGERS['tts'].debug("TTS stream end_input() called - text sending complete")
                        except (AttributeError, Exception) as e:
                            LOGGERS['tts'].warning(f"Could not call end_input() on TTS stream: {str(e)}")
                    
                    except asyncio.TimeoutError:
                        # Call end_input() even on timeout to prevent deadlock
                        try:
                            if tts_stream is not None:
                                tts_stream.end_input()
                                LOGGERS['tts'].debug("TTS stream end_input() called (timeout)")
                        except (NameError, AttributeError, UnboundLocalError, Exception):
                            pass  # Stream not initialized yet or already closed
                    except Exception as e:
                        error_msg = str(e)
                        # CRITICAL FIX: Better error handling for ElevenLabs WebSocket failures
                        if "connection closed" in error_msg.lower() or "status_code=-1" in error_msg or "websocket" in error_msg.lower():
                            LOGGERS['error'].error(f"ElevenLabs TTS WebSocket error: {error_msg}")
                            LOGGERS['error'].error(f"This may indicate:")
                            LOGGERS['error'].error(f"  1. Invalid voice ID (current: {ELEVENLABS_VOICE_ID})")
                            LOGGERS['error'].error(f"  2. API key issues")
                            LOGGERS['error'].error(f"  3. Network connectivity problems")
                            LOGGERS['error'].error(f"  4. Rate limiting or quota exceeded")
                        
                        # CRITICAL FIX: Ensure audio_task is cleaned up if LLM/translation fails
                        # This prevents RuntimeWarning: Task 'consume_audio' was destroyed but it is pending!
                        if audio_task is not None and not audio_task.done():
                            LOGGERS['tts'].debug("Cleaning up audio_task due to translation error")
                            audio_task.cancel()
                            try:
                                await audio_task
                            except asyncio.CancelledError:
                                pass
                            except Exception as task_e:
                                LOGGERS['error'].warning(f"Audio task cleanup error: {task_e}")
                        
                        # Call end_input() even on error to prevent deadlock
                        try:
                            if tts_stream is not None:
                                tts_stream.end_input()
                                LOGGERS['tts'].debug("TTS stream end_input() called (error)")
                        except (NameError, AttributeError, UnboundLocalError, Exception):
                            pass  # Stream not initialized yet or already closed
                    
                    # 🚨 CRITICAL FIX: Wait for audio consumption BEFORE exiting async with block
                    # We must wait for consume_audio() to finish receiving frames from ElevenLabs
                    # Otherwise the stream closes and we get 0 frames (no audio)
                    # This is different from waiting for playback - we're waiting for TTS to send frames to our queue
                    # NOTE: If audio_task was already cancelled in the exception handler above, it will be done()
                    if audio_task is not None and not audio_task.done():
                        try:
                            # Wait for audio consumption with reasonable timeout (10 seconds)
                            await asyncio.wait_for(audio_task, timeout=10.0)
                        except asyncio.TimeoutError:
                            LOGGERS['tts'].warning("Audio consumption timeout - cancelling task")
                            audio_task.cancel()
                            try:
                                await audio_task
                            except Exception:
                                pass
                        except asyncio.CancelledError:
                            # Task was already cancelled (e.g., in exception handler above)
                            pass
                    elif audio_task is not None and audio_task.done():
                        # Task already completed or was cancelled - no need to wait
                        LOGGERS['tts'].debug("Audio task already completed/cancelled - skipping wait")
                    
                    # Log TTS timing
                    if tts_start_time:
                        tts_elapsed = int((time.time() - tts_start_time) * 1000)
                        if tts_elapsed > 0:
                            print(f"T2_TTS: {tts_elapsed}ms")
                            LOGGERS['perf'].info(f"T2_TTS latency: {tts_elapsed}ms")
            except Exception as e:
                # CRITICAL FIX: Handle TTS stream creation failures
                error_msg = str(e)
                LOGGERS['error'].error(f"ElevenLabs TTS stream creation failed: {error_msg}")
                LOGGERS['error'].error(f"This may indicate:")
                LOGGERS['error'].error(f"  1. Invalid voice ID (current: {ELEVENLABS_VOICE_ID})")
                LOGGERS['error'].error(f"  2. API key issues")
                LOGGERS['error'].error(f"  3. Network connectivity problems")
                LOGGERS['error'].error(f"  4. Rate limiting or quota exceeded")
                return ""  # Return empty to skip this translation
        
    except Exception as e:
        LOGGERS['error'].error(f"Translation processing error: {str(e)}")
        pass
        
    trans_end = time.time()
    trans_elapsed = int((trans_end - trans_start) * 1000)
    if trans_elapsed > 0:
        print(f"T2_TRANS: {trans_elapsed}ms")
        LOGGERS['perf'].info(f"T2_TRANS total latency: {trans_elapsed}ms")
        
    # Return cleaned translation with post-processing to remove incomplete markers
    # CRITICAL FIX: LLM now outputs raw text (no JSON) - use directly
    cleaned_translation = full_translation.strip()
        
        
    # ============================================================================
    # CRITICAL FIX #3: Keep ALL sentences - Never throw away user speech
    # ============================================================================
    # Split by sentence boundaries (., !, ?) but KEEP ALL sentences
    # We must NEVER throw away user speech. If the LLM output 3 sentences, we play 3 sentences.
    sentences = re.split(r'([.!?]+)', cleaned_translation)
    # Reconstruct sentences with their punctuation
    reconstructed_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
            if sentence:
                reconstructed_sentences.append(sentence)
    # If odd number of parts, add the last one
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        reconstructed_sentences.append(sentences[-1].strip())
    
    # CRITICAL: Keep EVERYTHING - join all sentences
    if len(reconstructed_sentences) > 1:
        LOGGERS['llm'].info(f"✅ Multiple sentences detected ({len(reconstructed_sentences)} sentences) - keeping ALL sentences to prevent data loss")
        cleaned_translation = " ".join(reconstructed_sentences)
    elif reconstructed_sentences:
        cleaned_translation = reconstructed_sentences[0]
    else:
        # Fallback: use original if splitting failed
        cleaned_translation = cleaned_translation.strip()
    
    # Detect context bleeding (output contains previous translations or extra content)
    # CRITICAL FIX: Only log warning, DO NOT truncate - Hindi-to-English translation often expands significantly
    output_length = len(cleaned_translation)
    expansion_ratio = output_length / input_length if input_length > 0 else 0
    
    if expansion_ratio > MAX_OUTPUT_EXPANSION:
        LOGGERS['llm'].warning(f"⚠️ SUSPICIOUS OUTPUT EXPANSION: Input: {input_length} chars, Output: {output_length} chars (ratio: {expansion_ratio:.2f}x)")
        LOGGERS['llm'].warning(f"⚠️ This may indicate context bleeding - output may contain extra words or context")
        # CRITICAL: DO NOT truncate - truncating destroys meaning. Hindi-to-English naturally expands.
    
    # Quality check: Ensure translation is not empty or too short
    if len(cleaned_translation.strip()) < MIN_OUTPUT_LENGTH_MEANINGFUL:
        LOGGERS['llm'].warning(f"⚠️ Translation too short ({len(cleaned_translation)} chars) - may be incomplete")
        
    # Remove trailing ellipsis that indicate incomplete translations
    # This handles cases where LLM generates "..." thinking it's continuing
    while cleaned_translation.endswith('...'):
        cleaned_translation = cleaned_translation[:-3].strip()
        
    # Remove trailing incomplete sentence markers
    if cleaned_translation.endswith('…'):
        cleaned_translation = cleaned_translation[:-1].strip()
        
    # Ensure translation ends with proper punctuation if it's a complete sentence
    if cleaned_translation and not cleaned_translation[-1] in '.!?':
        # Only add period if it looks like a complete sentence (has subject-verb structure)
        # Don't add if it's a fragment or list item
        if len(cleaned_translation.split()) > MIN_SENTENCE_LENGTH and not cleaned_translation.endswith(','):
            cleaned_translation += '.'
        
    # Final validation: Check for incomplete translations
    if cleaned_translation.endswith('...') or cleaned_translation.endswith('…'):
        LOGGERS['llm'].warning(f"⚠️ Translation appears incomplete (ends with ellipsis): '{cleaned_translation[-50:]}'")
        # Try to remove the incomplete ending and add proper punctuation
        cleaned_translation = cleaned_translation.rstrip('…').rstrip('...').strip()
        if cleaned_translation and not cleaned_translation[-1] in '.!?':
            cleaned_translation += '.'
    
    # FINAL CHECK: Ensure we have valid translation (keep all sentences)
    # No longer restricting to single sentence - we keep everything the user said
        
    LOGGERS['llm'].info(f"Translation completed - Result: '{cleaned_translation[:100]}{'...' if len(cleaned_translation) > 100 else ''}' ({len(cleaned_translation)} chars, expansion: {expansion_ratio:.2f}x)")
    return cleaned_translation


async def translate_and_synthesize_loop(translation_queue: asyncio.Queue, 
                                    audio_queue: asyncio.Queue, 
                                    shutdown_event: asyncio.Event) -> None:
    """Team 2: Brain - Sequential TTS with optimized pre-buffering for continuous flow"""
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=TRANSLATION_PROMPT)
        
    LOGGERS['llm'].info(f"Translation context initialized - System prompt: '{TRANSLATION_PROMPT[:100]}...'")
        
    tts_semaphore = asyncio.Semaphore(TTS_CONCURRENT_LIMIT)
    active_translations = set()
    translation_lock = asyncio.Lock()  # For thread-safe context updates
    translation_count = 0
        
    LOGGERS['main'].info(f"Translation loop ready - TTS concurrent limit: {TTS_CONCURRENT_LIMIT}")
        
    try:
        while not shutdown_event.is_set():
            try:
                # Get next translation (non-blocking with timeout)
                try:
                    hindi_text = await asyncio.wait_for(
                        translation_queue.get(), 
                        timeout=TRANSLATION_QUEUE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    continue
                    
                if hindi_text is None:
                    LOGGERS['queue'].info("Received shutdown signal in translation queue")
                    break
                    
                translation_count += 1
                LOGGERS['queue'].info(f"Translation #{translation_count} dequeued - Text: '{hindi_text[:100]}...'")
                    
                # Process translation in parallel (don't wait for completion)
                async def process_with_context_update(text: str):
                    """Process translation and update context with quality validation"""
                    current_task = asyncio.current_task()
                    try:
                        full_translation = await process_single_translation(
                            text, chat_ctx, audio_queue,
                            tts_semaphore, shutdown_event, translation_lock
                        )
                            
                        # CRITICAL FIX: DO NOT store assistant messages to prevent hallucination
                        # Only user messages are kept in context for pronoun reference
                        # Storing assistant translations causes LLM to continue/merge previous translations
                        if full_translation and len(full_translation.strip()) >= 3:
                            # Translation successful - but DO NOT add to context
                            # Context only contains user messages (Hindi inputs) for pronoun understanding
                            # This prevents context bleeding and hallucination
                            async with translation_lock:
                                # Prune context to keep it clean (remove any stray assistant messages)
                                context_size_before = len(chat_ctx.items)
                                context_size_after = prune_chat_context(chat_ctx, logger_name='llm')
                                if context_size_after < context_size_before:
                                    LOGGERS['llm'].debug(f"Context pruned - Before: {context_size_before}, After: {context_size_after}")
                                else:
                                    LOGGERS['llm'].debug(f"Context maintained - Size: {len(chat_ctx.items)} (only user messages for pronoun reference)")
                        else:
                            # Empty or invalid translation - don't add to context
                            LOGGERS['llm'].warning(f"⚠️ Skipping context update - translation empty or too short: '{full_translation[:50] if full_translation else 'EMPTY'}...'")
                                    
                    except Exception as e:
                        LOGGERS['error'].error(f"Translation processing error: {str(e)}")
                        pass
                    finally:
                        translation_queue.task_done()
                        if current_task:
                            active_translations.discard(current_task)
                    
                # Start translation task (runs in parallel)
                task = asyncio.create_task(process_with_context_update(hindi_text))
                active_translations.add(task)
                    
                # Cleanup completed translations (keep max active for better quality)
                # CRITICAL FIX: Cleanup more frequently to prevent memory buildup
                # FIX: Use in-place modification instead of reassignment to prevent stale references
                if len(active_translations) > ACTIVE_TRANSLATIONS_CLEANUP_THRESHOLD or translation_count % ACTIVE_TRANSLATIONS_CLEANUP_INTERVAL == 0:
                    # Remove done tasks in-place (safer than reassignment - prevents stale references)
                    done_tasks = {t for t in active_translations if t.done()}
                    active_translations.difference_update(done_tasks)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                try:
                    translation_queue.task_done()
                except ValueError:
                    pass
    finally:
        # Wait for active translations to complete
        if active_translations:
            await asyncio.gather(*active_translations, return_exceptions=True)


# ============================================================================
# TEAM 3: THE MOUTH (playback_loop)
# Picks audio frames from Conveyor Belt B and plays them
# ZERO LATENCY: Direct playback without resampling (removed CPU-intensive NumPy operations)
# ============================================================================

async def playback_loop(source: rtc.AudioSource, audio_queue: asyncio.Queue, 
                    shutdown_event: asyncio.Event) -> None:
    """
    Team 3: Mouth - Direct frame playback (LiveKit handles buffering)
    
    ⚠️ ARCHITECTURAL LIMITATION: No "Barge-In" / Interruption Handling
    This playback loop plays everything in the audio_queue sequentially without interruption.
    Drawback: If the user interrupts the translation (starts speaking while the bot is speaking),
    the bot will keep talking until the queue is empty, which feels unnatural in arguments or
    rapid-fire conversation.
    Impact: No natural conversation flow - bot cannot be interrupted mid-sentence.
    Recommendation: Implementing a "Clear Queue on User Speech" feature would require:
    - Listening for START_OF_SPEECH events from STT while playing audio
    - Clearing audio_queue when user speech is detected
    - Coordinating between listen_loop and playback_loop
    This adds complexity but would significantly improve conversation feel.
    """
    LOGGERS['main'].info("Playback loop started - Direct frame playback")
        
    playback_count = 0
    error_count = 0
        
    try:
        while not shutdown_event.is_set():
            try:
                # Get frame from queue (no timeout - let LiveKit handle buffering)
                audio_frame = await audio_queue.get()
                    
                if audio_frame is None:
                    break
                    
                # ⚡ CPU LATENCY FIX: Pass frames directly - no validation/recreation per frame
                # ElevenLabs outputs 24kHz mono frames that match source configuration
                # Python object creation is slow - avoid recreating frames for every 10ms packet
                # Only handle errors if they occur, don't pre-validate every frame
                try:
                    # Push frame directly to LiveKit (it has its own jitter buffer)
                    playback_start = time.time()
                    await source.capture_frame(audio_frame)
                    playback_end = time.time()
                    playback_elapsed = int((playback_end - playback_start) * 1000)
                    playback_count += 1
                    error_count = 0  # Reset error count on success
                        
                    if playback_count % PLAYBACK_LOG_INTERVAL == 0:  # Log every Nth playback
                        LOGGERS['perf'].debug(f"T3_PLAY #{playback_count} - Latency: {playback_elapsed}ms")
                except Exception as e:
                    error_count += 1
                    error_msg = str(e)
                    
                    # Handle format mismatch errors by recreating frame with correct format
                    if "sample_rate" in error_msg.lower() or "num_channels" in error_msg.lower() or "InvalidState" in error_msg:
                        try:
                            # Get frame data and recreate with correct format (24kHz, 1 channel)
                            frame_data = None
                            samples_per_channel = None
                            
                            # Try to get frame data
                            if hasattr(audio_frame, 'data') and audio_frame.data:
                                frame_data = audio_frame.data
                            elif hasattr(audio_frame, '__bytes__'):
                                try:
                                    frame_data = bytes(audio_frame)
                                except Exception:
                                    pass
                            
                            # Try to get samples_per_channel from original frame (most accurate)
                            if hasattr(audio_frame, 'samples_per_channel'):
                                samples_per_channel = audio_frame.samples_per_channel
                            
                            if frame_data:
                                # Calculate samples_per_channel if not available from frame
                                # For 16-bit PCM (2 bytes per sample), mono (1 channel)
                                # samples_per_channel = total_bytes / (bytes_per_sample * num_channels)
                                if samples_per_channel is None:
                                    samples_per_channel = len(frame_data) // 2  # 16-bit = 2 bytes per sample
                                
                                # Recreate frame with source format (24kHz, 1 channel)
                                corrected_frame = rtc.AudioFrame(
                                    data=frame_data,
                                    sample_rate=24000,
                                    num_channels=1,
                                    samples_per_channel=samples_per_channel
                                )
                                # Retry with corrected frame
                                await source.capture_frame(corrected_frame)
                                playback_count += 1
                                error_count = 0  # Reset on successful retry
                                if playback_count <= 10:  # Log first few fixes
                                    LOGGERS['error'].debug(f"Fixed format mismatch - recreated frame (24kHz, 1ch, {samples_per_channel} samples)")
                                # Success - frame played, continue to next iteration
                                # (audio_queue.task_done() will be called below)
                        except Exception as retry_error:
                            # If retry also fails, log and continue
                            if error_count <= 5:
                                LOGGERS['error'].error(f"Playback error #{error_count} (retry failed): {str(retry_error)}")
                    else:
                        # Other errors - just log
                        if error_count <= 5:  # Only log first few errors to avoid spam
                            LOGGERS['error'].error(f"Playback error #{error_count}: {error_msg}")
                    # Continue processing - don't break on individual frame errors
                    # Mark task done even if there was an error (we tried our best)
                    audio_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGERS['error'].error(f"Playback loop error: {str(e)}")
                pass
    except asyncio.CancelledError:
        pass


async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint - The Factory Line coordinator"""
    # CRITICAL FIX: Set up exception handler for asyncio callbacks (catches winloop._read_from_self errors)
    def handle_exception(loop, context):
        """Handle exceptions in asyncio callbacks (like winloop._read_from_self)"""
        exception = context.get('exception')
        if exception:
            error_type_name = type(exception).__name__
            error_module = type(exception).__module__ if hasattr(type(exception), '__module__') else ''
            
            # Suppress _ExitCli exceptions in callbacks (expected shutdown signal)
            if (error_type_name == '_ExitCli' or 
                'ExitCli' in error_type_name or 
                'livekit.agents.cli.cli' in error_module):
                # This is expected - don't log as error
                return
        
        # Log other exceptions normally
        loop.default_exception_handler(context)
    
    # Set the exception handler
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_exception)
    
    # Initialize memory monitoring if available
    memory_monitor = None
    memory_task = None
    try:
        from memory_monitor import init_memory_monitor, memory_monitoring_loop
        memory_monitor = init_memory_monitor(log_interval=5.0, alert_threshold_mb=500.0)
        if memory_monitor:
            LOGGERS['main'].info("✅ Memory monitoring enabled")
    except ImportError:
        LOGGERS['main'].info("ℹ️ Memory monitoring not available (install psutil: pip install psutil)")
    except Exception as e:
        LOGGERS['main'].warning(f"Memory monitoring initialization failed: {e}")
    
    LOGGERS['main'].info("=" * 80)
    LOGGERS['main'].info("AGENT STARTING - Entrypoint called")
    LOGGERS['main'].info(f"Models configured:")
    LOGGERS['main'].info(f"  - STT: Deepgram Nova-2 (Hindi) - Ultra-fast for zero latency")
    LOGGERS['main'].info(f"  - LLM: Groq (llama-3.3-70b-versatile) - Best Quality (superior translation accuracy)")
    LOGGERS['main'].info(f"  - TTS: ElevenLabs ({ELEVENLABS_MODEL_ID}, voice: {ELEVENLABS_VOICE_ID})")
    LOGGERS['main'].info("=" * 80)
    
    # Pre-warm connections to eliminate cold start latency
    LOGGERS['main'].info("Pre-warming connections...")
    try:
        # Pre-warm TTS connection
        # CRITICAL FIX: Get TTS instance dynamically to ensure current voice ID is used
        tts_instance = get_elevenlabs_tts()
        if tts_instance:
            LOGGERS['tts'].info("Pre-warming ElevenLabs TTS connection...")
            try:
                tts_instance.prewarm()
                LOGGERS['tts'].info("ElevenLabs TTS connection pre-warmed successfully")
            except Exception as warm_error:
                LOGGERS['tts'].warning(f"ElevenLabs pre-warming failed (non-critical): {str(warm_error)}")
        else:
            LOGGERS['tts'].warning("ElevenLabs TTS not available - skipping pre-warming")
    except Exception as e:
        LOGGERS['tts'].warning(f"TTS pre-warming failed (non-critical): {str(e)}")
        # Continue anyway - pre-warming is optional
        
    # Pre-warm STT connection by creating a test stream
    try:
        LOGGERS['stt'].info("Pre-warming STT connection...")
        test_stt_stream = DEEPGRAM_STT.stream()
        await asyncio.sleep(0.1)  # Brief connection establishment
        try:
            await test_stt_stream.aclose()
        except Exception:
            pass
        LOGGERS['stt'].info("STT connection pre-warmed successfully")
    except Exception as e:
        LOGGERS['stt'].warning(f"STT pre-warming failed (non-critical): {str(e)}")
        # Continue anyway - pre-warming is optional
        
    # LLM connection ready (Groq API is stateless, no pre-warming needed)
    LOGGERS['main'].info("LLM connection ready (Groq API)")
        
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    LOGGERS['main'].info("Connected to LiveKit room")
        
    room = ctx.room
    if not room:
        LOGGERS['error'].error("Room not available")
        return
        
    # Wait for local participant
    waited = 0
    while not room.local_participant and waited < 10:
        await asyncio.sleep(0.2)
        waited += 0.2
        
    local_participant = room.local_participant
    if not local_participant:
        LOGGERS['error'].error("Local participant not available")
        return
        
    LOGGERS['main'].info("Local participant ready")
        
    # Set up audio output (24kHz mono - standard for voice)
    source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("tts-output", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        
    try:
        await local_participant.publish_track(track, options)
        LOGGERS['main'].info("Audio track published - Sample rate: 24kHz, Channels: 1")
    except Exception as e:
        LOGGERS['error'].error(f"Failed to publish audio track: {str(e)}")
        return
        
    # Create queues (bounded to prevent memory leaks)
    translation_queue = asyncio.Queue(maxsize=50)
    # Increase to 2500 (approx 50 seconds of audio) to prevent dropping frames 
    # when ElevenLabs generates faster than real-time playback.
    audio_queue = asyncio.Queue(maxsize=2500)
        
    LOGGERS['queue'].info("Queues initialized - Translation: 50, Audio: 2500")
        
    shutdown_event = asyncio.Event()
    
    # Start memory monitoring if available
    if memory_monitor:
        memory_task = asyncio.create_task(memory_monitoring_loop(memory_monitor, shutdown_event))
        LOGGERS['main'].info("Memory monitoring started")
    else:
        memory_task = None
        
    # Start all three teams
    LOGGERS['main'].info("Starting factory line teams...")
    team1_ears = asyncio.create_task(listen_loop(room, translation_queue))
    LOGGERS['main'].info("Team 1 (Ears) started - Deepgram STT listening")
        
    team2_brain = asyncio.create_task(translate_and_synthesize_loop(
        translation_queue, audio_queue, shutdown_event
    ))
    LOGGERS['main'].info("Team 2 (Brain) started - LLM translation + TTS synthesis")
        
    team3_mouth = asyncio.create_task(playback_loop(source, audio_queue, shutdown_event))
    LOGGERS['main'].info("Team 3 (Mouth) started - Audio playback")
    LOGGERS['main'].info("All teams started - Factory line operational")
        
        
    # Run until cancelled - Handle KeyboardInterrupt and _ExitCli gracefully
    try:
        LOGGERS['main'].info("Factory line running - Waiting for tasks...")
        results = await asyncio.gather(team1_ears, team2_brain, team3_mouth, return_exceptions=True)
        # Check if any task completed unexpectedly (not cancelled)
        for i, (task, result) in enumerate(zip([team1_ears, team2_brain, team3_mouth], results), 1):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                LOGGERS['error'].error(f"Team {i} task failed with exception: {str(result)}")
            elif not isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                LOGGERS['main'].warning(f"Team {i} task completed unexpectedly (not cancelled)")
    except KeyboardInterrupt:
        # CRITICAL FIX: Immediately set shutdown event on KeyboardInterrupt
        LOGGERS['main'].warning("⚠️ KeyboardInterrupt detected in entrypoint - Triggering graceful shutdown")
        shutdown_event.set()
        # Re-raise to allow outer handler to catch it
        raise
    except Exception as e:
        # CRITICAL FIX: Handle LiveKit's _ExitCli exception (Windows shutdown signal)
        # Check if it's _ExitCli by type name or module (since it's a private class)
        error_type_name = type(e).__name__
        error_module = type(e).__module__ if hasattr(type(e), '__module__') else ''
        
        if (error_type_name == '_ExitCli' or 
            'ExitCli' in error_type_name or 
            'livekit.agents.cli.cli' in error_module):
            LOGGERS['main'].info("⚠️ LiveKit shutdown signal (_ExitCli) detected - Triggering graceful shutdown")
            shutdown_event.set()
            # Don't re-raise - this is expected for clean shutdown
            return
        
        # Handle other exceptions
        LOGGERS['error'].error(f"Factory line error: {str(e)}")
        # Set shutdown event on any exception to ensure cleanup
        shutdown_event.set()
        pass
    finally:
        # Log final memory stats if monitoring is enabled
        if memory_monitor:
            summary = memory_monitor.get_summary()
            LOGGERS['main'].info("=" * 80)
            LOGGERS['main'].info("MEMORY MONITORING SUMMARY:")
            LOGGERS['main'].info(f"  Baseline: {summary.get('baseline_mb', 0):.2f}MB")
            LOGGERS['main'].info(f"  Current: {summary.get('current_mb', 0):.2f}MB")
            LOGGERS['main'].info(f"  Peak: {summary.get('peak_mb', 0):.2f}MB")
            LOGGERS['main'].info(f"  Max Increase: {summary.get('max_increase_mb', 0):.2f}MB")
            LOGGERS['main'].info(f"  Total Checks: {summary.get('total_checks', 0)}")
            LOGGERS['main'].info(f"  Alerts: {summary.get('alerts', 0)}")
            LOGGERS['main'].info("=" * 80)
        
        # Cancel memory monitoring task
        if memory_task:
            memory_task.cancel()
            try:
                await memory_task
            except asyncio.CancelledError:
                pass
        
        # Graceful shutdown
        LOGGERS['main'].info("Shutting down factory line...")
        shutdown_event.set()
            
        try:
            translation_queue.put_nowait(None)
            LOGGERS['queue'].debug("Sent shutdown signal to translation queue")
        except Exception:
            pass
        try:
            audio_queue.put_nowait(None)
            LOGGERS['queue'].debug("Sent shutdown signal to audio queue")
        except Exception:
            pass
            
        await asyncio.sleep(0.1)
            
        for task in [team1_ears, team2_brain, team3_mouth]:
            if not task.done():
                task.cancel()
            
        tasks_to_wait = [t for t in [team1_ears, team2_brain, team3_mouth] if not t.done()]
        if tasks_to_wait:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_wait, return_exceptions=True),
                    timeout=1.0
                )
            except (asyncio.TimeoutError, Exception):
                for task in tasks_to_wait:
                    if not task.done():
                        try:
                            task.cancel()
                        except Exception:
                            pass
            
        LOGGERS['main'].info("=" * 80)
        LOGGERS['main'].info("AGENT SHUTDOWN COMPLETE")
        LOGGERS['main'].info(f"Log file: {LOGGERS['file']}")
        LOGGERS['main'].info("=" * 80)


if __name__ == "__main__":
    # CRITICAL FIX: Handle Windows shutdown crash gracefully
    # Wrap in try/except to catch KeyboardInterrupt, _ExitCli, and winloop exit signals
    try:
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    except KeyboardInterrupt:
        # User pressed Ctrl+C - trigger clean shutdown
        print("\n⚠️ KeyboardInterrupt detected - Initiating graceful shutdown...")
        # The entrypoint function's finally block will handle cleanup
        # Give it a moment to complete shutdown
        import time
        time.sleep(0.5)
        print("✓ Shutdown complete")
    except Exception as exit_error:
        # CRITICAL FIX: Handle LiveKit's _ExitCli exception (Windows shutdown signal from winloop)
        # Check if it's _ExitCli by type name or module (since it's a private class)
        error_type_name = type(exit_error).__name__
        error_module = type(exit_error).__module__ if hasattr(type(exit_error), '__module__') else ''
        
        if (error_type_name == '_ExitCli' or 
            'ExitCli' in error_type_name or 
            'livekit.agents.cli.cli' in error_module):
            # This is raised by LiveKit's CLI when it receives a shutdown signal
            # It's expected behavior, not an error - suppress the traceback
            print("\n⚠️ LiveKit shutdown signal detected - Clean shutdown completed")
            # Exit gracefully without showing traceback
            import sys
            sys.exit(0)
        # If not _ExitCli, continue to next exception handler
        raise
    except Exception as e:
        # Catch any other exceptions (including winloop._read_from_self errors)
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Check if it's a winloop-related error (even if not _ExitCli)
        if "winloop" in error_msg.lower() or "_read_from_self" in error_msg.lower() or "_ExitCli" in error_type:
            print(f"\n⚠️ Windows event loop shutdown signal detected - This is normal on Windows")
            print("✓ Clean shutdown completed")
            import sys
            sys.exit(0)
        else:
            # Re-raise unexpected errors
            print(f"\n❌ Unexpected error: {error_type}: {error_msg}")
            raise
