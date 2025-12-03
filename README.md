# Voxzen - Zero Latency Hindi→English Translation Agent

A high-performance, real-time Hindi-to-English translation system using LiveKit Agents with optimized async I/O and decoupled pipeline architecture.

## 🏭 Architecture: The Factory Line

Three independent teams working in parallel:

1. **Team 1 (Ears)**: `listen_loop`
   - Continuously listens via Deepgram STT (Nova-3)
   - Feeds transcripts to translation queue
   - Never blocks other teams

2. **Team 2 (Brain)**: `translate_and_synthesize_loop`
   - Translates Hindi→English using Groq LLM (GPT-OSS-20B)
   - Synthesizes audio via Cartesia TTS (Sonic-3)
   - Runs in parallel, never blocking

3. **Team 3 (Mouth)**: `playback_loop`
   - Plays audio with strict real-time pacing
   - Prevents buffer bloat
   - Maintains 1.0x playback speed

## 🚀 Key Features

- **Ultra-Low Latency**: Median ~1.25s end-to-end (best: 815ms)
- **Platform-Optimized**: winloop (Windows) / uvloop (Linux/Mac) for 2-4x faster async I/O
- **Sample Rate Matching**: 24kHz throughout (no CPU resampling overhead)
- **Decoupled Pipeline**: Processing and playback never compete for CPU
- **Production-Ready**: Comprehensive logging, error handling, and performance metrics

## 📋 Prerequisites

- Python 3.12+
- API Keys for:
  - LiveKit (Cloud Project)
  - Deepgram
  - Groq
  - Cartesia
  - Cerebras (optional, fallback)

## 🔧 Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   Create a `.env` file with:
   ```env
   LIVEKIT_URL=wss://your-project.livekit.cloud
   LIVEKIT_API_KEY=your_api_key
   LIVEKIT_API_SECRET=your_api_secret
   DEEPGRAM_API_KEY=your_deepgram_key
   GROQ_API_KEY=your_groq_key
   CARTESIA_API_KEY=your_cartesia_key
   CARTESIA_VOICE_ID=your_voice_id
   CEREBRAS_API_KEY=your_cerebras_key  # Optional
   ```

## 🏃 Running Locally

```bash
python main.py dev
```

## ☁️ Deploying to Fly.io

1. **Install Fly CLI:**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login to Fly.io:**
   ```bash
   fly auth login
   ```

3. **Set environment variables:**
   ```bash
   fly secrets set LIVEKIT_URL=wss://your-project.livekit.cloud
   fly secrets set LIVEKIT_API_KEY=your_api_key
   fly secrets set LIVEKIT_API_SECRET=your_api_secret
   fly secrets set DEEPGRAM_API_KEY=your_deepgram_key
   fly secrets set GROQ_API_KEY=your_groq_key
   fly secrets set CARTESIA_API_KEY=your_cartesia_key
   fly secrets set CARTESIA_VOICE_ID=your_voice_id
   ```

4. **Deploy:**
   ```bash
   fly deploy
   ```

## ⚙️ System Configuration

- **STT**: Deepgram Nova-3 (Hindi language)
- **LLM**: Groq GPT-OSS-20B (~1000 tokens/sec) with Cerebras fallback
- **TTS**: Cartesia Sonic-3 (24kHz, optimized for natural speech)
- **Event Loop**: winloop (Windows) / uvloop (Linux/Mac)
- **Sample Rate**: 24kHz (matched throughout pipeline)
- **Channels**: Mono (1)

## 📊 Performance Metrics

- **STT Latency**: Avg 1.67s, Best 6ms
- **TTS Latency**: Avg 1.22s, Best 800ms
- **Total Pipeline**: Avg 1.30s, Best 815ms, Median 1.25s
- **Async Overhead**: <1ms (winloop/uvloop optimized)

## 🔍 Troubleshooting

- **Event Loop Errors**: Ensure winloop (Windows) or uvloop (Linux/Mac) is installed
- **API Errors**: Verify all API keys are set in environment variables
- **High Latency**: Check network connectivity to API services
- **Audio Issues**: Verify sample rate is 24kHz throughout

## 📝 License

MIT





