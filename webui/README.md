# SimpleUI - Web Interface for MemEvolve

A clean, modern web interface for interacting with MemEvolve's memory-augmented LLM API.

## Features

- **Real-time Streaming**: Live streaming responses from your LLM
- **Memory Integration**: Automatically benefits from MemEvolve's memory system
- **Clean UI**: Dark theme with chat-like interface
- **Configurable**: Easy model and parameter selection
- **History**: Persistent chat history during session

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start MemEvolve server:**
   ```bash
   source .venv/bin/activate
   python scripts/start_api.py
   ```

3. **Launch the web UI:**
   ```bash
   cd webui
   streamlit run main.py --server.port 11437
   ```

4. **Open browser** to `http://localhost:11437`

## Configuration

The UI automatically connects to MemEvolve at `http://localhost:11436/v1`. You can change this in the sidebar settings.

### Environment Variables

If you need to customize the default connection:
- **API URL**: Change in the UI sidebar (default: `http://localhost:11436/v1`)
- **Model**: Select from the dropdown or type custom model name

## How It Works

1. **User Input**: You type a message in the chat input
2. **Memory Injection**: MemEvolve automatically retrieves relevant memories
3. **LLM Processing**: Enhanced prompt sent to your LLM with context
4. **Streaming Response**: Real-time streaming back to the UI
5. **Experience Encoding**: MemEvolve captures the interaction for future learning

## Requirements

- Python 3.8+
- Streamlit
- OpenAI Python client
- Running MemEvolve API server

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SimpleUI  │───▶│  MemEvolve  │───▶│     LLM     │
│  (Streamlit)│    │    API      │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
   Chat UI         Memory Injection     Streaming Response
   History         Experience Learning   Real-time Display
```

## Troubleshooting

**Connection Issues:**
- Ensure MemEvolve server is running on port 11436
- Check API URL in the sidebar settings

**Streaming Problems:**
- Some LLM servers may not support streaming
- Try disabling streaming in your LLM server config

**Memory Not Working:**
- Check MemEvolve logs for memory injection messages
- Ensure MEMEVOLVE_STORAGE_PATH is configured

## Contributing

This is a simple interface - feel free to enhance it with:
- Chat export/import
- Multiple conversation management
- Advanced parameter controls
- Custom themes
- Integration with other LLM APIs