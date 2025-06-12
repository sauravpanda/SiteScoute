# SiteScout 🕵️

A powerful website monitoring tool that checks the status of various websites across different categories using a local LLM and browser automation. It provides detailed analysis of website accessibility and functionality.

## Features

- 🤖 Uses local LLM (Qwen 2.5 32B) for intelligent website status analysis
- 🌐 Real browser visits with multi-tab support
- 📊 JSON output with timestamps and detailed status reports
- 🚀 Parallel processing of websites within categories
- 🔒 Everything stays on your machine
- 📱 Monitors websites across multiple categories (Entertainment, Gaming, AI Platforms, etc.)

## Quick Start

```bash
# Create and activate uv environment
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -r requirements.txt

# Run the monitor
python main.py
```

## Environment Setup

The project uses `uv` for dependency management. Key dependencies:
- browser-use==0.2.5
- faiss-cpu==1.10.0
- langchain-ollama
- python-dotenv

## Output

The tool generates a `website_status.json` file with detailed results:

```json
{
  "timestamp": "2024-03-21 10:30:00",
  "categories": {
    "Entertainment & Media": {
      "Spotify": {
        "status": "UP",
        "url": "https://open.spotify.com",
        "error": null
      }
    }
  }
}
```

## Website Categories

The tool monitors websites across various categories:
- Entertainment & Media
- Gaming
- AI & Technology Platforms
- Google Services
- Cloud & Infrastructure
- E-commerce & Business
- Business Tools & Services
- Smart Home & IoT
- Specialized Business Software
- Education & Development
- Finance & Banking

## Requirements

- Python 3.8+
- Local LLM (Ollama with Qwen 2.5 32B model)
- Chrome browser
- uv package manager

## Configuration

Create a `.env` file in the project root to configure:
- LLM settings
- Browser preferences
- Other environment variables

---

*Built for reliable website monitoring with AI-powered analysis*