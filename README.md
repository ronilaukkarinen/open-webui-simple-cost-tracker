# 💰 Open WebUI Simple Cost Tracker

### Minimalist cost tracking function for [Open WebUI](https://github.com/open-webui/open-webui)

![Open WebUI](https://img.shields.io/badge/Open%20WebUI-222222?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSIjZmZmZmZmIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGhlaWdodD0iMWVtIiBzdHlsZT0iZmxleDpub25lO2xpbmUtaGVpZ2h0OjEiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjFlbSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xNy42OTcgMTJjMCA0Ljk3LTMuOTYyIDktOC44NDkgOUMzLjk2MiAyMSAwIDE2Ljk3IDAgMTJzMy45NjItOSA4Ljg0OC05YzQuODg3IDAgOC44NDkgNC4wMyA4Ljg0OSA5em0tMy42MzYgMGMwIDIuOTI4LTIuMzM0IDUuMzAxLTUuMjEzIDUuMzAxLTIuODc4IDAtNS4yMTItMi4zNzMtNS4yMTItNS4zMDFTNS45NyA2LjY5OSA4Ljg0OCA2LjY5OWMyLjg4IDAgNS4yMTMgMi4zNzMgNS4yMTMgNS4zMDF6Ij48L3BhdGg+PHBhdGggZD0iTTI0IDNoLTMuMzk0djE4SDI0VjN6Ij48L3BhdGg+PC9zdmc+Cg==)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Version](https://img.shields.io/badge/version-1.0.9-blue?style=for-the-badge)

A simple, lightweight cost tracking function that monitors token usage and calculates costs for different AI models in Open WebUI. Track your monthly and daily spending with minimal configuration.

## ✨ What it does

Simple Cost Tracker automatically monitors your AI conversations and provides real-time cost information including:

- **Monthly cost tracking** - Automatically resets each month
- **Daily cost tracking** - Shows today's spending
- **Per-message costs** - Individual message cost breakdown
- **Token usage** - Input and output token counts
- **Multi-provider support** - Works with OpenAI, Anthropic, Gemini, and more

## 🚀 Features

- 📊 **Real-time cost tracking** with status messages
- 🗓️ **Automatic monthly/daily reset** - No manual maintenance
- 🔧 **Easy configuration** via JSON valve
- 💾 **Persistent storage** in simple JSON file
- 🎯 **Accurate token estimation** including memories and system prompts
- 🏷️ **Provider prefix support** (openai., anthropic., gemini., etc.)

## 📋 Installation

1. Download `simple_cost_tracker.py`
2. In Open WebUI, go to **Admin Panel > Functions**
3. Click **Import Function** and upload the file
4. Enable the function

## ⚙️ Configuration

The function includes pre-configured costs for popular models. Update the **Model Costs JSON** valve to add or modify pricing:

```json
{
  "openai.gpt-4.1-mini": {"input": 0.40, "output": 1.6},
  "anthropic.claude-3-5-haiku-latest": {"input": 0.80, "output": 4.0},
  "google.gemini-2.5-flash": {"input": 0.30, "output": 2.50}
}
```

**Important:** Use the exact model name as shown in **Settings > Models**. Include connection prefixes if you use them.

For initial costs, edit `cost_tracker_data.json` file manually, costs will add up as you use the function.

## 💡 Usage

Once enabled, the cost tracker automatically displays status messages after each AI response:

```
0.0123 € this month, 0.0045 € today, 0.0012 € for this message, 1247 tokens used
```

## 📁 Data Storage

Costs are stored in `cost_tracker_data.json` with automatic:

- Monthly reset on the 1st of each month
- Daily reset at midnight
- Persistent tracking across sessions

## 🛠️ Technical Details

- **Input token estimation** includes memories, system prompts, and full context
- **Output token estimation** based on assistant response content  
- **Costs calculated** using standard pricing per 1M tokens (EUR)
- **Unknown models** default to 0 cost (for local models)

## 🆚 Comparison

Unlike complex cost trackers that require API keys and external connections, Simple Cost Tracker:

- ✅ **No API keys required**
- ✅ **No external dependencies** 
- ✅ **Automatic operation**
- ✅ **Minimal configuration**
- ✅ **Local storage only**
