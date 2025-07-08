"""
title: Simple Cost Tracker
author: Roni Laukkarinen
description: A minimalist cost tracking function that tracks token usage and costs per model.
repository_url: https://github.com/ronilaukkarinen/open-webui-simple-cost-tracker
version: 1.0.2
required_open_webui_version: >= 0.5.0
"""

import json
import os
from datetime import datetime, date
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import tiktoken


class SimpleCostTracker:
    def __init__(self, model_costs_json: str = "{}"):
        # Parse model costs from JSON valve
        try:
            self.model_costs = json.loads(model_costs_json) if model_costs_json else {}
        except json.JSONDecodeError:
            # Fallback to empty dict if JSON is invalid
            self.model_costs = {}

        # Storage file for tracking costs
        self.storage_file = "cost_tracker_data.json"
        self.current_month = datetime.now().strftime("%Y-%m")
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.monthly_cost, self.daily_cost = self.load_costs()

    def load_costs(self) -> tuple[float, float]:
        """Load monthly and daily costs from storage file"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)

                    # Get current month and day costs from history
                    monthly_history = data.get('monthly_history', {})
                    daily_history = data.get('daily_history', {})

                    monthly_cost = monthly_history.get(self.current_month, 0.0)
                    daily_cost = daily_history.get(self.current_date, 0.0)

                    return monthly_cost, daily_cost
        except:
            pass
        return 0.0, 0.0

    def save_costs(self):
        """Save monthly and daily costs to storage file with history"""
        try:
            # Load existing data to preserve history
            existing_data = {}
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    existing_data = json.load(f)

            # Get existing histories or create empty ones
            monthly_history = existing_data.get('monthly_history', {})
            daily_history = existing_data.get('daily_history', {})


            # Update current month/day costs
            monthly_history[self.current_month] = self.monthly_cost
            daily_history[self.current_date] = self.daily_cost

            # Prepare data structure with history
            data = {
                'current_month': self.current_month,
                'current_date': self.current_date,
                'monthly_history': monthly_history,
                'daily_history': daily_history,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
        except:
            pass

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> tuple[float, bool]:
        """Calculate cost for given model and token usage"""
        model_key = self.find_model_key(model)
        found_model = model_key is not None

        if not model_key:
            # Unknown models default to 0 cost (likely local models)
            costs = {"input": 0.0, "output": 0.0}
        else:
            costs = self.model_costs[model_key]

        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]

        return input_cost + output_cost, found_model

    def get_base_model_id(self, model: str) -> str:
        """Extract base model ID from Open WebUI Model if it's a character"""
        try:
            # Try to access Open WebUI Models database
            from open_webui.models.models import Models

            model_obj = Models.get_model_by_id(model)
            if model_obj and model_obj.base_model_id:
                return model_obj.base_model_id
        except Exception:
            # If we can't access the database, return the original model
            pass

        return model

    def find_model_key(self, model: str) -> Optional[str]:
        """Find matching model key (case-insensitive, partial match)"""
        # First try to get the base model if this is a character
        base_model = self.get_base_model_id(model)
        base_model_lower = base_model.lower()

        # Exact match first
        for key in self.model_costs:
            if key.lower() == base_model_lower:
                return key

        # Partial match
        for key in self.model_costs:
            if key.lower() in base_model_lower or base_model_lower in key.lower():
                return key

        return None

    def track_usage(self, model: str, input_tokens: int, output_tokens: int) -> str:
        """Track usage and return formatted cost message"""
        message_cost, found_model = self.calculate_cost(model, input_tokens, output_tokens)

        # Load current values from file to respect manual edits
        current_monthly, current_daily = self.load_costs()

        # Add message cost to current file values (not memory values)
        self.monthly_cost = current_monthly + message_cost
        self.daily_cost = current_daily + message_cost

        self.save_costs()

        total_tokens = input_tokens + output_tokens

        # Show different message based on whether model was found
        if found_model:
            return f"{message_cost:.4f} € for this message, {self.daily_cost:.4f} € today, {self.monthly_cost:.4f} € this month, {total_tokens} tokens used"
        else:
            return f"{message_cost:.4f} € for this message, {self.daily_cost:.4f} € today, {self.monthly_cost:.4f} € this month, {total_tokens} tokens used"

# Open WebUI Filter Implementation
class Filter:
    class Valves(BaseModel):
        priority: int = 0

        # Model costs JSON - easily add/remove models and update prices
        model_costs_json: str = Field(
            default="""{
    "openai.gpt-4.1": {"input": 2.0, "output": 8.0},
    "openai.gpt-4.1-mini": {"input": 0.40, "output": 1.6},
    "openai.gpt-4.1-nano": {"input": 0.1, "output": 0.4},
    "openai.gpt-4o-2024-11-20": {"input": 2.5, "output": 10.0},
    "openai.gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "openai.gpt-o3": {"input": 2.00, "output": 8.00},
    "anthropic.claude-3-5-haiku-latest": {"input": 0.80, "output": 4.0},
    "anthropic.claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    "anthropic.claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "google.gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "google.gemini-2.5-pro": {"input": 1.25, "output": 10.00}
}""",
            description="Model costs per 1M tokens (EUR) in JSON format. Add/remove models as needed. IMPORTANT: Use the exact model name as shown in Settings > Models. If you use connection prefixes (like 'openai.', 'anthropic.', 'google.'), include the full prefixed name as the key (e.g., 'openai.gpt-4o-mini' not just 'gpt-4o-mini')."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.tracker = None  # Will be initialized on first use

    def get_tracker(self):
        """Get tracker instance with current valve settings"""
        if self.tracker is None:
            self.tracker = SimpleCostTracker(self.valves.model_costs_json)
        return self.tracker

    async def get_daily_usage_summary(self) -> str:
        """Get daily usage summary from all configured APIs"""
        api_keys = {
            "openai": self.valves.openai_api_key,
            "anthropic": self.valves.anthropic_api_key,
            "gemini": self.valves.gemini_api_key,
            "openrouter": self.valves.openrouter_api_key
        }

        # Filter out empty API keys
        api_keys = {k: v for k, v in api_keys.items() if v.strip()}

        if not api_keys:
            return ""

        try:
            async with APIUsageFetcher(api_keys) as fetcher:
                usage_data = await fetcher.fetch_all_usage()

                if "error" in usage_data:
                    return f"\n\n📊 Usage fetch error: {usage_data['error']}"

                summary_lines = [f"\n\n📊 Daily Usage Summary ({usage_data['date']}):"]

                for provider_data in usage_data["providers"]:
                    provider = provider_data.get("provider", "Unknown")
                    if "error" in provider_data:
                        summary_lines.append(f"• {provider}: {provider_data['error']}")
                    elif "note" in provider_data:
                        summary_lines.append(f"• {provider}: {provider_data['note']}")
                    else:
                        summary_lines.append(f"• {provider}: Data available")

                if usage_data["errors"]:
                    summary_lines.append("Errors:")
                    for error in usage_data["errors"]:
                        summary_lines.append(f"• {error}")

                return "\n".join(summary_lines)
        except Exception as e:
            return f"\n\n📊 Usage summary error: {str(e)}"

    def count_tokens_exact(self, text: str) -> int:
        """Count tokens exactly using tiktoken"""
        try:
            # Use cl100k_base encoding for most models (GPT-4, GPT-3.5, etc.)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            # Fallback to rough estimation if tiktoken fails
            return len(text) // 4

    async def inlet(self, body: dict, __user__: Optional[dict] = None, __event_emitter__ = None) -> dict:
        """
        Capture the full request being sent to the LLM
        """
        # Store the full input for cost calculation
        if not hasattr(self, '_requests'):
            self._requests = {}

        # Calculate input tokens from the complete request (includes memories, system prompt, etc.)
        all_messages = body.get("messages", [])
        total_input_text = ""

        for msg in all_messages:
            content = msg.get("content", "")
            total_input_text += content + " "

        model = body.get("model", "unknown")
        input_tokens = self.count_tokens_exact(total_input_text)

        # Store for later use in outlet - use multiple keys for reliability
        chat_id = body.get("chat_id", "unknown")
        session_id = body.get("session_id", "unknown")
        message_id = body.get("id", "unknown")

        # Store with multiple possible keys
        for key in [chat_id, session_id, message_id, "last_request"]:
            if key != "unknown":
                self._requests[key] = {
                    "input_tokens": input_tokens,
                    "model": model
                }

        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"Processing {input_tokens} input tokens with {model}...",
                    "done": False
                }
            })

        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None, __event_emitter__ = None) -> dict:
        """
        Process output after model response to track costs
        """
        try:
            # Extract model information
            model = body.get("model", "unknown")
            chat_id = body.get("chat_id", "unknown")

            # Always use our exact token counting as it includes all context (memories, system prompt, etc.)
            input_tokens = 0
            output_tokens = 0

            # Get input tokens from inlet method - try multiple keys
            session_id = body.get("session_id", "unknown")
            message_id = body.get("id", "unknown")

            if hasattr(self, '_requests'):
                for key in [chat_id, session_id, message_id, "last_request"]:
                    if key in self._requests:
                        input_tokens = self._requests[key]["input_tokens"]
                        # Clean up stored data
                        del self._requests[key]
                        break

            # Count output tokens exactly from the last assistant message
            messages = body.get("messages", [])
            if messages:
                for msg in reversed(messages):
                    if msg.get("role") == "assistant":
                        output_tokens = self.count_tokens_exact(msg.get("content", ""))
                        break

            # Debug: Add logging to see what's happening
            if __event_emitter__:
                available_chats = list(self._requests.keys()) if hasattr(self, '_requests') else []
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"DEBUG: inlet_tokens={input_tokens}, output_tokens={output_tokens}, chat_id={chat_id}, available_chats={available_chats}",
                        "done": False
                    }
                })

            # If we still don't have tokens, fall back to API data
            if input_tokens == 0 and output_tokens == 0:
                usage = body.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

            # Create status message
            if input_tokens == 0 and output_tokens == 0:
                status_message = f"No token usage data available for model: {model}"
            else:
                # Calculate cost and create message
                tracker = self.get_tracker()
                cost_message = tracker.track_usage(model, input_tokens, output_tokens)
                status_message = cost_message

            # Emit status using event emitter
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": status_message,
                        "done": True
                    }
                })

        except Exception as e:
            # Show error via event emitter
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Cost tracking error: {str(e)}",
                        "done": True
                    }
                })

        return body


