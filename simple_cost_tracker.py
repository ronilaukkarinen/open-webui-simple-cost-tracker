"""
title: Simple Cost Tracker
author: Roni Laukkarinen
description: A minimalist cost tracking function that tracks token usage and costs per model.
repository_url: https://github.com/ronilaukkarinen/open-webui-simple-cost-tracker
version: 1.0.0
required_open_webui_version: >= 0.5.0
"""

import json
import os
from datetime import datetime, date
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


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
                    
                    # Check if it's a new month
                    monthly_cost = 0.0 if data.get('month') != self.current_month else data.get('monthly_cost', 0.0)
                    
                    # Check if it's a new day
                    daily_cost = 0.0 if data.get('date') != self.current_date else data.get('daily_cost', 0.0)
                    
                    return monthly_cost, daily_cost
        except:
            pass
        return 0.0, 0.0

    def save_costs(self):
        """Save monthly and daily costs to storage file"""
        try:
            data = {
                'month': self.current_month,
                'date': self.current_date,
                'monthly_cost': self.monthly_cost,
                'daily_cost': self.daily_cost,
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

    def find_model_key(self, model: str) -> Optional[str]:
        """Find matching model key (case-insensitive, partial match)"""
        model_lower = model.lower()

        # Exact match first
        for key in self.model_costs:
            if key.lower() == model_lower:
                return key

        # Partial match
        for key in self.model_costs:
            if key.lower() in model_lower or model_lower in key.lower():
                return key

        return None

    def track_usage(self, model: str, input_tokens: int, output_tokens: int) -> str:
        """Track usage and return formatted cost message"""
        message_cost, found_model = self.calculate_cost(model, input_tokens, output_tokens)
        self.monthly_cost += message_cost
        self.daily_cost += message_cost
        self.save_costs()

        total_tokens = input_tokens + output_tokens

        # Show different message based on whether model was found
        if found_model:
            return f"{self.monthly_cost:.4f} € this month, {self.daily_cost:.4f} € today, {message_cost:.4f} € for this message, {total_tokens} tokens used"
        else:
            return f"{self.monthly_cost:.4f} € this month, {self.daily_cost:.4f} € today, {message_cost:.4f} € for this message (unknown model: {model}), {total_tokens} tokens used"




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

    async def inlet(self, body: dict, __user__: Optional[dict] = None, __event_emitter__ = None) -> dict:
        """
        Capture the full request being sent to the LLM
        """
        # Store the full input for cost calculation
        if not hasattr(self, '_requests'):
            self._requests = {}
        
        # Calculate input tokens from the complete request
        all_messages = body.get("messages", [])
        total_input_text = ""
        
        for msg in all_messages:
            content = msg.get("content", "")
            total_input_text += content + " "
        
        input_tokens = len(total_input_text) // 4
        
        # Store for later use in outlet
        chat_id = body.get("chat_id", "unknown")
        self._requests[chat_id] = {
            "input_tokens": input_tokens,
            "model": body.get("model", "unknown")
        }
        
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"Processing {input_tokens} input tokens with {body.get('model', 'unknown')}...",
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
            
            # Get token data from usage field first
            usage = body.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            # If no API usage data, use our stored data from inlet + estimate output
            if input_tokens == 0 and output_tokens == 0:
                # Get input tokens from inlet method
                if hasattr(self, '_requests') and chat_id in self._requests:
                    input_tokens = self._requests[chat_id]["input_tokens"]
                    # Clean up stored data
                    del self._requests[chat_id]
                
                # Estimate output tokens from the last assistant message
                messages = body.get("messages", [])
                if messages:
                    for msg in reversed(messages):
                        if msg.get("role") == "assistant":
                            output_tokens = len(msg.get("content", "")) // 4
                            break
            
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


