"""
title: Simple Cost Tracker
author: Roni Laukkarinen
description: A minimalist cost tracking function that tracks token usage and costs per model.
repository_url: https://github.com/ronilaukkarinen/open-webui-simple-cost-tracker
version: 1.1.0
required_open_webui_version: >= 0.5.0
"""

import json
import os
import asyncio
import aiohttp
from datetime import datetime, date, timezone
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import tiktoken

class OpenAIAPIFetcher:
    """Fetch OpenAI cost data directly from API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"

    async def fetch_organization_costs(self, start_time: int, limit: int = 100) -> dict:
        """Fetch organization costs from OpenAI API"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                url = f"{self.base_url}/organization/costs"
                params = {
                    "start_time": start_time,
                    "limit": limit
                }

                print(f"SIMPLE_COST_TRACKER DEBUG: API URL: {url}")
                print(f"SIMPLE_COST_TRACKER DEBUG: API params: {params}")

                async with session.get(url, headers=headers, params=params) as response:
                    print(f"SIMPLE_COST_TRACKER DEBUG: API response status: {response.status}")
                    if response.status == 200:
                        result = await response.json()
                        print(f"SIMPLE_COST_TRACKER DEBUG: API response JSON: {result}")
                        return result
                    else:
                        error_text = await response.text()
                        print(f"SIMPLE_COST_TRACKER DEBUG: API error text: {error_text}")
                        return {"error": f"HTTP {response.status}: {error_text}"}
        except Exception as e:
            print(f"SIMPLE_COST_TRACKER DEBUG: API exception: {str(e)}")
            return {"error": f"API request failed: {str(e)}"}

    async def get_daily_costs(self) -> dict:
        """Get today's costs from OpenAI API"""
        # Get start of today in Unix timestamp (OpenAI uses UTC)
        from datetime import timezone
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = int(today.timestamp())

        result = await self.fetch_organization_costs(start_time, 1)

        if "error" in result:
            return result

        # Extract cost data from OpenAI API response structure
        data = result.get("data", [])
        if not data:
            return {"cost": 0.0, "date": today.strftime("%Y-%m-%d")}

        # Sum up costs for today from the nested structure
        total_cost = 0.0
        for bucket in data:
            results = bucket.get("results", [])
            for result_item in results:
                amount = result_item.get("amount", {})
                cost_value = amount.get("value", 0.0)
                total_cost += cost_value

        return {
            "cost": total_cost,
            "date": today.strftime("%Y-%m-%d"),
            "currency": "USD"
        }

    async def get_monthly_costs(self) -> dict:
        """Get monthly costs from OpenAI API"""
        # Get start of current month in Unix timestamp (OpenAI uses UTC)
        from datetime import timezone
        today = datetime.now(timezone.utc)
        month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start_time = int(month_start.timestamp())

        result = await self.fetch_organization_costs(start_time, 100)  # Get more records for the month

        if "error" in result:
            return result

        # Extract cost data from OpenAI API response structure
        data = result.get("data", [])
        if not data:
            return {"cost": 0.0, "month": today.strftime("%Y-%m")}

        # Sum up costs for the month from the nested structure
        total_cost = 0.0
        for bucket in data:
            results = bucket.get("results", [])
            for result_item in results:
                amount = result_item.get("amount", {})
                cost_value = amount.get("value", 0.0)
                total_cost += cost_value

        return {
            "cost": total_cost,
            "month": today.strftime("%Y-%m"),
            "currency": "USD"
        }


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
        # Use UTC for consistency with OpenAI API
        utc_now = datetime.now(timezone.utc)
        self.current_month = utc_now.strftime("%Y-%m")
        self.current_date = utc_now.strftime("%Y-%m-%d")
        self.monthly_cost, self.daily_cost = self.load_costs()
        self.monthly_provider_costs, self.daily_provider_costs = self.load_provider_costs()

    def load_costs(self) -> tuple[float, float]:
        """Load monthly and daily costs from storage file"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)

                    # Always use current date/month when loading (UTC for consistency)
                    utc_now = datetime.now(timezone.utc)
                    current_month = utc_now.strftime("%Y-%m")
                    current_date = utc_now.strftime("%Y-%m-%d")

                    # Get current month and day costs from history
                    monthly_history = data.get('monthly_history', {})
                    daily_history = data.get('daily_history', {})

                    monthly_data = monthly_history.get(current_month, {})
                    daily_data = daily_history.get(current_date, {})

                    # Sum all providers for total cost
                    monthly_cost = sum(v for k, v in monthly_data.items() if k != 'total')
                    daily_cost = sum(v for k, v in daily_data.items() if k != 'total')

                    return monthly_cost, daily_cost
        except:
            pass
        return 0.0, 0.0

    def load_provider_costs(self) -> tuple[Dict[str, float], Dict[str, float]]:
        """Load provider-specific monthly and daily costs from storage file"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)

                    # Always use current date/month when loading (UTC for consistency)
                    utc_now = datetime.now(timezone.utc)
                    current_month = utc_now.strftime("%Y-%m")
                    current_date = utc_now.strftime("%Y-%m-%d")

                    # Get provider-specific costs from simplified history
                    monthly_providers = data.get('monthly_history', {}).get(current_month, {})
                    daily_providers = data.get('daily_history', {}).get(current_date, {})

                    # Remove 'total' key if it exists
                    monthly_providers = {k: v for k, v in monthly_providers.items() if k != 'total'}
                    daily_providers = {k: v for k, v in daily_providers.items() if k != 'total'}

                    return monthly_providers, daily_providers
        except:
            pass
        return {}, {}

    def get_combined_totals(self, openai_api_costs=None) -> tuple[float, float]:
        """Get combined totals from JSON (now includes API costs when available)"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)

                    # Always use current date/month when loading (UTC for consistency)
                    utc_now = datetime.now(timezone.utc)
                    current_month = utc_now.strftime("%Y-%m")
                    current_date = utc_now.strftime("%Y-%m-%d")

                    # Get current provider costs from simplified structure
                    monthly_providers = data.get('monthly_history', {}).get(current_month, {})
                    daily_providers = data.get('daily_history', {}).get(current_date, {})

                    print(f"SIMPLE_COST_TRACKER DEBUG: Raw provider data - monthly: {monthly_providers}")
                    print(f"SIMPLE_COST_TRACKER DEBUG: Raw provider data - daily: {daily_providers}")

                    # Calculate totals from all providers (including OpenAI API costs if stored)
                    monthly_total = sum(v for k, v in monthly_providers.items() if k != 'total')
                    daily_total = sum(v for k, v in daily_providers.items() if k != 'total')

                    print(f"SIMPLE_COST_TRACKER DEBUG: Final totals from JSON - daily: {daily_total}, monthly: {monthly_total}")
                    return monthly_total, daily_total
        except Exception as e:
            print(f"SIMPLE_COST_TRACKER DEBUG: Error in get_combined_totals: {e}")
            pass

        # Fallback to current instance values
        return self.monthly_cost, self.daily_cost



    def save_costs(self, enabled_providers=None):
        """Save monthly and daily costs to storage file with simplified structure"""
        if enabled_providers is None:
            enabled_providers = ['openai', 'anthropic', 'google']
        try:
            # Always use current date/month when saving (UTC for consistency)
            utc_now = datetime.now(timezone.utc)
            current_month = utc_now.strftime("%Y-%m")
            current_date = utc_now.strftime("%Y-%m-%d")

            # Load existing data to preserve history
            existing_data = {}
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    existing_data = json.load(f)

            # Get existing histories or create empty ones
            monthly_history = existing_data.get('monthly_history', {})
            daily_history = existing_data.get('daily_history', {})

            # Ensure current periods exist
            if current_month not in monthly_history:
                monthly_history[current_month] = {}
            if current_date not in daily_history:
                daily_history[current_date] = {}

            # Ensure all enabled providers exist (even if 0)
            for provider in enabled_providers:
                if provider not in monthly_history[current_month]:
                    monthly_history[current_month][provider] = 0.0
                if provider not in daily_history[current_date]:
                    daily_history[current_date][provider] = 0.0

            # Update only the specific providers that were used
            if not hasattr(self, 'monthly_provider_costs'):
                self.monthly_provider_costs = {}
            if not hasattr(self, 'daily_provider_costs'):
                self.daily_provider_costs = {}

            for provider, cost in self.monthly_provider_costs.items():
                monthly_history[current_month][provider] = cost
            for provider, cost in self.daily_provider_costs.items():
                daily_history[current_date][provider] = cost

            # Calculate totals for each period
            monthly_total = sum(v for k, v in monthly_history[current_month].items() if k != 'total')
            daily_total = sum(v for k, v in daily_history[current_date].items() if k != 'total')

            # Add totals to history
            monthly_history[current_month]['total'] = monthly_total
            daily_history[current_date]['total'] = daily_total

            # Prepare simplified data structure
            data = {
                'current_month': current_month,
                'current_date': current_date,
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
        print(f"SIMPLE_COST_TRACKER DEBUG: calculate_cost called with model: {model}, input: {input_tokens}, output: {output_tokens}")

        model_key = self.find_model_key(model)
        found_model = model_key is not None

        print(f"SIMPLE_COST_TRACKER DEBUG: Model key found: {model_key}, found_model: {found_model}")

        if not model_key:
            # Unknown models default to 0 cost (likely local models)
            costs = {"input": 0.0, "output": 0.0}
            print(f"SIMPLE_COST_TRACKER DEBUG: Using default costs: {costs}")
        else:
            costs = self.model_costs[model_key]
            print(f"SIMPLE_COST_TRACKER DEBUG: Using model costs: {costs}")

        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        total_cost = input_cost + output_cost

        print(f"SIMPLE_COST_TRACKER DEBUG: Calculated costs - input: {input_cost}, output: {output_cost}, total: {total_cost}")

        return total_cost, found_model

    def get_base_model_id(self, model: str) -> str:
        """Extract base model ID from Open WebUI Model if it's a character"""
        print(f"SIMPLE_COST_TRACKER DEBUG: Getting base model for: {model}")

        try:
            # Try to access Open WebUI Models database
            from open_webui.models.models import Models

            model_obj = Models.get_model_by_id(model)
            print(f"SIMPLE_COST_TRACKER DEBUG: Model object: {model_obj}")

            if model_obj and hasattr(model_obj, 'base_model_id') and model_obj.base_model_id:
                print(f"SIMPLE_COST_TRACKER DEBUG: Found base model: {model_obj.base_model_id}")
                return model_obj.base_model_id
            else:
                print(f"SIMPLE_COST_TRACKER DEBUG: No base model found, using original: {model}")
        except Exception as e:
            print(f"SIMPLE_COST_TRACKER DEBUG: Error getting base model: {e}")
            # If we can't access the database, return the original model
            pass

        return model

    def find_model_key(self, model: str) -> Optional[str]:
        """Find matching model key (case-insensitive, partial match)"""
        print(f"SIMPLE_COST_TRACKER DEBUG: Finding model key for: {model}")

        # First try to get the base model if this is a character
        base_model = self.get_base_model_id(model)
        base_model_lower = base_model.lower()

        print(f"SIMPLE_COST_TRACKER DEBUG: Base model: {base_model}, lowercase: {base_model_lower}")
        print(f"SIMPLE_COST_TRACKER DEBUG: Available model costs keys: {list(self.model_costs.keys())}")

        # Exact match first
        for key in self.model_costs:
            if key.lower() == base_model_lower:
                print(f"SIMPLE_COST_TRACKER DEBUG: Exact match found: {key}")
                return key

        # Partial match
        for key in self.model_costs:
            if key.lower() in base_model_lower or base_model_lower in key.lower():
                print(f"SIMPLE_COST_TRACKER DEBUG: Partial match found: {key}")
                return key

        print(f"SIMPLE_COST_TRACKER DEBUG: No match found for: {base_model}")
        return None

    def get_provider_from_model(self, model: str) -> str:
        """Extract provider name from model string, using base model if needed"""
        print(f"SIMPLE_COST_TRACKER DEBUG: get_provider_from_model called with: {model}")
        # First try to get the base model if this is a character
        base_model = self.get_base_model_id(model)
        print(f"SIMPLE_COST_TRACKER DEBUG: Base model for provider detection: {base_model}")
        model_lower = base_model.lower()
        if model_lower.startswith('openai.'):
            return 'openai'
        elif model_lower.startswith('anthropic.'):
            return 'anthropic'
        elif model_lower.startswith('google.'):
            return 'google'
        elif model_lower.startswith('openrouter.'):
            return 'openrouter'
        else:
            # For models without prefix, try to infer from model name
            if 'gpt' in model_lower or 'openai' in model_lower:
                return 'openai'
            elif 'claude' in model_lower or 'anthropic' in model_lower:
                return 'anthropic'
            elif 'gemini' in model_lower or 'google' in model_lower:
                return 'google'
            elif 'openrouter' in model_lower:
                return 'openrouter'
            else:
                print(f"SIMPLE_COST_TRACKER DEBUG: No provider found for model: {model} (base: {base_model})")
                return None  # For truly unknown models, skip tracking

    def track_usage(self, model: str, input_tokens: int, output_tokens: int, skip_unknown: bool = True, openai_api_costs: Optional[dict] = None, enabled_providers: Optional[list] = None, tokens_only: bool = False, valves=None) -> Optional[str]:
        """Track usage and return formatted cost message"""
        print(f"SIMPLE_COST_TRACKER DEBUG: track_usage called with model: {model}, input: {input_tokens}, output: {output_tokens}, skip_unknown: {skip_unknown}")

        message_cost, found_model = self.calculate_cost(model, input_tokens, output_tokens)

        print(f"SIMPLE_COST_TRACKER DEBUG: track_usage - message_cost: {message_cost}, found_model: {found_model}")

        # Handle unknown models based on valves
        if not found_model:
            # If tokens_only is enabled, return tokens-only message (overrides skip_unknown)
            if tokens_only:
                total_tokens = input_tokens + output_tokens
                print(f"SIMPLE_COST_TRACKER DEBUG: Tokens-only mode for unknown model: {model}")
                return f"{total_tokens} tokens used ({input_tokens} in, {output_tokens} out)"

            # If skip_unknown is enabled, return None (no tracking)
            if skip_unknown:
                print(f"SIMPLE_COST_TRACKER DEBUG: Skipping unknown model: {model}")
                return None

        # Update provider-specific costs
        provider = self.get_provider_from_model(model)
        if provider is None:
            # Skip tracking for unknown models without provider
            return None

        # Load current values from file to respect manual edits
        current_monthly, current_daily = self.load_costs()
        current_monthly_providers, current_daily_providers = self.load_provider_costs()

        self.monthly_provider_costs = current_monthly_providers.copy()
        self.daily_provider_costs = current_daily_providers.copy()

        # For OpenAI, use API costs if available, otherwise use manual tracking
        if provider == 'openai' and openai_api_costs:
            # Use API costs directly for OpenAI (replace, don't add to existing)
            self.monthly_provider_costs['openai'] = openai_api_costs.get('monthly_cost', 0.0)
            self.daily_provider_costs['openai'] = openai_api_costs.get('daily_cost', 0.0)
            print(f"SIMPLE_COST_TRACKER DEBUG: Using OpenAI API costs - Monthly: {self.monthly_provider_costs['openai']}, Daily: {self.daily_provider_costs['openai']}")
        else:
            # Manual tracking for all other providers or OpenAI without API
            self.monthly_provider_costs[provider] = self.monthly_provider_costs.get(provider, 0.0) + message_cost
            self.daily_provider_costs[provider] = self.daily_provider_costs.get(provider, 0.0) + message_cost
            print(f"SIMPLE_COST_TRACKER DEBUG: Using manual tracking for {provider} - added {message_cost}")

        # Calculate totals from all provider costs (mix of API and manual)
        self.monthly_cost = sum(self.monthly_provider_costs.values())
        self.daily_cost = sum(self.daily_provider_costs.values())

        self.save_costs(enabled_providers)

        total_tokens = input_tokens + output_tokens

        # Get combined totals for display (manual + OpenAI API if available)
        provider = self.get_provider_from_model(model)
        print(f"SIMPLE_COST_TRACKER DEBUG: track_usage - openai_api_costs: {openai_api_costs}, provider: {provider}")

        # Get combined totals (manual tracking for all providers + OpenAI API if available)
        display_monthly_cost, display_daily_cost = self.get_combined_totals(openai_api_costs)

        print(f"SIMPLE_COST_TRACKER DEBUG: Combined totals - daily: {display_daily_cost}, monthly: {display_monthly_cost}")

        # Show different message based on whether model was found
        if found_model:
            # Build token details string based on valve setting
            if valves and hasattr(valves, 'show_token_details') and valves.show_token_details:
                token_details = f"{total_tokens} tokens used ({input_tokens} in, {output_tokens} out)"
            else:
                token_details = f"{total_tokens} tokens used"

            # Always show per-message cost regardless of API or manual tracking
            return f"{message_cost:.4f} € this message, {display_daily_cost:.2f} € today, {display_monthly_cost:.2f} € this month. {token_details}"
        else:
            # Unknown model
            if valves and hasattr(valves, 'show_token_details') and valves.show_token_details:
                token_details = f"{total_tokens} tokens used ({input_tokens} in, {output_tokens} out)"
            else:
                token_details = f"{total_tokens} tokens used"
            return f"{message_cost:.4f} € this message, {display_daily_cost:.2f} € today, {display_monthly_cost:.2f} € this month. {token_details}"

# Open WebUI Filter Implementation
class Filter:
    class Valves(BaseModel):
        priority: int = 1000  # Much higher priority to run after memory system

        # Valve to ignore memory prompt calculation for specific custom models
        ignore_memory_for_custom_models: str = Field(
            default="english-refiner,task-manager,custom-,character-",
            description="Comma-separated list of custom model names/patterns to ignore memory prompt calculation for (e.g., 'english-refiner,task-manager,custom-') - prevents inflated costs"
        )

        # Valve to completely exclude cost tracking for specific custom models
        exclude_cost_tracking_for_models: str = Field(
            default="",
            description="Comma-separated list of model names/patterns to completely exclude from cost tracking (e.g., 'my-custom-model,another-model'). No cost tracking will be performed for these models."
        )

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

        skip_unknown_models: bool = Field(
            default=True,
            description="Skip cost tracking for models not found in the costs JSON (typically local models with no cost). When enabled, unknown models won't show any cost tracking messages."
        )

        tokens_only_for_unknown: bool = Field(
            default=False,
            description="For unknown models, only show token counts without cost calculation. When enabled, unknown models will display input/output tokens but no cost information."
        )

        enable_debug: bool = Field(
            default=False,
            description="Show debug messages with token counts and chat IDs for troubleshooting."
        )

        openai_admin_key: str = Field(
            default="",
            description="OpenAI Admin Key for fetching real-time cost data. Get it from https://platform.openai.com/settings/organization/admin-keys. Leave empty to use only manual tracking."
        )

        fetch_openai_costs: bool = Field(
            default=False,
            description="Enable fetching real-time OpenAI costs from API. Requires OpenAI Admin Key."
        )

        show_token_details: bool = Field(
            default=True,
            description="Show detailed input/output token breakdown in the cost message. When disabled, only shows total tokens."
        )

        enable_openai_tracking: bool = Field(
            default=True,
            description="Enable OpenAI cost tracking. Disable if you don't use OpenAI models."
        )

        enable_anthropic_tracking: bool = Field(
            default=True,
            description="Enable Anthropic (Claude) cost tracking. Disable if you don't use Anthropic models."
        )

        enable_google_tracking: bool = Field(
            default=True,
            description="Enable Google (Gemini) cost tracking. Disable if you don't use Google models."
        )

        enable_openrouter_tracking: bool = Field(
            default=False,
            description="Enable OpenRouter cost tracking. Disable if you don't use OpenRouter models."
        )

        enable_processing_emitter: bool = Field(
            default=False,
            description="Enable processing status emitter that shows input tokens and model while processing. Disable if it gets stuck with local models."
        )

        disable_for_image_generation: bool = Field(
            default=True,
            description="Disable cost tracking for image generation requests (when Image button is clicked). Prevents interference with image generation prompts."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.tracker = None  # Will be initialized on first use

    def get_tracker(self):
        """Get tracker instance with current valve settings"""
        if self.tracker is None:
            self.tracker = SimpleCostTracker(self.valves.model_costs_json)
        return self.tracker

    def get_enabled_providers(self):
        """Get list of enabled providers based on valve settings"""
        enabled_providers = []
        if self.valves.enable_openai_tracking:
            enabled_providers.append('openai')
        if self.valves.enable_anthropic_tracking:
            enabled_providers.append('anthropic')
        if self.valves.enable_google_tracking:
            enabled_providers.append('google')
        if self.valves.enable_openrouter_tracking:
            enabled_providers.append('openrouter')
        return enabled_providers

    def get_provider_from_model(self, model: str) -> str:
        """Extract provider name from model string"""
        model_lower = model.lower()
        if model_lower.startswith('openai.'):
            return 'openai'
        elif model_lower.startswith('anthropic.'):
            return 'anthropic'
        elif model_lower.startswith('google.'):
            return 'google'
        elif model_lower.startswith('openrouter.'):
            return 'openrouter'
        else:
            # For models without prefix, try to infer from model name
            if 'gpt' in model_lower or 'openai' in model_lower:
                return 'openai'
            elif 'claude' in model_lower or 'anthropic' in model_lower:
                return 'anthropic'
            elif 'gemini' in model_lower or 'google' in model_lower:
                return 'google'
            elif 'openrouter' in model_lower:
                return 'openrouter'
            else:
                return None  # For truly unknown models, skip tracking

    async def get_openai_costs(self) -> str:
        """Get OpenAI costs from API if enabled"""
        if not self.valves.fetch_openai_costs or not self.valves.openai_admin_key.strip():
            return ""

        try:
            fetcher = OpenAIAPIFetcher(self.valves.openai_admin_key)
            result = await fetcher.get_daily_costs()

            if "error" in result:
                return f"\n\n📊 OpenAI API Error: {result['error']}"

            cost = result.get("cost", 0.0)
            date = result.get("date", "today")
            currency = result.get("currency", "USD")

            return f"\n\n📊 OpenAI costs for {date}: ${cost:.4f} {currency}"

        except Exception as e:
            return f"\n\n📊 OpenAI API fetch error: {str(e)}"

    def count_tokens_exact(self, text) -> int:
        """Count tokens exactly using tiktoken"""
        try:
            # Handle multimodal content (list of content objects)
            if isinstance(text, list):
                # Extract only text content for token counting
                text_content = ""
                for content in text:
                    if isinstance(content, dict) and content.get("type") == "text":
                        text_content += content.get("text", "")
                text = text_content
            
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text)
            
            # Use cl100k_base encoding for most models (GPT-4, GPT-3.5, etc.)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            # Fallback to rough estimation if tiktoken fails
            if isinstance(text, str):
                return len(text) // 4
            else:
                return 0

    def calculate_complete_tokens(self, body: dict) -> int:
        """Calculate tokens from the complete request including all system prompts"""
        print(f"SIMPLE_COST_TRACKER DEBUG: Request body keys: {list(body.keys())}")

        messages = body.get("messages", [])
        total_text = ""
        system_token_count = 0
        user_token_count = 0
        assistant_token_count = 0

        print(f"SIMPLE_COST_TRACKER DEBUG: Processing {len(messages)} messages")

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            content_tokens = self.count_tokens_exact(content)

            if role == "system":
                system_token_count += content_tokens
                print(f"SIMPLE_COST_TRACKER DEBUG: System message {i}: {content_tokens} tokens")
                if content_tokens > 100:  # Only show preview for long messages
                    # Handle multimodal content for preview
                    if isinstance(content, list):
                        # Extract text content for preview
                        text_content = ""
                        for content_item in content:
                            if isinstance(content_item, dict) and content_item.get("type") == "text":
                                text_content += content_item.get("text", "")
                        preview_text = text_content[:200]
                    else:
                        preview_text = str(content)[:200]
                    print(f"SIMPLE_COST_TRACKER DEBUG: System content preview: {preview_text}...")
            elif role == "user":
                user_token_count += content_tokens
                print(f"SIMPLE_COST_TRACKER DEBUG: User message {i}: {content_tokens} tokens")
            elif role == "assistant":
                assistant_token_count += content_tokens
                print(f"SIMPLE_COST_TRACKER DEBUG: Assistant message {i}: {content_tokens} tokens")
            else:
                print(f"SIMPLE_COST_TRACKER DEBUG: Unknown role '{role}' message {i}: {content_tokens} tokens")

            # Handle multimodal content for total_text accumulation
            if isinstance(content, list):
                # Extract only text content for total_text
                text_content = ""
                for content_item in content:
                    if isinstance(content_item, dict) and content_item.get("type") == "text":
                        text_content += content_item.get("text", "")
                total_text += text_content + " "
            else:
                total_text += str(content) + " "

        # Check for additional system/prompt fields
        for key in body.keys():
            if key != "messages" and isinstance(body[key], str):
                if "prompt" in key.lower() or "system" in key.lower():
                    additional_tokens = self.count_tokens_exact(body[key])
                    system_token_count += additional_tokens
                    print(f"SIMPLE_COST_TRACKER DEBUG: Additional {key}: {additional_tokens} tokens")
                    total_text += body[key] + " "

        total_tokens = self.count_tokens_exact(total_text)
        print(f"SIMPLE_COST_TRACKER DEBUG: Token breakdown - system: {system_token_count}, user: {user_token_count}, assistant: {assistant_token_count}")
        print(f"SIMPLE_COST_TRACKER DEBUG: Total tokens: {total_tokens}")
        print(f"SIMPLE_COST_TRACKER DEBUG: Total text length: {len(total_text)}")

        return total_tokens

    def should_ignore_memory_calculation(self, model: str) -> bool:
        """Check if model should ignore memory prompt calculation (custom models from valve)"""
        if not model:
            return False

        # Get the comma-separated list from valve
        custom_models_list = self.valves.ignore_memory_for_custom_models.strip()
        print(f"SIMPLE_COST_TRACKER DEBUG: Valve patterns: '{custom_models_list}'")
        if not custom_models_list:
            return False

        # Parse the comma-separated list
        custom_model_patterns = [pattern.strip() for pattern in custom_models_list.split(',') if pattern.strip()]
        print(f"SIMPLE_COST_TRACKER DEBUG: Parsed patterns: {custom_model_patterns}")

        # Check if the model matches any pattern in the list
        for pattern in custom_model_patterns:
            print(f"SIMPLE_COST_TRACKER DEBUG: Checking if '{pattern.lower()}' is in '{model.lower()}'")
            if pattern.lower() in model.lower():
                print(f"SIMPLE_COST_TRACKER DEBUG: Ignoring memory calculation for custom model: {model} (matched pattern: {pattern})")
                return True

        print(f"SIMPLE_COST_TRACKER DEBUG: No pattern matched for model: {model}")
        return False

    def should_exclude_cost_tracking(self, model: str) -> bool:
        """Check if model should be completely excluded from cost tracking"""
        if not model:
            return False

        # Get the comma-separated list from valve
        excluded_models_list = self.valves.exclude_cost_tracking_for_models.strip()
        print(f"SIMPLE_COST_TRACKER DEBUG: Excluded models patterns: '{excluded_models_list}'")
        if not excluded_models_list:
            return False

        # Parse the comma-separated list
        excluded_model_patterns = [pattern.strip() for pattern in excluded_models_list.split(',') if pattern.strip()]
        print(f"SIMPLE_COST_TRACKER DEBUG: Parsed excluded patterns: {excluded_model_patterns}")

        # Check if the model matches any pattern in the list
        for pattern in excluded_model_patterns:
            print(f"SIMPLE_COST_TRACKER DEBUG: Checking if '{pattern.lower()}' is in '{model.lower()}'")
            if pattern.lower() in model.lower():
                print(f"SIMPLE_COST_TRACKER DEBUG: Excluding cost tracking for model: {model} (matched pattern: {pattern})")
                return True

        print(f"SIMPLE_COST_TRACKER DEBUG: No exclusion pattern matched for model: {model}")
        return False

    def is_image_generation_request(self, body: dict, __request__ = None) -> bool:
        """Detect if this is an image generation request using official Open WebUI detection"""
        
        # Debug: Log the request details for analysis
        print(f"SIMPLE_COST_TRACKER DEBUG: is_image_generation_request - body keys: {list(body.keys())}")
        if __request__ and hasattr(__request__, 'url'):
            print(f"SIMPLE_COST_TRACKER DEBUG: is_image_generation_request - URL: {__request__.url}")
        
        # Method 1: Check if request URL contains image generation endpoint
        if __request__ and hasattr(__request__, 'url'):
            url_str = str(__request__.url)
            if '/api/images/generations' in url_str:
                print(f"SIMPLE_COST_TRACKER DEBUG: Detected image generation request - URL contains /api/images/generations")
                return True
        
        # Method 2: Check for image generation specific parameters in request body
        # Based on Open WebUI's GenerateImageForm parameters
        image_params = ['size', 'n', 'response_format']
        has_image_params = any(param in body for param in image_params)
        
        # Also check if this looks like an image generation request structure
        has_prompt_only = 'prompt' in body and len([k for k in body.keys() if k not in ['prompt', 'model']]) == 0
        
        if has_image_params or has_prompt_only:
            print(f"SIMPLE_COST_TRACKER DEBUG: Detected image generation request - contains image-specific parameters")
            return True
        
        # Method 3: Check for Open WebUI image generation context
        if 'prompt' in body and not 'messages' in body:
            # Image requests typically have 'prompt' but not 'messages' (which is for chat)
            print(f"SIMPLE_COST_TRACKER DEBUG: Detected image generation request - has prompt but no messages")
            return True
        
        # Method 4: Check request metadata for image generation hints
        # Look for Open WebUI specific metadata that indicates image generation
        metadata = body.get('metadata', {})
        if metadata:
            # Check if metadata contains image generation flags
            if metadata.get('image_generation') or metadata.get('generate_image'):
                print(f"SIMPLE_COST_TRACKER DEBUG: Detected image generation request - metadata contains image generation flag")
                return True
        
        # Method 5: Check features field for image generation
        features = body.get('features', {})
        if features:
            # Check if features indicates image generation
            if features.get('image_generation') or features.get('generate_image'):
                print(f"SIMPLE_COST_TRACKER DEBUG: Detected image generation request - features contains image generation flag")
                return True
        
        # Method 6: Check options field for image generation
        options = body.get('options', {})
        if options:
            # Check if options indicates image generation
            if options.get('image_generation') or options.get('generate_image'):
                print(f"SIMPLE_COST_TRACKER DEBUG: Detected image generation request - options contains image generation flag")
                return True
        
        return False

    async def inlet(self, body: dict, __user__: Optional[dict] = None, __event_emitter__ = None, __request__ = None) -> dict:
        """
        This runs AFTER memory system - capture the final request with all content
        """
        # Check if this is an image generation request and if we should skip processing
        if self.valves.disable_for_image_generation and self.is_image_generation_request(body, __request__):
            print(f"SIMPLE_COST_TRACKER DEBUG: Skipping cost tracking for image generation request")
            # Set flag to skip outlet processing as well
            self._skip_image_generation = True
            return body
        
        # Reset skip flag for regular requests
        self._skip_image_generation = False
        
        # Store the full input for cost calculation (simple global storage)
        model = body.get("model", "unknown")

        # Check if we should completely exclude cost tracking for this model
        if self.should_exclude_cost_tracking(model):
            print(f"SIMPLE_COST_TRACKER DEBUG: Completely excluding cost tracking for model: {model}")
            # Set flag to skip outlet processing as well
            self._skip_cost_tracking = True
            return body

        # Reset skip flag for regular requests
        self._skip_cost_tracking = False

        # Check if we should ignore memory calculation for this model
        print(f"SIMPLE_COST_TRACKER DEBUG: Checking model '{model}' against valve patterns")
        if self.should_ignore_memory_calculation(model):
            # For comma-separated custom models, calculate tokens without memory content
            # Only count user messages to avoid inflated costs from memory prompts
            messages = body.get("messages", [])
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            input_tokens = sum(self.count_tokens_exact(msg.get("content", "")) for msg in user_messages)
            print(f"SIMPLE_COST_TRACKER DEBUG: Using simplified token count (user only): {input_tokens}")
        else:
            # Calculate tokens from the complete request (including memory content)
            input_tokens = self.calculate_complete_tokens(body)

        # Store globally for outlet method (only one request at a time)
        self._stored_input_tokens = input_tokens
        self._stored_model = model

        # Emit processing status with token count and timeout (only if valve is enabled)
        if __event_emitter__ and self.valves.enable_processing_emitter:
            try:
                import asyncio
                await asyncio.wait_for(__event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"Processing {input_tokens} input tokens with {model}...",
                        "done": False
                    }
                }), timeout=10.0)  # 10 second timeout
            except asyncio.TimeoutError:
                print(f"SIMPLE_COST_TRACKER DEBUG: Processing emitter timed out after 10 seconds")
            except Exception as emit_error:
                print(f"SIMPLE_COST_TRACKER DEBUG: Processing emitter error: {emit_error}")

        print(f"SIMPLE_COST_TRACKER DEBUG: Inlet FINAL - stored {input_tokens} tokens globally")
        print(f"SIMPLE_COST_TRACKER DEBUG: Total messages in request: {len(body.get('messages', []))}")

        # Additional debugging to understand memory injection
        messages = body.get("messages", [])
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]

        print(f"SIMPLE_COST_TRACKER DEBUG: Message breakdown - system: {len(system_messages)}, user: {len(user_messages)}, assistant: {len(assistant_messages)}")

        # Show first few characters of each message type for debugging
        if system_messages:
            content = system_messages[0].get('content', '')
            if isinstance(content, list):
                # Extract text content for preview
                text_content = ""
                for content_item in content:
                    if isinstance(content_item, dict) and content_item.get("type") == "text":
                        text_content += content_item.get("text", "")
                preview_text = text_content[:100]
            else:
                preview_text = str(content)[:100]
            print(f"SIMPLE_COST_TRACKER DEBUG: First system message preview: {preview_text}...")
        if user_messages:
            content = user_messages[0].get('content', '')
            if isinstance(content, list):
                # Extract text content for preview
                text_content = ""
                for content_item in content:
                    if isinstance(content_item, dict) and content_item.get("type") == "text":
                        text_content += content_item.get("text", "")
                preview_text = text_content[:100]
            else:
                preview_text = str(content)[:100]
            print(f"SIMPLE_COST_TRACKER DEBUG: First user message preview: {preview_text}...")

        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None, __event_emitter__ = None) -> dict:
        """
        Process output after model response to track costs
        """
        # Skip processing if this was an image generation request
        if hasattr(self, '_skip_image_generation') and self._skip_image_generation:
            print(f"SIMPLE_COST_TRACKER DEBUG: Skipping outlet processing for image generation request")
            return body
        
        # Skip processing if cost tracking is completely excluded for this model
        if hasattr(self, '_skip_cost_tracking') and self._skip_cost_tracking:
            print(f"SIMPLE_COST_TRACKER DEBUG: Skipping outlet processing for excluded model")
            return body
        
        try:
            # Extract model information
            model = body.get("model", "unknown")

            # Get stored input tokens from inlet method (which captured the complete request with memory)
            input_tokens = 0
            if hasattr(self, '_stored_input_tokens'):
                input_tokens = self._stored_input_tokens
                print(f"SIMPLE_COST_TRACKER DEBUG: Using stored input tokens from inlet: {input_tokens}")
            else:
                print(f"SIMPLE_COST_TRACKER DEBUG: No stored input tokens found")

            # Use actual LLM usage data for output tokens
            usage = body.get("usage", {})
            output_tokens = usage.get("completion_tokens", 0)

            print(f"SIMPLE_COST_TRACKER DEBUG: LLM Usage data - input: {usage.get('prompt_tokens', 0)}, output: {output_tokens}")

            # If no output tokens from usage data, calculate from the last assistant message
            if output_tokens == 0:
                print(f"SIMPLE_COST_TRACKER DEBUG: No output tokens from usage data, calculating from messages")
                messages = body.get("messages", [])
                if messages:
                    for msg in reversed(messages):
                        if msg.get("role") == "assistant":
                            output_tokens = self.count_tokens_exact(msg.get("content", ""))
                            break

            # Debug: Add logging to see what's happening
            if self.valves.enable_debug:
                print(f"DEBUG: Final token counts - input: {input_tokens}, output: {output_tokens}, chat_id={body.get('chat_id', 'unknown')}")

            # Create status message
            if input_tokens == 0 and output_tokens == 0:
                status_message = f"No token usage data available for model: {model}"
            else:
                # Get OpenAI API costs if enabled and model is OpenAI
                openai_api_costs = None
                print(f"SIMPLE_COST_TRACKER DEBUG: Checking OpenAI API - fetch_enabled: {self.valves.fetch_openai_costs}, has_key: {bool(self.valves.openai_admin_key.strip())}")

                if self.valves.fetch_openai_costs and self.valves.openai_admin_key.strip():
                    provider = self.get_provider_from_model(model)
                    print(f"SIMPLE_COST_TRACKER DEBUG: Model: {model}, Provider: {provider}")

                    if provider == 'openai':
                        print(f"SIMPLE_COST_TRACKER DEBUG: Fetching OpenAI API costs...")
                        try:
                            fetcher = OpenAIAPIFetcher(self.valves.openai_admin_key)
                            daily_costs = await fetcher.get_daily_costs()
                            monthly_costs = await fetcher.get_monthly_costs()

                            print(f"SIMPLE_COST_TRACKER DEBUG: Raw daily API response: {daily_costs}")
                            print(f"SIMPLE_COST_TRACKER DEBUG: Raw monthly API response: {monthly_costs}")

                            openai_api_costs = {
                                "daily_cost": daily_costs.get("cost", 0.0),
                                "monthly_cost": monthly_costs.get("cost", 0.0)
                            }

                            print(f"SIMPLE_COST_TRACKER DEBUG: OpenAI API costs fetched: {openai_api_costs}")
                        except Exception as e:
                            print(f"SIMPLE_COST_TRACKER DEBUG: OpenAI API fetch failed: {e}")
                            pass  # Fallback to manual tracking if API fails

                # Calculate cost and create message
                tracker = self.get_tracker()
                enabled_providers = self.get_enabled_providers()
                print(f"SIMPLE_COST_TRACKER DEBUG: Calling track_usage with model: {model}, enabled_providers: {enabled_providers}")

                cost_message = tracker.track_usage(model, input_tokens, output_tokens, self.valves.skip_unknown_models, openai_api_costs, enabled_providers, self.valves.tokens_only_for_unknown, self.valves)

                print(f"SIMPLE_COST_TRACKER DEBUG: track_usage returned: {cost_message}")

                # If cost_message is None (unknown model and skip_unknown enabled), don't show anything
                if cost_message is None:
                    print(f"SIMPLE_COST_TRACKER DEBUG: Cost message is None, returning body without emitter")
                    return body

                status_message = cost_message
                print(f"SIMPLE_COST_TRACKER DEBUG: Final status message: {status_message}")

            # Clean up stored data
            if hasattr(self, '_stored_input_tokens'):
                del self._stored_input_tokens
            if hasattr(self, '_stored_model'):
                del self._stored_model

            # Emit final status using event emitter with timeout
            print(f"SIMPLE_COST_TRACKER DEBUG: About to emit status message: {status_message}")
            if __event_emitter__:
                print(f"SIMPLE_COST_TRACKER DEBUG: Event emitter available, emitting status")
                try:
                    import asyncio
                    await asyncio.wait_for(__event_emitter__({
                        "type": "status",
                        "data": {
                            "description": status_message,
                            "done": True
                        }
                    }), timeout=10.0)  # 10 second timeout
                except asyncio.TimeoutError:
                    print(f"SIMPLE_COST_TRACKER DEBUG: Event emitter timed out after 10 seconds")
                except Exception as emit_error:
                    print(f"SIMPLE_COST_TRACKER DEBUG: Event emitter error: {emit_error}")
            else:
                print(f"SIMPLE_COST_TRACKER DEBUG: No event emitter available")

        except Exception as e:
            # Show error via event emitter with timeout
            if __event_emitter__:
                try:
                    import asyncio
                    await asyncio.wait_for(__event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"Cost tracking error: {str(e)}",
                            "done": True
                        }
                    }), timeout=10.0)  # 10 second timeout
                except asyncio.TimeoutError:
                    print(f"SIMPLE_COST_TRACKER DEBUG: Error emitter timed out after 10 seconds")
                except Exception as emit_error:
                    print(f"SIMPLE_COST_TRACKER DEBUG: Error emitter failed: {emit_error}")

        return body
