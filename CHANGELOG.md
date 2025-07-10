### 1.0.7: 2025-07-10

* Fix cost tracker not returning today's date
* Fix unix timestamp issue

### 1.0.6: 2025-07-09

* Get real-time costs from OpenAI API
* Add categorized JSON
* Add valves for model groups
* Various bugfixes
* Fix tokens not being counted reliably

### 1.0.5: 2025-07-09

* Simplify JSON structure by removing duplicate provider_history
* Store provider costs directly in daily_history and monthly_history
* Add migrate_old_format method to handle migration from old structure
* Fix complex and messy dual structure issue
* Improve data storage efficiency and readability

### 1.0.4: 2025-07-09

* Add OpenAI Admin Key integration for real-time cost fetching
* Add OpenAI API organization costs endpoint support
* Add fetch_openai_costs valve to enable/disable API fetching
* Add OpenAI costs display alongside manual tracking

### 1.0.3: 2025-07-08

* Add skip_unknown_models valve to disable tracking for models without prices
* Improve token counting accuracy using exact tiktoken encoding
* Fix monthly/daily cost persistence with historical data storage
* Add support for Open WebUI Models (characters) base model extraction
* Improve manual edit handling to respect user modifications

### 1.0.2: 2025-07-08

* Add usage history
* Fix base model not being fetched for OpenAI Models (Characters)
* Fix manual edits being overwritten by automatic tracking

### 1.0.1: 2025-07-08

* Fix costs not being tracked correctly
* Use tiktoken for token estimation

### 1.0.0: 2025-07-08

* Add monthly cost tracking with automatic reset
* Add daily cost tracking with automatic reset
* Add per-message cost calculation
* Add token usage monitoring for input and output tokens
* Add multi-provider support for OpenAI, Anthropic, Google, OpenRouter
* Add JSON valve configuration for model costs
* Add provider prefix support (openai., anthropic., google.)
* Add persistent storage using JSON file format
* Add automatic token estimation including memories and system prompts
* Add real-time status messages via event emitter
* Add unknown model handling with 0 cost default
* Add inlet method for full input context capture
* Add outlet method for output token estimation
* Add automatic data cleanup for temporary request data
* Add error handling with graceful fallbacks
* Add standard pricing format using costs per 1M tokens (EUR)
* Add pre-configured models with current market pricing
