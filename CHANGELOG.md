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
