### 1.1.0: 2025-07-24

* Add valve to completely exclude chosen custom models
* Fix possible syntaxerrors

### 1.0.13: 2025-07-17

* Add valve to disable function for image generation (default: true) to prevent interference with image generation prompts
* Add comprehensive Open WebUI image generation request detection (URL endpoint, metadata, features, options fields)
* Add debug logging to improve detection accuracy

### 1.0.12: 2025-07-16

* Fix multimodal content (text + images) handling in token counting
* Fix "can only concatenate list (not "str") to list" error when sending images
* Add proper type checking for message content in count_tokens_exact method
* Fix debug message previews to handle multimodal content properly
* Improve error handling for mixed content types

### 1.0.11: 2025-07-16

* Fix initial multimodal content handling in token counting method

### 1.0.10: 2025-07-15

* Add valve to control processing emitter (default: false) to prevent getting stuck with local models

### 1.0.9: 2025-07-12

* Fix timezone issue causing 0.00 € cost between 00-03 hours by switching to UTC timestamps
* Restore "processing x tokens" emitter with 10-second timeout to prevent infinite hangs
* Add timeout protection to all event emitters

### 1.0.8: 2025-07-11

* Fix OpenAI API cost tracking to use API costs exclusively when enabled
* Update display message to show tokens for information while using API costs for tracking
* Prevent double-counting of OpenAI costs between manual calculation and API fetching
* Add valve to control showing detailed token breakdown (input/output)
* Change message format to show costs first, then tokens with parentheses format
* Fix display to show total costs from all providers instead of just OpenAI API costs

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

### 1.0.3: 2025-07-24

* Add valve to completely exclude cost tracking for specific comma-separated custom models
* Add should_exclude_cost_tracking method for checking model exclusion patterns
* Modify inlet and outlet methods to skip processing for excluded models

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
