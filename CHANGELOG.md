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
