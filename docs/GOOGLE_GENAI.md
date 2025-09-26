# Google GenAI Provider Setup

SmolRouter supports Google's Generative AI (Gemini) models while maintaining OpenAI API compatibility.

## Quick Start

1. **Use Google GenAI models directly:**
   ```python
   model="gemini-2.0-flash-exp"  # Automatically routes to Google GenAI
   ```

2. **Configure in `routes.yaml`:**
   ```yaml
   providers:
     - name: "google-prod"
       type: "google-genai"
       enabled: true
       api_keys:
         - "YOUR_GOOGLE_API_KEY_1"
         - "YOUR_GOOGLE_API_KEY_2"  # Multiple keys for rotation
       max_requests_per_day: 1500  # Google's free tier limit
   ```

## Available Models

Common Google GenAI models:
- `gemini-2.0-flash-exp` - Fast, experimental model
- `gemini-1.5-flash` - Production flash model
- `gemini-1.5-pro` - Advanced capabilities
- `gemini-2.0-flash-thinking-exp` - Reasoning-focused model

## Configuration Options

### Provider Configuration

```yaml
providers:
  - name: "google-main"
    type: "google-genai"
    enabled: true
    priority: 1

    # API Keys - supports multiple for automatic rotation
    api_keys:
      - "AIza..."  # Add your keys here
      - "AIza..."

    # Or load from file
    api_keys_file: "/path/to/google_api_keys.txt"  # One key per line

    # Quota management (per key, per model)
    max_requests_per_day: 1500  # Default for free tier

    # Request timeout
    timeout: 30.0
```

### Model Aliases

Map custom names to Google models:

```yaml
strategy:
  config:
    aliases:
      "flash": "gemini-2.0-flash-exp"
      "pro": "gemini-1.5-pro"
```

## API Key Management

### Multiple Key Support
- Configure multiple API keys for load balancing and quota management
- SmolRouter automatically rotates keys based on usage
- Tracks usage per (API_KEY, MODEL) combination

### Quota Tracking
- Google's quotas are **per API key, per model**
- Requests per day (RPD) quotas reset at **midnight Pacific time**
- SmolRouter tracks usage and rotates to keys with lowest usage for each model

### Invalid Key Detection
- Keys returning "permission denied" are permanently marked invalid
- Invalid keys are excluded from rotation
- View status in the Google GenAI dashboard at `/google-genai`

## Monitoring

Access the Google GenAI dashboard at `http://localhost:1234/google-genai` to view:
- Real-time quota usage per model
- API key status (available/exhausted/invalid)
- Time until quota reset (Pacific timezone)
- Request and token usage statistics

## Error Handling

SmolRouter handles these Google GenAI-specific scenarios:

1. **Quota Exhaustion (429)**: Marks key as exhausted until midnight Pacific
2. **Invalid API Key**: Permanently excludes key from rotation
3. **Temporary Errors**: Tracks error count, temporarily excludes problematic keys

## Full Example

```yaml
# routes.yaml
providers:
  - name: "google-production"
    type: "google-genai"
    enabled: true
    priority: 1
    api_keys_file: "/secrets/google_keys.txt"
    max_requests_per_day: 1500
    timeout: 30.0

strategy:
  type: "smart"
  config:
    aliases:
      # Convenience aliases
      "fast": "gemini-2.0-flash-exp"
      "smart": "gemini-1.5-pro"

      # Map OpenAI models to Google equivalents
      "gpt-3.5-turbo": "gemini-1.5-flash"
      "gpt-4": "gemini-1.5-pro"
```

## Usage Example

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="unused"  # Google API keys are configured in SmolRouter
)

# Direct Google model access
response = client.chat.completions.create(
    model="gemini-2.0-flash-exp",  # Automatically routes to Google GenAI
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=100
)

# Or use an alias
response = client.chat.completions.create(
    model="fast",  # Maps to gemini-2.0-flash-exp via alias
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## API Compatibility

SmolRouter translates between OpenAI and Google GenAI formats:

### Supported Parameters
- `messages` - Converted to Google's content format
- `temperature` - Passed through
- `max_tokens` - Maps to `max_output_tokens`
- `top_p` - Passed through

### Response Format
Google responses are converted to OpenAI format with:
- Standard message structure
- Usage statistics (tokens)
- Finish reasons

## Troubleshooting

### "All API keys exhausted"
- Check the dashboard at `/google-genai` for quota status
- Wait until midnight Pacific for quota reset
- Add more API keys to increase capacity

### "Invalid API key" errors
- Verify API key is correct and active
- Check Google Cloud Console for key status
- Invalid keys are shown in the dashboard

### Model not found
- Ensure model name starts with `gemini-` for automatic routing
- Check available models in Google AI Studio
- Verify model name spelling