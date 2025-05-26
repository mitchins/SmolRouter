# openai-model-rerouter

A lightweight proxy service that lets you remap OpenAI-compatible model names on the fly.

## Why

Many tools and libraries hard-code specific model IDs (e.g., `gpt-3.5-turbo-16k`), making it difficult to switch to locally hosted or alternative models without changing source code. This service sits between your client and the model endpoint, rewriting model names according to your configuration.

## What

- **Model Mapping**: Redirect any incoming model ID to one of your choice.
- **Flexible Configuration**: Define exact or regex-based mappings via environment variables or Docker settings.
- **Streaming & Non-Streaming**: Fully compatible with both chat streaming (SSE) and standard JSON completions.
- **Disable Internal Thinking**: Optionally append a `/no_think` flag to prompts to suppress internal `<think>...</think>` tokens.
- **OpenAI API Interface**: List models, create completions, chat completions—just like the official API.

## How

1. **Build or pull the Docker image**  
   ```bash
   docker build -t openai-model-rerouter .
   # or
   docker pull your-registry/openai-model-rerouter:latest
   ```

2. **Run the service**  
   ```bash
   docker run -d \
     --name openai-model-rerouter \
     --restart unless-stopped \
     -p 1234:1234 \
     -e UPSTREAM_URL="http://localhost:8000" \
     -e MODEL_MAP='{"gpt-3.5-turbo-16k":"qwen3-4b"}' \
     -e DISABLE_THINKING="true" \
     openai-model-rerouter
   ```

3. **Point your client at the proxy**  
   Use `http://<host>:1234/v1/...` exactly as you would the OpenAI API. The service will rewrite the `model` field in the payload according to your mappings.

4. **Customize**  
   - **MODEL_MAP**: JSON object mapping source IDs (or `/regex/`) to target model IDs.  
   - **DISABLE_THINKING**: Set to `true` to append `/no_think` to prompts.  
   - **LISTEN_HOST, LISTEN_PORT, UPSTREAM_URL**: Configure networking via environment variables.

Enjoy seamless model swapping without changing your code!  
