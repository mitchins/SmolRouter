# SmolRouter

A smart, lightweight proxy for routing AI model requests with performance analytics. Perfect for local LLM enthusiasts who want intelligent routing, real-time monitoring, and seamless model switching.

## Quick Start

### Using Docker

1.  **Build the image:**
    ```bash
    docker build -t smolrouter .
    ```

2.  **Run the container:**
    ```bash
    docker run -d \
      --name smolrouter \
      --restart unless-stopped \
      -p 1234:1234 \
      -e DEFAULT_UPSTREAM="http://localhost:8000" \
      -e MODEL_MAP='{"gpt-3.5-turbo":"llama3-8b"}' \
      -v ./routes.yaml:/app/routes.yaml \
      smolrouter
    ```

### Using Python

1.  **Install SmolRouter:**
    ```bash
    pip install smolrouter
    ```

2.  **Run the application:**
    ```bash
    export DEFAULT_UPSTREAM="http://localhost:8000"
    export MODEL_MAP='{"gpt-3.5-turbo":"llama3-8b"}'
    smolrouter
    ```

### Usage

Point your applications to `http://localhost:1234` instead of the OpenAI API:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="your-api-key"  # This is passed through to the upstream server
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # This will be rewritten to "llama3-8b"
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Core Features

### Smart Routing
- **Host-based & Model-based Routing:** Route requests from specific IPs or for specific models to different upstream servers.
- **Regex & Exact Matching:** Use regex patterns (e.g., `"/.*-8b/"`) or exact model names for flexible routing.
- **Model Overrides:** Automatically change model names on-the-fly for each route.
- **YAML Configuration:** Define all routing rules in a simple, human-readable `routes.yaml` file.

### Performance Analytics & Monitoring
- **Interactive Dashboard:** A web UI to view real-time and historical request data.
- **Performance Scatter Plots:** Visualize token counts vs. response times to compare model performance.
- **Detailed Request Views:** Inspect the full request/response transcripts for any logged event.
- **SQLite Backend:** All request data is stored in a local SQLite database for persistence.

### API Compatibility & Content Processing
- **OpenAI & Ollama Support:** Acts as a drop-in replacement for both OpenAI and Ollama APIs.
- **Model Mapping:** Remap model names using a simple JSON object for legacy or alternative model support.
- **Streaming Support:** Full support for streaming responses for both API formats.
- **Content Manipulation:**
    - **Think-Chain Stripping:** Automatically remove `<think>...</think>` blocks from responses.
    - **JSON Markdown Scrubbing:** Convert markdown-fenced JSON into pure JSON.

## Configuration

### Environment Variables

| Variable                | Default                   | Description                                                              |
| ----------------------- | ------------------------- | ------------------------------------------------------------------------ |
| `DEFAULT_UPSTREAM`      | `http://localhost:8000`   | The default upstream server to use when no routing rules match.          |
| `ROUTES_CONFIG`         | `routes.yaml`             | Path to the YAML/JSON file containing smart routing rules.               |
| `MODEL_MAP`             | `{}`                      | A JSON string for simple, legacy model name remapping.                   |
| `STRIP_THINKING`        | `true`                    | If `true`, removes `<think>...</think>` blocks from responses.            |
| `STRIP_JSON_MARKDOWN`   | `false`                   | If `true`, converts markdown-fenced JSON blocks to pure JSON.            |
| `DISABLE_THINKING`      | `false`                   | If `true`, appends a `/no_think` marker to prompts to disable thinking.  |
| `ENABLE_LOGGING`        | `true`                    | If `true`, enables request logging and the web UI.                       |
| `REQUEST_TIMEOUT`       | `3000.0`                  | Timeout in seconds for upstream requests.                                |
| `DB_PATH`               | `requests.db`             | Path to the SQLite database file.                                        |
| `MAX_LOG_AGE_DAYS`      | `7`                       | Automatically delete logs older than this many days.                     |
| `LISTEN_HOST`           | `127.0.0.1`               | The host address for the application to bind to.                         |
| `LISTEN_PORT`           | `1234`                    | The port for the application to listen on.                               |

### Smart Routing (`routes.yaml`)

Create a `routes.yaml` file to define your routing logic. The first rule that matches a request is used.

```yaml
routes:
  # Route requests for small models to a specific GPU server using regex
  - match:
      model: "/.*-1.5b/"
    route:
      upstream: "http://gpu-server:8000"

  # Route requests from a specific developer's machine to a dev server
  - match:
      source_host: "10.0.1.100"
    route:
      upstream: "http://dev-server:8000"

  # Route requests for "gpt-4" and override the model name to "claude-3-opus"
  - match:
      model: "gpt-4"
    route:
      upstream: "http://claude-server:8000"
      model: "claude-3-opus"
```

## Web UI & Monitoring

The web UI provides insights into your model usage and performance.

- **Dashboard (`/`):** View the latest request logs and general statistics.
- **Performance (`/performance`):** Analyze model performance with an interactive scatter plot.
- **Request Detail (`/request/{id}`):** See the full transcript of a specific request.

## Development

### Running Tests

To run the test suite, use `pytest`:

```bash
pytest
```

### Contributing

This project is open source. Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.