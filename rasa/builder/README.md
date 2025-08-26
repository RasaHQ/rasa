# Rasa Prompt-to-Bot Service

A production-ready service that generates Rasa chatbots from natural language descriptions using LLMs.

## Architecture

The service follows functional programming principles with minimal use of classes:

### Core Modules

- **`config.py`** - Configuration management using module-level constants
- **`exceptions.py`** - Custom exception hierarchy for error handling
- **`models.py`** - Pydantic models for request/response validation
- **`llm_service.py`** - LLM interactions (minimal class for state management)
- **`validation_service.py`** - Project validation functions
- **`training_service.py`** - Model training functions
- **`project_generator.py`** - Project generation (class for bot files state)
- **`service.py`** - Main orchestrating service (class for app state)
- **`logging_utils.py`** - Thread-safe logging utilities
- **`llm_context.py`** - Conversation context formatting
- **`main.py`** - Application entry point

### Utility Scripts

- **`scrape_rasa_docs.py`** - Documentation scraping
- **`create_openai_vector_store.py`** - Documentation indexing

## Key Design Principles

1. **Functional First**: Use functions for stateless operations
2. **Minimal Classes**: Classes only when state management is needed
3. **Configuration**: Environment variables with sensible defaults
4. **Error Handling**: Structured exception hierarchy
5. **Type Safety**: Full type annotations and Pydantic validation
6. **Thread Safety**: Safe concurrent operations
7. **Resource Management**: Proper cleanup and lifecycle handling

## Usage

### Running the Service

```bash
python rasa/builder/main.py
```

### Environment Configuration

```bash
# OpenAI Settings
export OPENAI_MODEL="gpt-4.1-2025-04-14"
export OPENAI_TEMPERATURE="0.7"
export OPENAI_VECTOR_STORE_ID="vs_xxxxx"
export OPENAI_TIMEOUT="30"

# Server Settings
export SERVER_HOST="0.0.0.0"
export SERVER_PORT="5005"
export MAX_RETRIES="5"
export CORS_ORIGINS="http://localhost:3000,https://example.com"

# Validation Settings
export VALIDATION_FAIL_ON_WARNINGS="false"
```

### API Endpoints

- `POST /api/prompt-to-bot` - Generate bot from description
- `GET /api/bot-data` - Get current bot configuration
- `PUT /api/bot-data` - Update bot configuration (SSE)
- `POST /api/llm-builder` - LLM helper for bot development
- `GET /` - Health check

### Documentation Setup

```bash
# 1. Scrape Rasa documentation
python rasa/builder/scrape_rasa_docs.py

# 2. Create OpenAI vector store
python rasa/builder/create_openai_vector_store.py
```

## Benefits of Functional Approach

- **Simpler**: Easy to understand and reason about
- **Testable**: Functions are easier to unit test
- **Reusable**: Pure functions can be composed
- **Maintainable**: Clear separation of concerns
- **Performant**: No unnecessary object overhead
- **Debuggable**: Clear call stacks and data flow

## Error Handling

The service uses a structured exception hierarchy:

- `PromptToBotError` - Base exception
- `ValidationError` - Project validation failures
- `TrainingError` - Model training issues
- `LLMGenerationError` - LLM API problems
- `ProjectGenerationError` - Generation retry exhaustion
- `AgentLoadError` - Agent loading failures

## Logging

Structured logging with context using `structlog`:

```python
structlogger.info("operation.success", key="value")
structlogger.error("operation.failed", error=str(e))
```

## State Management

Only classes that truly need state:

1. **`LLMService`** - Caches schemas and manages OpenAI client
2. **`ProjectGenerator`** - Maintains current bot files
3. **`BotBuilderService`** - Manages Sanic app and agent state

Everything else uses pure functions for maximum simplicity and testability.
