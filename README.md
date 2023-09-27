# promonitor
Monitoring LLMs made easy

## Overview

The `promonitor` package provides a set of helper classes and functions to facilitate monitoring of Language Learning Models (LLMs). The main components are:

- `RetrieverManagerMixin`: Mixin for Retriever callbacks.
- `LLMManagerMixin`: Mixin for LLM callbacks.
- `ChainManagerMixin`: Mixin for chain callbacks.
- `ToolManagerMixin`: Mixin for tool callbacks.
- `CallbackManagerMixin`: Mixin for callback manager.
- `RunManagerMixin`: Mixin for run manager.
- `BaseCallbackHandler`: Base callback handler that can be used to handle callbacks from langchain.
- `AsyncCallbackHandler`: Async callback handler that can be used to handle callbacks from langchain.
- `OpenAICallbackHandler`: Callback Handler that tracks OpenAI info.
- `BaseMetadataCallbackHandler`: This class handles the metadata and associated function states for callbacks.
- `StdOutCallbackHandler`: Callback Handler that prints to std out.

Each of these components has specific methods that are triggered at different stages of the LLM lifecycle. These methods can be overridden to provide custom behavior.

## Next Steps

- Add support for log debugging and chain/agent visualization.
- Add frontend for monitoring.
- Add option to save logs to file.
- Add option to save logs to database.
- Add org-level logging.
- Add support for multiple LLMs, use liteLLM to track different LLMs
- Add support for LlamaIndex
- Use GPTCache for caching prompts.
- Support prompt validation using (PGit)[https://github.com/rachittshah/pgit]
- Allow user to self host and make SOC2 compliant.

Note: this is a passion project. I wanted to work in LLMOps, but everyone wanted senior engineers. This is me building something I like!