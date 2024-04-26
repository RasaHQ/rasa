# Command annotation of E2E tests

This guide provides steps to automatically annotate e2e tests with commands.
It is done based on an interaction with a strong LLM like GPT-4 and taking
the resulting internal/post-processed commands as the correct annotations.

## Requirements

An assistant with e2e tests, such as the Rasa Calm Demo Assistant.

## Steps

1. **Train the model:**
   Begin by training a model on the assistant. This can be done using commands like `make train` or `rasa train`.

2. **Annotate e2e tests with commands:**
   Use the `rasa.e2e_test.rephrasing.rephrase_e2e_tests` module to generate an augmented dataset.

   Run the following command to generate data:

   ```shell
   python -m rasa.e2e_test.command_annotation.annotate_e2e_tests_with_commands
   --domain_path=<path_to_domain>
   --model_path=<path_to_model>
   --endpoints=<path_to_endpoints.ylm>
   --test_path=<path_to_e2e_tests>
   --output_path=<path_to_output>
   ```
   