import tiktoken

# Print all available encodings
print(tiktoken.list_encoding_names())
# Print the encoding for a specific model
model_name = "gpt-4o"
print(tiktoken.encoding_for_model(model_name))

# Choose the tokenizer for your model
# gpt-4 uses "cl100k_base"
# gpt-4o and gpt-4.1 use "o200k_base"
tokenizer = tiktoken.get_encoding("o200k_base")

# The tokens you want to check
tokens = [" A", " B", " C"]

# Encode tokens and get their IDs
token_ids = [tokenizer.encode(token) for token in tokens]

# Print the results
for token, token_id in zip(tokens, token_ids):
    print(f"Token: {token}, Token ID: {token_id}")
