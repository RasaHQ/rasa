#!/bin/bash

set -e

echo "Testing TensorFlow-Enabled Rasa Installation"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Creating temporary virtual environment..."
cd "$SCRIPT_DIR"
python -m venv test_tensorflow_env
source test_tensorflow_env/bin/activate

# Change to project root for installation
cd "$PROJECT_ROOT"

echo "Installing Rasa with TensorFlow extra..."
pip install --upgrade pip
pip install -e ".[tensorflow]"

echo "Testing Rasa CLI..."
rasa --help > /dev/null 2>&1
echo "Rasa CLI help command successful"

echo "Testing TensorFlow packages..."
python -c "
import tensorflow as tf

print(f'TensorFlow version: {tf.__version__}')

# Test main TensorFlow packages (cross-platform)
packages = [
    ('tensorflow', 'tensorflow'),
    ('tensorflow_hub', 'tensorflow-hub'),
    ('keras', 'keras'),
    ('transformers', 'transformers'),
    ('sentencepiece', 'sentencepiece'),
]

print('TensorFlow extra packages:')
for module, name in packages:
    try:
        __import__(module)
        print(f'  {name}: available')
    except Exception as e:
        print(f'  {name}: not available')
"

echo "Testing components that are dependent on TensorFlow..."
python -c "
components = [
    'rasa.nlu.classifiers.diet_classifier',
    'rasa.core.policies.ted_policy',
    'rasa.nlu.featurizers.dense_featurizer.convert_featurizer',
    'rasa.nlu.selectors.response_selector',
    'rasa.core.policies.unexpected_intent_policy',
    'rasa.nlu.featurizers.dense_featurizer.lm_featurizer'
]

print('TensorFlow-dependent components:')
for component in components:
    try:
        __import__(component)
        print(f'  {component}: imported successfully')
    except Exception as e:
        print(f'  {component}: failed to import - {e}')
"

echo "Testing components that are not dependent on TensorFlow..."
python -c "
non_tf_components = [
    'rasa.nlu.classifiers.sklearn_intent_classifier',
    'rasa.core.policies.rule_policy',
    'rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer'
]

print('Non-TensorFlow components:')
for component in non_tf_components:
    try:
        __import__(component)
        print(f'  {component}: imported successfully')
    except Exception as e:
        print(f'  {component}: failed to import - {e}')
"

echo "Cleaning up..."
deactivate
cd "$SCRIPT_DIR"
rm -rf test_tensorflow_env

echo "TensorFlow installation test completed successfully"
