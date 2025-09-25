#!/bin/bash

set -e

echo "Testing Minimal Rasa Installation (No TensorFlow)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Creating temporary virtual environment..."
cd "$SCRIPT_DIR"
python -m venv test_minimal_env
source test_minimal_env/bin/activate

# Change to project root for installation
cd "$PROJECT_ROOT"

echo "Installing Rasa without TensorFlow extra..."
pip install --upgrade pip
pip install -e .

echo "Testing Rasa CLI..."
rasa --help > /dev/null 2>&1
echo "Rasa CLI help command successful"

echo "Testing non-TensorFlow components..."
python -c "
import rasa.nlu.classifiers.sklearn_intent_classifier
import rasa.core.policies.rule_policy
import rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer
print('Import of non-TensorFlow components successful')
"

echo "Testing components relying on TensorFlow (should fail gracefully as TensorFlow uninstalled)..."
python -c "
try:
    import rasa.nlu.classifiers.diet_classifier
    print('DIETClassifier should not be available in minimal installation (failure)')
    exit(1)
except ImportError as e:
    print('DIETClassifier correctly not available (success):', str(e))

try:
    import rasa.core.policies.ted_policy
    print('TEDPolicy should not be available in minimal installation (failure)')
    exit(1)
except ImportError as e:
    print('TEDPolicy correctly not available (success):', str(e))
"

echo "Testing TensorFlow availability..."
python -c "
try:
    import tensorflow
    print('TensorFlow should not be available in minimal installation (failure)')
    exit(1)
except ImportError:
    print('TensorFlow correctly not available (success)')
"

echo "Cleaning up..."
deactivate
cd "$PROJECT_ROOT"
rm -rf test_minimal_env

echo "Minimal installation test completed successfully"
