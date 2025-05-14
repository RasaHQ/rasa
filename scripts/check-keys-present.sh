#!/bin/bash

set -euo pipefail

# Create temporary directories
TEMP_DIR_TARGZ=$(mktemp -d)
TEMP_DIR_WHEEL=$(mktemp -d)

# Function to check for rasa/keys
check_keys() {
    local temp_dir=$1
    local archive_type=$2
    local keys_path=$3

    if [ -f "$keys_path" ]; then
        echo "✅ Found rasa/keys in $archive_type $temp_dir"
    else
        echo "❌ rasa/keys not found in $archive_type $temp_dir"
        exit 1
    fi
}

# Function to get version directory for tar.gz
get_version_dir() {
    local temp_dir=$1
    local version_dir

    version_dir=$(find "$temp_dir" -maxdepth 1 -type d -name "rasa_pro-*" | head -n 1)
    if [ -z "$version_dir" ]; then
        echo "❌ No version directory found in tar.gz $temp_dir"
        exit 1
    fi
    echo "$version_dir"
}

# Process tar.gz file
TARGZ_FILE=$(find dist/ -name "*.tar.gz")
if [ -n "$TARGZ_FILE" ]; then
    echo "Checking tar.gz file: $TARGZ_FILE"
    tar -xzf "$TARGZ_FILE" -C "$TEMP_DIR_TARGZ"
    VERSION_DIR=$(get_version_dir "$TEMP_DIR_TARGZ")
    check_keys "$TEMP_DIR_TARGZ" "tar.gz" "$VERSION_DIR/rasa/keys"
    rm -rf "$TEMP_DIR_TARGZ"
else
    echo "No tar.gz file found in dist/"
    exit 1
fi

# Process wheel file
WHEEL_FILE=$(find dist/ -name "*.whl")
if [ -n "$WHEEL_FILE" ]; then
    echo "Checking wheel file: $WHEEL_FILE"
    unzip -q "$WHEEL_FILE" -d "$TEMP_DIR_WHEEL"
    check_keys "$TEMP_DIR_WHEEL" "wheel" "$TEMP_DIR_WHEEL/rasa/keys"
    rm -rf "$TEMP_DIR_WHEEL"
else
    echo "No wheel file found in dist/"
    exit 1
fi
