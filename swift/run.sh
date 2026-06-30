#!/bin/bash

# This script builds and runs the SwiftFinetuneGemma2b project using Swift Package Manager.

# Use the swift directory as project root
cd "$(dirname "$0")"

echo "Building the project..."
swift build

if [ $? -eq 0 ]; then
  echo "Build successful. Running the program..."
  swift run SwiftFinetuneGemma2b
else
  echo "Build failed."
  exit 1
fi
