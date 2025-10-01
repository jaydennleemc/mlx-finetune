#!/bin/bash

# This script builds and runs the SwiftFinetuneGemma2b project using Swift Package Manager.

# Navigate to the project directory
cd "$(dirname "$0")/SwiftFinetuneGemma2b"

echo "Building the project..."
swift build

if [ $? -eq 0 ]; then
  echo "Build successful. Running the program..."
  swift run SwiftFinetuneGemma2b
else
  echo "Build failed."
  exit 1
fi
