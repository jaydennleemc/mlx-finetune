# Swift Fine-tuning Solution for Gemma 2B with MLX

## Project Overview

This project implements a complete solution for fine-tuning Large Language Models using Apple's MLX framework with Swift integration. Specifically, it focuses on fine-tuning the Gemma 2B model using LoRA (Low-Rank Adaptation) for parameter-efficient training.

The solution leverages Swift's performance and safety features while maintaining compatibility with the MLX ecosystem for efficient machine learning on Apple Silicon.

### Core Components

1. **Native Swift MLX Implementation** (`/SwiftFinetuneGemma2b/`)
   - Complete LoRA (Low-Rank Adaptation) implementation in Swift
   - Swift-native model definition and training loops
   - Data processing utilities for JSONL format training data
   - Integration with `MLXLLM` and `MLXLMCommon` modules for LoRA functionality

2. **Swift Package Integration** (`/Package.swift`)
   - Integration with MLX Swift framework
   - Proper dependency management for MLX components
   - Cross-platform compatibility for Apple ecosystems

3. **Project Structure** (`/SwiftFinetuneGemma2b/`)
   - `data/` - Contains training and validation datasets in JSONL format
   - `models/` - Storage for fine-tuned models and adapters
   - `main.swift` - Complete implementation in a single Swift file

## System Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4)
- **Operating System**: macOS 14.0+ (Sonoma or later)
- **Development Environment**: Xcode 15.0+ and Swift 5.9+
- **Memory**: At least 16GB of RAM (recommended 32GB+ for larger models)

## Installation and Setup

### Prerequisites

1. Ensure you have Xcode 15.0 or later installed
2. Verify you're running macOS 14.0 or later
3. Make sure you have at least 16GB of RAM

### Setting Up the Project

1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd swift_finetune_gemma2b/SwiftFinetuneGemma2b
   ```

2. **Open in Xcode**:
   - Open Xcode
   - Select "Open a project or file"
   - Navigate to the `Package.swift` file in the project directory
   - Open the project

3. **Resolve Dependencies**:
   - Xcode will automatically resolve Swift Package dependencies
   - Ensure all MLX dependencies are successfully downloaded and built

4. **Build the Project**:
   - Select your target device (Mac)
   - Press `Cmd+B` to build the project
   - Address any build errors if they occur

### Data Preparation

Training data should be in JSONL format with `text` and `label` fields:

```json
{"text": "This is a great product!", "label": 1}
{"text": "Terrible experience.", "label": 0}
```

Place your training data in the `data/` directory:
- `data/train.jsonl` - Training dataset
- `data/valid.jsonl` - Validation dataset

## Usage Instructions

### Running the Fine-tuning Process

1. **Configure Training Parameters**:
   - Modify the default configuration in `main.swift` if needed:
     ```swift
     let defaultConfig = FineTuneConfiguration(
         modelId: "mlx-community/gemma-2b-it-4bit",
         loraRank: 8,
         loraAlpha: 16,
         batchSize: 1,
         learningRate: 1e-4 as Float,
         epochs: 3,
         trainDataPath: "./data/train.jsonl",
         valDataPath: "./data/valid.jsonl",
         savePath: "./models/fine_tuned_adapter.npz"
     )
     ```

2. **Run in Xcode**:
   - Select the appropriate run target
   - Press the "Run" button or `Cmd+R`
   - Monitor the Xcode console for training progress

### Monitoring Training Progress

During training, the console will display:
- Epoch progress and loss metrics
- Batch-level training information
- Validation metrics after each epoch
- Model saving notifications

### Using the Fine-tuned Model

After training completes, the LoRA adapter weights will be saved to:
`./models/fine_tuned_adapter.npz`

## Development Guide

### Project Structure

The project consists of a single Swift file (`main.swift`) containing all functionality:

1. **Configuration and Constants** - Defines training parameters
2. **Data Structures** - Training sample and metrics representations
3. **Data Loading and Processing** - Handles JSONL data loading and preprocessing
4. **LoRA Implementation** - Core LoRA layer implementations
5. **Model Components** - Gemma-specific model components with LoRA support
6. **Training Components** - Complete training loop implementation
7. **Main Application** - Orchestrates the fine-tuning process
8. **Xcode Integration** - Xcode-specific entry points and utilities

### Extending the Implementation

1. **Modify Model Architecture**:
   - Extend `GemmaForFineTuning` class for custom model modifications
   - Add new components in the `GemmaTransformerBlock` class

2. **Adjust Training Parameters**:
   - Modify the `FineTuneConfiguration` struct
   - Update learning rate schedules in `Trainer` class

3. **Enhance Data Processing**:
   - Improve tokenization in `DataLoader` class
   - Add data augmentation techniques

4. **Improve Loss Functions**:
   - Modify `computeLoss` methods in `Trainer` class
   - Implement custom loss functions for specific tasks

### API Documentation

Detailed API documentation is available through Swift documentation comments in the source code. In Xcode:

1. Option-click on any class or method to view documentation
2. Use the Quick Help inspector for inline documentation
3. Refer to headerdoc-style comments throughout the codebase

## Troubleshooting

### Common Issues

1. **Build Errors**:
   - Ensure all dependencies are resolved in Xcode
   - Check that you're using Xcode 15.0+ and macOS 14.0+
   - Clean and rebuild the project (`Cmd+Shift+K` then `Cmd+B`)

2. **Runtime Errors**:
   - Verify data files exist in the expected locations
   - Check file permissions for data and model directories
   - Ensure sufficient memory is available

3. **Performance Issues**:
   - Reduce batch size if experiencing memory issues
   - Monitor Activity Monitor for memory usage
   - Consider using smaller LoRA ranks for reduced memory footprint

### Getting Help

For additional support:
1. Check the Swift documentation comments in the source code
2. Review the MLX Swift documentation
3. Consult the official Gemma and MLX documentation

## Contributing

We welcome contributions to improve this implementation:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

Please ensure your code follows Swift best practices and includes appropriate documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Apple's MLX team for the excellent MLX framework
- Google for the Gemma model
- The open-source community for Swift and machine learning tools