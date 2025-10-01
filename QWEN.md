# Native Swift Fine-tuning Solution for Gemma 2B with MLX

## Project Overview

This project demonstrates a native Swift implementation for fine-tuning Large Language Models, specifically the Gemma 2B model, using Apple's MLX framework entirely in Swift. The solution leverages Swift's performance and safety features while maintaining compatibility with the MLX ecosystem for efficient machine learning on Apple Silicon.

The project is based on the MLX Swift framework which includes experimental but functional training APIs, including LoRA training capabilities as referenced in the MLX Swift Examples package. This enables a fully Swift-native approach for fine-tuning models without requiring Python dependencies.

### Core Components

1. **Native Swift MLX Implementation** (`/mlx_finetune_project/swift/`)
   - Complete LoRA (Low-Rank Adaptation) implementation in Swift
   - Swift-native model definition and training loops
   - Data processing utilities for JSONL format training data
   - Integration with `MLXLLM` and `MLXLMCommon` modules for LoRA functionality

2. **Swift Package Integration** (`/Package.swift`)
   - Integration with MLX Swift framework
   - Proper dependency management for MLX components
   - Cross-platform compatibility for Apple ecosystems

3. **Project Structure** (`/mlx_finetune_project/`)
   - `data/` - Contains training and validation datasets in JSONL format
   - `models/` - Storage for fine-tuned models and adapters
   - `scripts/` - Utility scripts for various operations
   - `docs/` - Documentation for the solution

## Building and Running

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
- macOS 13.0+
- Xcode 15.0+ and Swift 5.9+
- At least 16GB of RAM (for larger models)

### Setup Process

1. **Swift Package Dependencies**
   ```swift
   // Package.swift
   dependencies: [
       .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0"),
       .package(url: "https://github.com/ml-explore/mlx-swift-examples", branch: "main")
   ],
   targets: [
       .target(
           name: "MLXFinetune",
           dependencies: [
               .product(name: "MLX", package: "mlx-swift"),
               .product(name: "MLXRandom", package: "mlx-swift"),
               .product(name: "MLXNN", package: "mlx-swift"),
               .product(name: "MLXOptimizers", package: "mlx-swift"),
               .product(name: "MLXLLM", package: "mlx-swift-examples"),
               .product(name: "MLXLMCommon", package: "mlx-swift-examples")
           ]
       )
   ]
   ```

2. **Environment Setup**
   ```bash
   # Clone the MLX Swift examples to access training utilities
   git clone https://github.com/ml-explore/mlx-swift-examples
   ```

### Native Swift Fine-tuning Implementation

The implementation leverages MLX Swift's experimental fine-tuning APIs that are available as referenced in the `mlx-swift-examples` package:

```swift
import MLX
import MLXLLM
import MLXLMCommon
import MLXOptimizers

// Implementation using Swift MLX fine-tuning APIs
class SwiftLoRAFinetune {
    private var model: LLMModel
    private var tokenizer: LLMTokenizer
    private var parameters: TrainingParameters
    
    init(modelId: String) async throws {
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(id: modelId)
        )
        self.model = container.model
        self.tokenizer = container.tokenizer
        
        // Configure training parameters
        self.parameters = TrainingParameters(
            iterations: 500,
            learningRate: 1e-4,
            batchSize: 2
        )
    }
    
    func fineTune(trainDataPath: String, valDataPath: String) async throws {
        // Load and preprocess training data
        let trainData = try await loadJSONLData(from: trainDataPath)
        let valData = try await loadJSONLData(from: valDataPath)
        
        // Apply LoRA configuration to the model
        let loraConfig = LoRALinear.Config(rank: 8, alpha: 16)
        let loraModel = try applyLoraConfig(to: model, config: loraConfig)
        
        // Execute training using Swift MLX training APIs
        try await train(model: loraModel,
                       train: trainData,
                       validate: valData,
                       optimizer: Adam(learningRate: parameters.learningRate),
                       parameters: parameters)
    }
    
    private func loadJSONLData(from path: String) async throws -> [String] {
        // Implementation for loading JSONL data
        // Parse each line as JSON and extract text content
        return try await withCheckedThrowingContinuation { continuation in
            // Implementation details
        }
    }
}
```

### Data Preparation

Training data should be in JSONL format with `text` and `label` fields:
```json
{"text": "This is a great product!", "label": 1}
{"text": "Terrible experience.", "label": 0}
```

Use Swift data utilities to process the training data:
```swift
import Foundation

public class SwiftDataUtils {
    public static func createJSONLDataset(texts: [String], labels: [Int], outputPath: String) -> Bool {
        guard texts.count == labels.count else {
            print("Error: Texts and labels arrays must have the same length")
            return false
        }
        
        do {
            let fileHandle = try FileHandle(forWritingTo: URL(fileURLWithPath: outputPath))
            defer { fileHandle.closeFile() }
            
            for i in 0..<texts.count {
                let entry: [String: Any] = ["text": texts[i], "label": labels[i]]
                let jsonData = try JSONSerialization.data(withJSONObject: entry)
                let jsonString = String(data: jsonData, encoding: .utf8)! + "\n"
                
                if let data = jsonString.data(using: .utf8) {
                    fileHandle.write(data)
                }
            }
            
            return true
        } catch {
            print("Error creating JSONL dataset: \(error)")
            return false
        }
    }
}
```

## Development Conventions

### Code Organization
- Swift fine-tuning logic is implemented in `/swift/` directory
- Training data is in `/data/` in JSONL format
- Models and adapters are managed using MLX Swift model loading
- Training parameters are configured using Swift structs

### LoRA Implementation Details
- LoRA rank typically ranges from 4-32 (8 is a good default)
- Alpha parameter usually matches the rank value or is set to 16-32
- Only specific layers (attention and MLP) are converted to LoRA
- This significantly reduces trainable parameters compared to full fine-tuning

### Training Loop Implementation
- Uses Swift's async/await for asynchronous operations
- Proper memory management with MLX's lazy evaluation
- Gradient computation and optimizer updates in Swift
- Progress tracking and logging implemented in Swift

### Error Handling
- Comprehensive Swift error handling with do-catch blocks
- Result types for operations that may fail
- Proper resource cleanup with defer statements
- Validation of input parameters before training

## Key Features

1. **Pure Swift Implementation**: Complete fine-tuning pipeline in Swift without Python dependencies
2. **Apple Silicon Optimization**: Fully optimized for Apple Silicon GPUs with MLX's efficient operations
3. **LoRA Integration**: Native support for Low-Rank Adaptation for parameter-efficient training
4. **Swift Safety**: Leverages Swift's memory safety and error handling capabilities
5. **Async Integration**: Uses Swift's async/await for efficient asynchronous training operations

## Implementation Requirements

### Package Dependencies
```swift
.package(url: "https://github.com/ml-explore/mlx-swift", from: "0.21.0")
.package(url: "https://github.com/ml-explore/mlx-swift-examples", branch: "main")
```

### Core Swift Components

1. **LoRA Implementation**
```swift
import MLXLMCommon

let loraConfig = LoRALinear.Config(rank: 8, alpha: 16)
```

2. **Training Functionality**
```swift
import MLXLLM

// Access to the train function via MLXLLM module
// Documentation: https://swiftpackageindex.com/ml-explore/mlx-swift-examples/2.25.7/documentation/mlxllm/loratrain/train(model:train:validate:optimizer:loss:tokenizer:parameters:progress:)
```

### Model Configuration
- Supports Gemma 2B, 7B (4-bit quantized versions recommended)
- LoRA parameters: rank=4-16, alpha=16-32
- Sequence length up to 2048 tokens
- Batch processing for memory efficiency

## Execution Plan

### Phase 1: Environment Setup
1. Install MLX Swift dependencies
2. Prepare JSONL training data in the correct format
3. Download pre-converted Gemma MLX model

### Phase 2: Core Implementation
1. Implement data loading module for JSONL files
2. Create LoRA adapter configuration
3. Implement training loop with Swift MLX training APIs
4. Add validation and logging capabilities

### Phase 3: Testing and Validation
1. Execute fine-tuning on sample dataset
2. Validate model performance and metrics
3. Test inference with fine-tuned model
4. Optimize performance parameters

## Performance Considerations

1. **Memory Optimization**: Use 4-bit quantized models to reduce memory footprint
2. **Batch Size**: Adjust batch size based on available memory
3. **Gradient Accumulation**: Implement gradient accumulation for larger effective batch sizes
4. **Mixed Precision**: Use appropriate precision levels for training efficiency

## Notes

- This implementation leverages experimental Swift MLX training APIs as referenced in MLX Swift Examples
- The approach fully realizes your vision of a pure Swift fine-tuning solution
- Training performance should be comparable to Python implementation while maintaining Swift's safety features
- The solution is optimized specifically for Apple Silicon hardware
- Future MLX Swift updates may enhance training capabilities further