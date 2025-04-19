# Part 3: Quantization and Pruning

This part of the assignment explores techniques for model compression through quantization and pruning of neural networks. We use PyTorch to implement these techniques on a ResNet18 model trained on the CIFAR-10 dataset.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy

You can install the required dependencies with:

```bash
pip install torch torchvision matplotlib numpy
```

## Structure

The implementation is organized into two main directories:

- `quantization/`: Code for post-training static quantization
- `pruning/`: Code for unstructured L1 weight pruning

## Dataset

The code uses the CIFAR-10 dataset, which will be automatically downloaded when the scripts are run.

## Pretrained Model

Before running the code, you need to download the pretrained ResNet18 model weights:

1. Download the weights from the provided link: https://uofi.box.com/s/bwmqq6aet01cozhhtes10btsj7rpd6o
2. Save the weights file as `resnet18_cifar10.pth` in the same directory where you run the scripts.

## Running the Code

### 1. Quantization

```bash
cd quantization
python quantization.py
```

This will:
- Load the pretrained ResNet18 model
- Measure the model size and accuracy
- Calibrate and quantize the model to INT8
- Measure the quantized model size and accuracy
- Compare inference times

### 2. Pruning

```bash
cd pruning
python pruning.py
```

This will:
- Analyze the ResNet18 model structure and count parameters by layer
- Prune each layer individually by 90% and measure accuracy impact
- Create a bar plot showing the effect of pruning each layer
- Find the maximum pruning percentage for linear layers that maintains accuracy within 2% of the original

## Expected Results

### Quantization
- Reduced model size (typically 75% smaller)
- Minimal loss in accuracy (usually <1%)
- Faster inference times on CPU

### Pruning
- Identification of the most important layers in the model
- Demonstration that different layers have varying sensitivity to pruning
- Finding an optimal pruning percentage that maintains accuracy

## Notes

- The pruning code uses a subset of the test dataset for faster evaluation. Modify the subset size as needed.
- Both implementations support GPU usage if available, but quantization is primarily optimized for CPU inference.
- The binary search for finding the maximum pruning percentage is more efficient than a linear search. 