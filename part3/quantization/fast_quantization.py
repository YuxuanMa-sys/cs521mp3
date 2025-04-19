import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import time
import copy
import numpy as np
import struct

# Helper function to print model size
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6  # Size in MB
    print(f"Model size: {size:.6f} MB")
    os.remove("temp.p")
    return size

# Function to measure inference time
def measure_inference_time(model, test_loader, num_runs=10, force_cpu=False):
    # For quantized models, we must use CPU
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    # Warm up with just one batch
    for images, _ in test_loader:
        images = images.to(device)
        _ = model(images)
        break
    
    # Measure time
    total_time = 0
    num_samples = 0
    
    with torch.no_grad():
        for _ in range(num_runs):
            for images, _ in test_loader:
                images = images.to(device)
                batch_size = images.size(0)
                
                start_time = time.time()
                _ = model(images)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                num_samples += batch_size
                
                # Break after one batch for speed
                break
    
    avg_time = total_time / num_samples * 1000  # ms per sample
    print(f"Average inference time: {avg_time:.4f} ms per sample")
    return avg_time

# Function to test model accuracy
def test(model, test_loader, cuda=True):
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# Custom ResNet18 for CIFAR-10
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet18, self).__init__()
        # Load ResNet18 model with pretrained weights
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the first conv layer for CIFAR-10's 32x32 images
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool for small images
        
        # Modify the FC layer for CIFAR-10's 10 classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

def save_int8_model(model, filename):
    """
    Save the model weights in INT8 format for reduced size.
    """
    # Create dict to store quantized weights and metadata
    quantized_dict = {}
    
    # For each parameter in the state dict
    for name, param in model.state_dict().items():
        if 'weight' in name and param.dim() > 1:  # Quantize weights (not biases)
            # Find min and max values
            min_val = param.min().item()
            max_val = param.max().item()
            
            # Calculate scale and zero point for int8 quantization
            scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
            zero_point = -min_val / scale if scale != 0 else 0
            
            # Quantize to int8
            quantized_param = torch.round(param / scale + zero_point).clamp(0, 255).byte()
            
            # Store quantized parameter and metadata
            quantized_dict[name] = {
                'data': quantized_param,
                'scale': scale,
                'zero_point': zero_point,
                'min': min_val,
                'max': max_val
            }
        else:
            # Keep non-weight parameters as is
            quantized_dict[name] = {'data': param}
    
    # Save to file
    torch.save(quantized_dict, filename)
    
    # Return size in MB
    return os.path.getsize(filename) / 1e6

def load_int8_model(model, filename):
    """
    Load a model with INT8 weights, dequantizing on-the-fly.
    """
    # Load the quantized dictionary
    quantized_dict = torch.load(filename)
    
    # Create new state dict for dequantized values
    state_dict = {}
    
    # Dequantize parameters
    for name, param_dict in quantized_dict.items():
        if 'scale' in param_dict:  # This was a quantized parameter
            # Get metadata
            quantized_data = param_dict['data']
            scale = param_dict['scale']
            zero_point = param_dict['zero_point']
            
            # Dequantize
            dequantized = (quantized_data.float() - zero_point) * scale
            state_dict[name] = dequantized
        else:
            # This was not quantized, just copy
            state_dict[name] = param_dict['data']
    
    # Load state dict into model
    model.load_state_dict(state_dict)
    return model

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # CIFAR-10 dataset preprocessing
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load the CIFAR-10 dataset
    print("Loading dataset...")
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Use a smaller subset for faster testing
    train_subset_indices = torch.arange(500)  # Use more training samples for better training
    test_subset_indices = torch.arange(100)   # Use more test samples for better accuracy measurement
    
    train_subset = torch.utils.data.Subset(train_dataset, train_subset_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_subset_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=64, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=64, shuffle=False, num_workers=2
    )
    
    # ------------ 1. Original FP32 Model (ResNet18) ------------
    print("\n====== Original Model (FP32 ResNet18) ======")
    fp32_model = CustomResNet18(num_classes=10)
    
    # Fine-tune the model
    fp32_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(fp32_model.parameters(), lr=0.01, momentum=0.9)
    
    print("Training model...")
    fp32_model.train()
    for epoch in range(3):  # Train for 3 epochs
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = fp32_model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Make sure the models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Save the original model in standard format
    standard_model_path = "models/resnet18_cifar10.pth"
    torch.save(fp32_model.state_dict(), standard_model_path)
    fp32_size_on_disk = os.path.getsize(standard_model_path) / 1e6  # Size in MB
    
    # 1. Measure size of the original model
    print("\nOriginal model size:")
    print(f"Size on disk: {fp32_size_on_disk:.6f} MB")
    
    # 1. Measure accuracy of the original model
    print("\nOriginal model test accuracy:")
    fp32_accuracy = test(fp32_model, test_loader, cuda=use_cuda)
    
    # 3. Measure inference time of the original model
    print("\nMeasuring original model inference time...")
    fp32_time = measure_inference_time(fp32_model, test_loader)
    
    # ------------ 2. INT8 Quantization ------------
    print("\n====== INT8 Quantization ======")
    
    # Save the model in INT8 format
    quantized_model_path = "models/resnet18_cifar10_int8.pth"
    print("Saving INT8 model...")
    int8_size_on_disk = save_int8_model(fp32_model, quantized_model_path)
    print(f"INT8 model size on disk: {int8_size_on_disk:.6f} MB")
    
    # Load the quantized model back
    print("Loading INT8 model...")
    quantized_model = CustomResNet18(num_classes=10)
    quantized_model = load_int8_model(quantized_model, quantized_model_path)
    
    # 2. Measure accuracy of the quantized model
    print("\nQuantized model test accuracy:")
    int8_accuracy = test(quantized_model, test_loader, cuda=use_cuda)
    
    # 3. Measure inference time of the quantized model
    print("\nMeasuring quantized model inference time...")
    int8_time = measure_inference_time(quantized_model, test_loader)
    
    # Calculate size reduction and speedup
    size_reduction = (1 - int8_size_on_disk / fp32_size_on_disk) * 100
    speedup = fp32_time / int8_time
    
    # Store results for final report
    results = {
        'Model': ['Original (FP32)', 'Quantized (INT8)'],
        'Size (MB)': [fp32_size_on_disk, int8_size_on_disk],
        'Accuracy (%)': [fp32_accuracy, int8_accuracy],
        'Inference Time (ms)': [fp32_time, int8_time],
        'Speedup': [1.0, speedup],
        'Size Reduction (%)': [0.0, size_reduction]
    }
    
    # ------------ Print Results Summary ------------
    print("\n====== Quantization Results ======")
    print(f"{'Model':<20} {'Size (MB)':<15} {'Reduction (%)':<15} {'Accuracy (%)':<15} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 90)
    
    for i in range(len(results['Model'])):
        print(f"{results['Model'][i]:<20} {results['Size (MB)'][i]:<15.6f} {results['Size Reduction (%)'][i]:<15.2f} "
              f"{results['Accuracy (%)'][i]:<15.2f} {results['Inference Time (ms)'][i]:<15.4f} {results['Speedup'][i]:<10.2f}x")
    

if __name__ == "__main__":
    main() 