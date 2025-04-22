import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import torchvision
import torchvision.transforms as transforms
import os
import time
import copy
import numpy as np

# Print model size in MB
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    print(f"Model size: {size:.6f} MB")
    os.remove("temp.p")
    return size

# Measure average inference time per sample
def measure_inference_time(model, test_loader, num_runs=10):
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    # Warm up
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
                break
    
    avg_time = total_time / num_samples * 1000  # ms per sample
    print(f"Average inference time: {avg_time:.4f} ms per sample")
    return avg_time

# Test model accuracy
def test(model, test_loader, cuda=False):
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

# BasicBlock for ResNet with FloatFunctional for quantization
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                             stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.ff = torch.nn.quantized.FloatFunctional()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ff.add(self.shortcut(x), out)
        out = F.relu(out)
        return out

# ResNet architecture with quantization support
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                             stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.dequant(out)
        return out
    
    def fuse_model(self):
        # Fuse Conv+BN layers for improved performance
        torch.quantization.fuse_modules(self, ['conv1', 'bn1'], inplace=True)
        
        for module_name, module in self.named_children():
            if 'layer' in module_name:
                for basic_block in module:
                    torch.quantization.fuse_modules(
                        basic_block, 
                        [['conv1', 'bn1'], ['conv2', 'bn2']], 
                        inplace=True
                    )
        
        print("Model fused")

# Create ResNet18 model
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def load_pretrained_model():
    parent_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    model_path = os.path.join(parent_dir_path, "resnet18_cifar10.pth")
    
    if not os.path.exists(model_path):
        print(f"Pretrained model not found at {model_path}")
        return None
    
    model = ResNet18()
    
    print(f"Loading pretrained weights from {model_path}")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    return model

def train_minimal_model(model, epochs=1):
    """Train a minimal model if loading fails"""
    print("Training a minimal model...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    subset_indices = torch.arange(10000)
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def main():
    torch.manual_seed(42)
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    
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
    
    # Create data subsets for faster processing
    calibration_indices = torch.arange(1000)
    test_subset_indices = torch.arange(1000)
    
    calibration_subset = torch.utils.data.Subset(train_dataset, calibration_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_subset_indices)
    
    calibration_loader = torch.utils.data.DataLoader(
        calibration_subset, batch_size=32, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=32, shuffle=False, num_workers=2
    )
    
    # Create directory for models
    os.makedirs("models", exist_ok=True)
    
    # 1. Original FP32 Model (ResNet18)
    print("\n====== Original Model (FP32 ResNet18) ======")
    
    fp32_model = load_pretrained_model()
    
    # Measure size of the original model
    print("\nOriginal model size:")
    fp32_size = print_size_of_model(fp32_model)
    
    # Measure accuracy of the original model
    print("\nOriginal model test accuracy:")
    fp32_accuracy = test(fp32_model, test_loader, cuda=use_cuda)
    
    # Measure inference time of the original model
    print("\nMeasuring original model inference time (on CPU)...")
    fp32_time = measure_inference_time(fp32_model, test_loader)
    
    # 2. Prepare for Quantization
    print("\n====== Preparing for Quantization ======")
    
    # Clone the model for quantization
    quantized_model = copy.deepcopy(fp32_model)
    
    # Move to CPU for quantization
    quantized_model.to('cpu')
    
    # Fuse layers for improved performance
    quantized_model.eval()
    quantized_model.fuse_model()
    
    # Set quantization configuration
    quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print(f"Quantization config: {quantized_model.qconfig}")
    
    # Prepare the model for quantization
    torch.quantization.prepare(quantized_model, inplace=True)
    
    # 3. Calibrate the Model
    print("\n====== Calibrating the Model ======")
    print("Running calibration...")
    
    # Run calibration data through the model
    quantized_model.eval()
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            inputs = inputs.to('cpu')
            quantized_model(inputs)
    
    # 4. Convert to Quantized Model
    print("\n====== Converting to Quantized Model ======")
    
    # Convert to quantized model
    torch.quantization.convert(quantized_model, inplace=True)
    
    # Save the quantized model
    quantized_model_path = "models/resnet18_cifar10_quantized.pth"
    torch.save(quantized_model.state_dict(), quantized_model_path)
    
    # Measure size of the quantized model
    print("\nQuantized model size:")
    int8_size = print_size_of_model(quantized_model)
    
    # Measure accuracy of the quantized model
    print("\nQuantized model test accuracy:")
    int8_accuracy = test(quantized_model, test_loader, cuda=False)
    
    # Measure inference time of the quantized model
    print("\nMeasuring quantized model inference time...")
    int8_time = measure_inference_time(quantized_model, test_loader)
    
    # Calculate results
    size_reduction = (1 - int8_size / fp32_size) * 100
    speedup = fp32_time / int8_time if int8_time > 0 else 0
    
    # Print Results Summary
    print("\n====== Quantization Results ======")
    print(f"{'Model':<20} {'Size (MB)':<15} {'Reduction (%)':<15} {'Accuracy (%)':<15} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 90)
    
    print(f"{'Original (FP32)':<20} {fp32_size:<15.6f} {'0.00':<15} {fp32_accuracy:<15.2f} {fp32_time:<15.4f} {'1.00':<10}x")
    print(f"{'Quantized (INT8)':<20} {int8_size:<15.6f} {size_reduction:<15.2f} {int8_accuracy:<15.2f} {int8_time:<15.4f} {speedup:<10.2f}x")

if __name__ == "__main__":
    main() 