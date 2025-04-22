import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import copy
import numpy as np
import os

# 1. Function to count convolutional and linear layers and their parameters
def count_model_layers(model):
    conv_layers = []
    linear_layers = []
    
    def find_layers(module, prefix=''):
        for name, layer in module.named_children():
            layer_name = prefix + ('.' if prefix else '') + name
            
            if isinstance(layer, nn.Conv2d):
                params = layer.weight.numel()
                conv_layers.append((layer_name, layer, params))
            elif isinstance(layer, nn.Linear):
                params = layer.weight.numel()
                linear_layers.append((layer_name, layer, params))
            
            find_layers(layer, layer_name)
    
    find_layers(model)
    
    print(f"ResNet18 Architecture Analysis:")
    print(f"Total convolutional layers: {len(conv_layers)}")
    total_conv_params = sum(params for _, _, params in conv_layers)
    print(f"Total parameters in convolutional layers: {total_conv_params:,}")
    
    print(f"\nConvolutional Layers:")
    for name, layer, params in conv_layers:
        shape_str = f"{layer.out_channels}x{layer.in_channels}x{layer.kernel_size[0]}x{layer.kernel_size[1]}"
        print(f"  {name}: {shape_str}, Parameters: {params:,}")
    
    print(f"\nTotal linear layers: {len(linear_layers)}")
    total_linear_params = sum(params for _, _, params in linear_layers)
    print(f"Total parameters in linear layers: {total_linear_params:,}")
    
    print(f"\nLinear Layers:")
    for name, layer, params in linear_layers:
        shape_str = f"{layer.out_features}x{layer.in_features}"
        print(f"  {name}: {shape_str}, Parameters: {params:,}")
    
    print(f"\nTotal parameters: {total_conv_params + total_linear_params:,}")
    
    return conv_layers, linear_layers

# 2. Function to prune a single layer at a time by 90% and measure accuracy
def prune_individual_layers(model, conv_layers, linear_layers, test_loader, device):
    baseline_model = copy.deepcopy(model)
    baseline_accuracy = test(baseline_model, test_loader, device)
    print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    
    results = {
        'layer_name': [],
        'accuracy': [],
        'accuracy_drop': []
    }
    
    # Test pruning each convolutional layer
    for name, _, _ in conv_layers:
        print(f"Pruning layer: {name}")
        pruned_model = copy.deepcopy(model)
        
        # Find the layer by name
        layer = pruned_model
        for part in name.split('.'):
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        
        # Prune the layer by 90%
        prune.l1_unstructured(layer, name='weight', amount=0.9)
        
        # Evaluate the model
        accuracy = test(pruned_model, test_loader, device)
        
        # Save results
        results['layer_name'].append(name)
        results['accuracy'].append(accuracy)
        results['accuracy_drop'].append(baseline_accuracy - accuracy)
    
    # Test pruning the linear layer
    for name, _, _ in linear_layers:
        print(f"Pruning layer: {name}")
        pruned_model = copy.deepcopy(model)
        
        # Find the layer by name
        layer = pruned_model
        for part in name.split('.'):
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        
        # Prune the layer by 90%
        prune.l1_unstructured(layer, name='weight', amount=0.9)
        
        # Evaluate the model
        accuracy = test(pruned_model, test_loader, device)
        
        # Save results
        results['layer_name'].append(name)
        results['accuracy'].append(accuracy)
        results['accuracy_drop'].append(baseline_accuracy - accuracy)
    
    # Create and save the bar plot
    plt.figure(figsize=(14, 8))
    y_pos = np.arange(len(results['layer_name']))
    plt.bar(y_pos, results['accuracy'])
    plt.xticks(y_pos, results['layer_name'], rotation=90)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Layer Name')
    plt.title('Model Accuracy After Pruning Individual Layers by 90%')
    plt.tight_layout()
    plt.savefig('individual_layer_pruning.png')
    
    plt.figure(figsize=(14, 8))
    sorted_indices = np.argsort(results['accuracy_drop'])[::-1] 
    top_layers = [results['layer_name'][i] for i in sorted_indices]
    top_drops = [results['accuracy_drop'][i] for i in sorted_indices]
    
    y_pos = np.arange(len(top_layers))
    plt.bar(y_pos, top_drops)
    plt.xticks(y_pos, top_layers, rotation=90)
    plt.ylabel('Accuracy Drop (%)')
    plt.xlabel('Layer Name')
    plt.title('Accuracy Drop After Pruning Individual Layers by 90% (Sorted)')
    plt.tight_layout()
    plt.savefig('layer_sensitivity.png')
    
    print(f"Plots saved as 'individual_layer_pruning.png' and 'layer_sensitivity.png'")
    
    # Return the most sensitive layers
    max_drop_idx = np.argmax(results['accuracy_drop'])
    print(f"\nMost sensitive layer: {results['layer_name'][max_drop_idx]}")
    print(f"Accuracy drop: {results['accuracy_drop'][max_drop_idx]:.2f}%")
    
    return results

# 3. Function to find the maximum pruning percentage for linear layers
def find_max_pruning_percentage(model, linear_layers, test_loader, device):
    baseline_model = copy.deepcopy(model)
    baseline_accuracy = test(baseline_model, test_loader, device)
    print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    
    # Binary search to find the maximum pruning percentage
    def binary_search_pruning(low, high):
        if high - low <= 1:
            return low
        
        mid = (low + high) // 2
        print(f"Testing pruning percentage: {mid}%")
        
        pruned_model = copy.deepcopy(model)
        
        # Prune all linear layers by the current percentage
        for name, _, _ in linear_layers:
            layer = pruned_model
            for part in name.split('.'):
                if part.isdigit():
                    layer = layer[int(part)]
                else:
                    layer = getattr(layer, part)
            
            prune.l1_unstructured(layer, name='weight', amount=mid/100.0)
        
        # Evaluate the model
        pruned_accuracy = test(pruned_model, test_loader, device)
        accuracy_drop = baseline_accuracy - pruned_accuracy
        
        print(f"Pruning {mid}%: Accuracy = {pruned_accuracy:.2f}%, Drop = {accuracy_drop:.2f}%")
        
        # If accuracy drop is less than 2%, try higher pruning
        if accuracy_drop <= 2.0:
            return binary_search_pruning(mid, high)
        else:
            return binary_search_pruning(low, mid)
    
    # Start binary search between 0% and 100%
    max_pruning = binary_search_pruning(0, 100)
    
    print(f"\nMaximum pruning percentage for linear layers (k): {max_pruning}%")
    
    # Verify the result with a final test
    final_model = copy.deepcopy(model)
    for name, _, _ in linear_layers:
        layer = final_model
        for part in name.split('.'):
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        
        prune.l1_unstructured(layer, name='weight', amount=max_pruning/100.0)
    
    final_accuracy = test(final_model, test_loader, device)
    final_drop = baseline_accuracy - final_accuracy
    
    print(f"Final verification - Pruning {max_pruning}%:")
    print(f"  Accuracy: {final_accuracy:.2f}%")
    print(f"  Accuracy drop: {final_drop:.2f}%")
    
    return max_pruning

# BasicBlock for ResNet18 with proper residual handling
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

# ResNet architecture that matches the pretrained model
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

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Create ResNet18 model that matches the pretrained weights
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# Function to test model accuracy
def test(model, test_loader, device):
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
    return accuracy

def main():
    torch.manual_seed(42)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the pretrained model
    print("Loading pretrained model...")
    parent_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    model_path = os.path.join(parent_dir_path, "resnet18_cifar10.pth")
    
    model = ResNet18()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load the CIFAR-10 dataset
    print("Loading dataset...")
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    
    # Use a small subset for faster testing
    test_subset_indices = torch.arange(1000)   
    test_subset = torch.utils.data.Subset(test_dataset, test_subset_indices)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=50,
                                         shuffle=False, num_workers=1)
    
    # 1. Count convolutional and linear layers
    print("\n===== Task 1: Counting Layers =====")
    conv_layers, linear_layers = count_model_layers(model)
    
    # 2. Prune individual layers by 90%
    print("\n===== Task 2: Pruning Individual Layers =====")
    layer_pruning_results = prune_individual_layers(model, conv_layers, linear_layers, test_loader, device)
    
    # 3. Find maximum pruning percentage for linear layers
    print("\n===== Task 3: Finding Maximum Pruning Percentage =====")
    max_pruning = find_max_pruning_percentage(model, linear_layers, test_loader, device)
    
    # Print summary
    max_drop_idx = np.argmax(layer_pruning_results['accuracy_drop'])
    print("\n===== Pruning Analysis Summary =====")
    print(f"Total convolutional layers: {len(conv_layers)}")
    print(f"Total linear layers: {len(linear_layers)}")
    print(f"Most sensitive layer: {layer_pruning_results['layer_name'][max_drop_idx]}")
    print(f"Maximum accuracy drop: {layer_pruning_results['accuracy_drop'][max_drop_idx]:.2f}%")
    print(f"Maximum pruning percentage for linear layers: {max_pruning}%")

if __name__ == "__main__":
    main() 