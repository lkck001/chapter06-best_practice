import torch
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

def calculate_output_size(input_size, kernel_size, stride, padding=0):
    return ((input_size + 2*padding - kernel_size) // stride) + 1

def print_clean_structure(model):
    """Print clean model structure matching the paper diagram"""
    print("\n=== SqueezeNet Structure ===")
    
    # Track spatial dimensions
    h = w = 224  # Input size
    
    # Process each layer in order
    fire_count = 0
    for name, module in model.features.named_children():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            # Print conv1
            h = calculate_output_size(h, 3, 2)
            w = calculate_output_size(w, 3, 2)
            print(f"└─ conv1: Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2))")
            print(f"   └─ output size: {h}x{w}, channels: 64")
        elif isinstance(module, torch.nn.modules.pooling.MaxPool2d):
            h = calculate_output_size(h, 3, 2)
            w = calculate_output_size(w, 3, 2)
            print(f"└─ maxpool/2")
            print(f"   └─ output size: {h}x{w}")
        elif 'Fire' in str(type(module)):
            fire_count += 1
            in_channels = module.squeeze.in_channels
            out_channels = module.expand1x1.out_channels + module.expand3x3.out_channels
            print(f"└─ fire{fire_count}: Fire({in_channels} -> {out_channels})")
            print(f"   └─ output size: {h}x{w}, channels: {out_channels}")
    
    # Print final layers
    print(f"└─ conv10: Conv2d(512, 1000, kernel_size=(1, 1))")
    print(f"   └─ output size: {h}x{w}, channels: 1000")
    print("└─ global avgpool")
    print("   └─ output size: 1x1, channels: 1000")
    print("└─ softmax")
    
    print(f"\nTotal Fire modules: {fire_count}")

# Create model
model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
print_clean_structure(model)
