import torch
import torch.nn as nn
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights


class Fire(nn.Module):
    def __init__(
        self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels
    ):
        super(Fire, self).__init__()
        self.in_channels = in_channels

        # Squeeze layer
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        # Expand layer - two parallel paths
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(
            squeeze_channels, expand3x3_channels, kernel_size=3, padding=1
        )
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x)),
            ],
            1,
        )

    def __repr__(self):
        return f"Fire(in={self.in_channels}, s={self.squeeze.out_channels}, e1x1={self.expand1x1.out_channels}, e3x3={self.expand3x3.out_channels})"


class OurSqueezeNet(nn.Module):
    def __init__(self, version="1.1", num_classes=1000):
        super(OurSqueezeNet, self).__init__()
        self.num_classes = num_classes

        if version == "1.1":
            self.features = nn.Sequential(
                # Initial conv layer
                nn.Conv2d(3, 64, kernel_size=3, stride=2),  # Output: 64x111x111
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),  # Output: 64x55x55
                # Fire modules
                Fire(64, 16, 64, 64),  # Output: 128x55x55
                Fire(128, 16, 64, 64),  # Output: 128x55x55
                nn.MaxPool2d(kernel_size=3, stride=2),  # Output: 128x27x27
                Fire(128, 32, 128, 128),  # Output: 256x27x27
                Fire(256, 32, 128, 128),  # Output: 256x27x27
                nn.MaxPool2d(kernel_size=3, stride=2),  # Output: 256x13x13
                Fire(256, 48, 192, 192),  # Output: 384x13x13
                Fire(384, 48, 192, 192),  # Output: 384x13x13
                Fire(384, 64, 256, 256),  # Output: 512x13x13
                Fire(512, 64, 256, 256),  # Output: 512x13x13
            )
        else:
            raise ValueError(f"Unsupported SqueezeNet version {version}")

        # Final convolution and classifier
        self.classifier = nn.Sequential(
            # Input: 512 channels, 13x13 spatial size
            nn.Dropout(p=0.5),  # Dropout for regularization, spatial size unchanged
            # Conv2d: reduces channels from 512->num_classes, maintains spatial size
            # Input: (512, 13, 13) -> Output: (1000, 13, 13)
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            # Global average pooling: reduces spatial dimensions to 1x1
            # Input: (1000, 13, 13) -> Output: (1000, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def train(self, optimizer):
        """
        Example to show how weight decay works
        """
        # Let's say we have a weight that's currently 10.0
        old_weight = 10.0
        gradient = 1.0
        learning_rate = 0.001
        weight_decay = 0.0001  # from your config

        # Without weight decay
        new_weight_no_decay = old_weight - learning_rate * gradient
        # = 10.0 - 0.001 * 1.0
        # = 9.999

        # With weight decay
        decay_term = weight_decay * old_weight  # = 0.0001 * 10.0 = 0.001
        new_weight_with_decay = old_weight - learning_rate * (gradient + decay_term)
        # = 10.0 - 0.001 * (1.0 + 0.001)
        # = 10.0 - 0.001001
        # = 9.998999

    def demonstrate_weight_decay_effect(self):
        """
        Demonstrate how weight decay affects weights over multiple iterations
        """
        # Initial setup
        weight = 10.0
        learning_rate = 0.001
        weight_decay = 0.0001
        gradient = 1.0
        
        # Track weights over 10 iterations
        weight_no_decay = weight
        weight_with_decay = weight
        
        print("Iteration | No Decay  | With Decay | Difference")
        print("-" * 50)
        
        for i in range(10):
            # Without weight decay
            weight_no_decay = weight_no_decay - learning_rate * gradient
            
            # With weight decay
            decay_term = weight_decay * weight_with_decay
            weight_with_decay = weight_with_decay - learning_rate * (gradient + decay_term)
            
            diff = weight_no_decay - weight_with_decay
            print(f"{i+1:9d} | {weight_no_decay:.6f} | {weight_with_decay:.6f} | {diff:.6f}")

    def demonstrate_important_weight_growth(self):
        """
        Demonstrate how important weights can still grow despite weight decay
        """
        # Start with a small weight (1.0)
        weight = 1.0
        learning_rate = 0.001
        weight_decay = 0.0001
        
        # This is a key part: negative gradient (-5.0) means:
        # "the loss would decrease if this weight gets bigger"
        important_gradient = -5.0  
        
        print("Iteration | Weight | Decay Effect | Gradient Effect | Net Change")
        print("-" * 70)
        
        for i in range(10):
            # 1. Weight Decay Effect:
            # Tries to shrink the weight towards zero
            decay_effect = -learning_rate * weight_decay * weight
            # Example: -0.001 * 0.0001 * 1.0 = -0.0000001
            
            # 2. Gradient Effect:
            # Tries to update weight to reduce loss
            gradient_effect = -learning_rate * important_gradient
            # Example: -0.001 * -5.0 = +0.005
            
            # 3. Combined Effect:
            net_change = decay_effect + gradient_effect
           
            
            # 4. Update the weight
            weight = weight + net_change
            
            print(f"{i+1:9d} | {weight:.6f} | {decay_effect:.6f} | {gradient_effect:.6f} | {net_change:.6f}")


def compare_models():
    # Create both models
    our_model = OurSqueezeNet(version="1.1", num_classes=1000)
    pytorch_model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)

    # Set both models to eval mode
    our_model.eval()
    pytorch_model.eval()

    # Create input tensor
    x = torch.randn(1, 3, 224, 224)

    print("\n=== Comparing Our SqueezeNet vs PyTorch Official SqueezeNet ===")

    # Track features through our model
    print("\nOur Model Feature Maps:")
    our_x = x
    for name, module in our_model.features.named_children():
        our_x = module(our_x)
        print(f"{type(module).__name__}: {tuple(our_x.shape)}")

    our_x = our_model.classifier(our_x)
    print(f"After classifier: {tuple(our_x.shape)}")
    our_x = torch.flatten(our_x, 1)
    print(f"Final output: {tuple(our_x.shape)}")

    # Track features through PyTorch model
    print("\nPyTorch Model Feature Maps:")
    torch_x = x
    for name, module in pytorch_model.features.named_children():
        torch_x = module(torch_x)
        print(f"{type(module).__name__}: {tuple(torch_x.shape)}")

    torch_x = pytorch_model.classifier(torch_x)
    print(f"After classifier: {tuple(torch_x.shape)}")
    torch_x = torch.flatten(torch_x, 1)
    print(f"Final output: {tuple(torch_x.shape)}")

    # Compare model structures
    print("\nOur Model Structure:")
    print(our_model)
    print("\nPyTorch Model Structure:")
    print(pytorch_model)


if __name__ == "__main__":
    compare_models()
