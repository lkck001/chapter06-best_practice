from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from models.basic_module import  BasicModule
from torch import nn
from torch.optim import Adam

class SqueezeNet(BasicModule):
    def __init__(self, num_classes=2):
        super(SqueezeNet, self).__init__()
        self.model_name = 'squeezenet'
        self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
        
        # The pretrained model already has:
        # self.model.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5),           # index [0]
        #     nn.Conv2d(512, 1000, 1),     # index [1] - we replace this
        #     nn.ReLU(inplace=True),       # index [2]
        #     nn.AdaptiveAvgPool2d((1, 1)) # index [3]
        # )
        
        # We only need to replace the Conv2d layer to change output classes
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.model.num_classes = num_classes

    def forward(self,x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay):
        return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay) 

    def print_trainable_parameters(self):
        # Count all parameters
        all_params = sum(p.numel() for p in self.model.parameters())
        
        # Count trainable parameters (classifier only)
        trainable_params = sum(p.numel() for p in self.model.classifier.parameters())
        
        print(f"Total parameters: {all_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Frozen parameters: {all_params - trainable_params}") 