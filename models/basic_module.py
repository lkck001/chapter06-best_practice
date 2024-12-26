#coding:utf8
import torch as t
import time
import os
from config import opt  # Import config to get checkpoint directory
import tempfile


class BasicModule(t.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found")
        
        self.load_state_dict(t.load(path, weights_only=True))

    def save(self, name=None):
        """
        Save model in temporary directory
        """
        # Create temp directory
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        
        if name is None:
            name = os.path.join(temp_dir, f'model_{time.strftime("%m%d_%H%M%S")}.pth')
        
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        """
        This is the default optimizer implementation.
        Each model can override this to use a different optimizer
        """
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class Flat(t.nn.Module):
    """
    把输入reshape成（batch_size,dim_length）
    """

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
