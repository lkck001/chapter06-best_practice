# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    def __init__(self):
        self.env = 'default'
        self.vis_port = 8097
        self.model = 'SqueezeNet'
        
        # Visualization settings
        self.use_visdom = False  # Set to False by default
        
        # Check if CUDA is available
        self.use_gpu = False  # Default to False
        try:
            if t.cuda.is_available():
                self.use_gpu = True
        except AssertionError:
            self.use_gpu = False
            print("CUDA not available, using CPU instead")
        
        # Data parameters
        self.train_data_root = 'data/train/train'  # 训练集存放路径
        self.test_data_root = 'data/test1/test1'  # 测试集存放路径
        self.load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

        # Training parameters
        self.batch_size = 256  # batch size
        self.num_workers = 4  # how many workers for loading data
        self.print_freq = 20  # print info every N batch

        # Debug and results
        self.debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
        self.result_file = 'result.csv'

        # Optimization parameters
        self.max_epoch = 10
        self.lr = 0.001  # initial learning rate
        self.lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
        self.weight_decay = 0.0  # 损失函数

    def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        # Create a mapping for deprecated parameter names
        param_mapping = {
            'data_root': 'test_data_root',
            'load_path': 'load_model_path'
        }
        
        for k, v in kwargs.items():
            # Check if this is a deprecated parameter name
            if k in param_mapping:
                actual_key = param_mapping[k]
                warnings.warn(f"Warning: '{k}' is deprecated, use '{actual_key}' instead")
                k = actual_key
            
            if k == 'use_gpu':
                # If CUDA is not available, force use_gpu to False
                if not t.cuda.is_available() and v:
                    print("Warning: CUDA not available, using CPU instead")
                    v = False
            if not hasattr(self, k):
                warnings.warn(f"Warning: Config has no attribute {k}")
                continue
            setattr(self, k, v)

        self.device = t.device('cuda' if self.use_gpu else 'cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
