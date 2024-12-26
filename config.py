# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    env = 'default'  # visdom environment
    vis_port = 8097  # visdom port
    model = 'SqueezeNet'  # use the model from models directory

    # Data paths
    train_data_root = '/kaggle/input/cat-and-dog/training_set'  # root path for training data
    test_data_root = '/kaggle/input/cat-and-dog/test_set'      # root path for test data
    load_model_path = None  # load pre-trained model path
    
    batch_size = 256  # batch size
    use_gpu = True  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
    
    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # L2 regularization

    def _parse(self, kwargs):
        """
        Update config parameters according to kwargs
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

        print('User Config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

    def parse(self, kwargs):
        """
        Update config parameters according to kwargs and setup device
        """
        self._parse(kwargs)
        self.device = t.device('cuda' if self.use_gpu else 'cpu')
        return self

opt = DefaultConfig()
