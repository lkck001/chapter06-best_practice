# coding:utf8
from config import opt
import os
import torch as t
import models
from data.dataset import DogCatDataset, get_dataloader
from torch.utils.data import DataLoader
from utils.visualize import Visualizer
from tqdm import tqdm


@t.no_grad()  # pytorch>=0.5
def test(**kwargs):
    opt.parse(kwargs)

    # Configure paths for Kaggle environment
    checkpoint_dir = '/kaggle/working/checkpoints'
    # List available models in checkpoints directory
    if os.path.exists(checkpoint_dir):
        models_available = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if not models_available:
            raise FileNotFoundError(f"No model files found in {checkpoint_dir}")
        print("Available models:")
        for i, model_file in enumerate(models_available):
            print(f"{i}: {model_file}")
        
        # Use the latest model if not specified
        if opt.load_model_path is None:
            opt.load_model_path = os.path.join(checkpoint_dir, models_available[-1])
            print(f"\nUsing latest model: {opt.load_model_path}")
    else:
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")

    # Configure model
    model = getattr(models, opt.model)().eval()
    
    try:
        model.load(opt.load_model_path)
        print(f"Successfully loaded model from {opt.load_model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    model.to(opt.device)

    # Configure test data
    if not os.path.exists(opt.test_data_root):
        raise FileNotFoundError(
            f"Test data directory {opt.test_data_root} does not exist. Please set correct test_data_root path."
        )

    # Create results directory
    results_dir = '/kaggle/working/results'
    os.makedirs(results_dir, exist_ok=True)
    opt.result_file = os.path.join(results_dir, 'result.csv')

    # Load and process test data
    from data.dataset import DogCat  # Import here to avoid circular import
    test_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )

    # Run inference
    results = []
    print("\nRunning inference on test data...")
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input_data = data.to(opt.device)
        score = model(input_data)
        probability = t.nn.functional.softmax(score, dim=1)[:, 0].detach().tolist()

        batch_results = [
            (path_.item(), probability_)
            for path_, probability_ in zip(path, probability)
        ]
        results += batch_results

    # Save results
    write_csv(results, opt.result_file)
    print(f"\nResults saved to {opt.result_file}")

    return results


def write_csv(results, file_name):
    import csv
    
    try:
        with open(file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label"])
            writer.writerows(results)
        print(f"Successfully wrote {len(results)} results to {file_name}")
    except Exception as e:
        print(f"Error writing results: {str(e)}")


def train(**kwargs):
    opt.parse(kwargs)
    
    # Create checkpoints directory with full Kaggle path
    os.makedirs('/kaggle/working/checkpoints', exist_ok=True)

    # Only create visualizer if use_visdom is True
    vis = Visualizer(opt.env, port=opt.vis_port) if opt.use_visdom else None

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # step2: data
    train_dataloader, test_dataloader = get_dataloader(opt)

    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # step4: meters
    loss_meter = AverageMeter()
    confusion_matrix = AverageMeter()
    previous_loss = 1e10

    # train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):

            # train model
            data = data.to(opt.device)
            label = label.to(opt.device)

            optimizer.zero_grad()
            score = model(data)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())
            # detach 一下更安全保险
            confusion_matrix.add(score.detach(), label.detach())

            if (ii + 1) % opt.print_freq == 0:
                # Only plot if visdom is enabled
                if vis:
                    vis.plot("loss", loss_meter.value()[0])
                else:
                    print(f"Training Loss: {loss_meter.value()[0]:.4f}")

                # 进入debug模式 (Chinese comment meaning "Enter debug mode")
                if os.path.exists(opt.debug_file):
                    import ipdb

                    ipdb.set_trace()

        model.save()

        # validate and visualize
        val_cm, val_accuracy = val(model, test_dataloader)

        # Only use visdom if enabled
        if vis:
            vis.plot("val_accuracy", val_accuracy)
            vis.log(
                "epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                    epoch=epoch,
                    loss=loss_meter.value()[0],
                    val_cm=str(val_cm.value()),
                    train_cm=str(confusion_matrix.value()),
                    lr=lr,
                )
            )
        else:
            print(f"Epoch: {epoch}, LR: {lr:.6f}, Loss: {loss_meter.value()[0]:.4f}")
            print(f"Val Accuracy: {val_accuracy:.2f}%")
            print(f"Train CM:\n{confusion_matrix.value()}")
            print(f"Val CM:\n{val_cm.value()}\n")

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        previous_loss = loss_meter.value()[0]


@t.no_grad()
def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = AverageMeter()
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100.0 * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy



def help():
    """
    打印帮助的信息： python file.py help
    """

    print(
        """
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(
            __file__
        )
    )

    from inspect import getsource

    source = getsource(opt.__class__)
    print(source)


if __name__ == "__main__":
    import fire

    fire.Fire()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
