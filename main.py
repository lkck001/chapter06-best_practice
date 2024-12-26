# coding:utf8
from config import opt
import os
import torch as t
import models
from data.dataset import DogCatDataset, get_dataloader
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm


@t.no_grad()  # pytorch>=0.5
def test(**kwargs):
    opt.parse(kwargs)

    # Create test directory if it doesn't exist
    os.makedirs(opt.test_data_root, exist_ok=True)

    if not os.path.exists(opt.test_data_root):
        raise FileNotFoundError(
            f"Test data directory {opt.test_data_root} does not exist. Please create it and add test images."
        )

    if len(os.listdir(opt.test_data_root)) == 0:
        raise FileNotFoundError(
            f"Test data directory {opt.test_data_root} is empty. Please add test images."
        )

    # configure model
    model = getattr(models, opt.model)().eval()

    # Check if model path exists
    if opt.load_model_path is None or not os.path.exists(opt.load_model_path):
        print(f"Model file {opt.load_model_path} not found.")
        print("Please train the model first using:")
        print(
            "python main.py train --train-data-root=./data/train --use-gpu=False --env=classifier"
        )
        return

    model.load(opt.load_model_path)
    model.to(opt.device)

    # data
    train_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )
    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)
        score = model(input)
        probability = t.nn.functional.softmax(score, dim=1)[:, 0].detach().tolist()
        # label = score.max(dim = 1)[1].detach().tolist()

        batch_results = [
            (path_.item(), probability_)
            for path_, probability_ in zip(path, probability)
        ]

        results += batch_results
    write_csv(results, opt.result_file)

    return results


def write_csv(results, file_name):
    import csv

    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(results)


def train(**kwargs):
    opt.parse(kwargs)

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
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
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
    confusion_matrix = meter.ConfusionMeter(2)
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
