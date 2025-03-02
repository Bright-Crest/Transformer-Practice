import torch
from torch import nn
import os
import sys
import datetime
import shutil
import numpy as np
import argparse
import time

# 解决模块导入路径问题
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from libs.task1_model import CNNModel
from libs.scaler import ModuleWithScaler, StandardScaler
from libs.pipeline import Pipeline
from data import read_data, split_data, create_dataloader


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest="command")

parser_train = subparser.add_parser("train", help="训练模型")

parser_predict = subparser.add_parser("predict", help="预测数据")
parser_predict.add_argument("-m", "--model-dir", type=str, help="模型所在的目录名，不要包含output目录名，不要包含模型文件名", required=True)
parser_predict.add_argument("-i", "--input-file", type=str, help="输入数据文件名，相对与本文件的路径或绝对路径", required=True)
parser_predict.add_argument("-o", "--output-file", type=str, help="(可选参数，不提供则直接命令行输出)输出数据文件名，相对与本文件的路径或绝对路径")


def train(is_with_tensorboard=True, seed=42):
    torch.manual_seed(seed)

    pre_model = CNNModel(INPUT_DIM, OUTPUT_DIM)
    model = ModuleWithScaler(pre_model)

    data = read_data(DATA_FILE, DATA_LENGTH)
    X_training, X_test, y_training, y_test = split_data(data, INPUT_DIM, TRAINING_RATIO)

    X_training = model.fit_scaler(X_training, True)
    X_test = model.apply_scaler(X_test, True)
    y_training = model.fit_scaler(y_training, False)
    y_test = model.apply_scaler(y_test, False)
    
    training_loader, test_loader = create_dataloader(X_training, X_test, y_training, y_test, batch_size=BATCH_SIZE)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE)

    # init model
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.kaiming_uniform_(p)
    
    # 计算可训练参数量
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    # 计算模型占用的内存大小（以字节为单位）
    param_size = sum(p.numel() * p.element_size()
                     for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size()
                      for b in model.buffers())
    total_size = param_size + buffer_size
    print(f"Model size: {total_size / 1024 ** 2:.2f} MB")

    pipeline = Pipeline(model, loss, optimizer)
    pipeline.set_seed(seed)
    pipeline.set_loaders(training_loader, test_loader)
    if is_with_tensorboard:
        pipeline.set_tensorboard("task1", "loss_log")
    pipeline.train(NUM_EPOCHS)

    output_dir = os.path.join("output", datetime.datetime.now().strftime('%m-%d_%H-%M'))
    if os.path.isdir(output_dir):
        output_dir = os.path.join("output", datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    os.makedirs(output_dir, exist_ok=True)

    pipeline.save_checkpoint(os.path.join(output_dir,"task1.pth"))

    fig = pipeline.plot_losses()
    fig.savefig(os.path.join(output_dir,"loss.png"))

    # save config
    shutil.copy("config.py", output_dir)


def predict(model_dir, input_file, output_file = None):
    input = read_data(input_file)
    input = input[:, :INPUT_DIM]

    pre_model = CNNModel(INPUT_DIM, OUTPUT_DIM) 
    model = ModuleWithScaler(pre_model, StandardScaler(shape=(1, INPUT_DIM)), StandardScaler(shape=(1, OUTPUT_DIM)))

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE)

    pipeline = Pipeline(model, loss, optimizer)
    pipeline.load_checkpoint(os.path.join("output", model_dir, "task1.pth"))
    
    output = pipeline.predict(input, is_scale_input=True, is_predict=True)
    if output_file:
        np.savetxt(output_file, output.numpy(), delimiter=",")
    else:
        print(output)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.command == "train":
        from config import *
        train(is_with_tensorboard=IS_WITH_TENSORBOARD, seed=SEED)
    elif args.command == "predict":
        os.makedirs("tmp", exist_ok=True)
        shutil.copy(os.path.join("output", args.model_dir, "config.py"), os.path.join("tmp", "config.py"))
        from tmp.config import *
        predict(args.model_dir, args.input_file, args.output_file)
        time.sleep(1)
        shutil.rmtree("tmp")