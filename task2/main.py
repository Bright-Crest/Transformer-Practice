import torch
from torch import nn
import datetime
import os
import sys
import shutil
import numpy as np
import argparse
import time

# 解决模块导入路径问题
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from libs.transformer import TransformerModel
from libs.pipeline import Pipeline
from data import read_data, create_dataloader


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest="command")

parser_train = subparser.add_parser("train", help="训练模型")

parser_predict = subparser.add_parser("predict", help="预测数据")
parser_predict.add_argument("-m", "--model-dir", type=str, help="模型所在的目录名，不要包含output目录名，不要包含模型文件名", required=True)
parser_predict.add_argument("-i", "--input-file", type=str, help="输入数据文件名，相对与本文件的路径或绝对路径", required=True)
parser_predict.add_argument("-o", "--output-file", type=str, help="(可选参数，不提供则直接命令行输出)输出数据文件名，相对与本文件的路径或绝对路径")
parser_predict.add_argument("-l", "--target-len", type=int, help="(可选参数)预测数据的长度，不能超过模型最长预测长度，默认为模型最长预测长度")


def train(is_with_tensorboard=True, seed=42):
    torch.manual_seed(seed)

    data = read_data(DATA_FILE, DATA_LENGTH)
    n_features = data.shape[1]

    training_loader, test_loader = create_dataloader(
        data, INPUT_LEN, tgt_len=TARGET_LEN, batch_size=BATCH_SIZE, training_ratio=TRAINING_RATIO, shuffle=IS_SHUFFLE)

    transformer = nn.Transformer(d_model=D_MODEL, nhead=NHEAD,
                                num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
                                dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, batch_first=True)
    model_transformer = TransformerModel(
        transformer, input_len=INPUT_LEN, target_len=TARGET_LEN, n_features=n_features)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model_transformer.parameters(), lr=LEARNING_RATE)

    # init model
    for p in model_transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # 计算可训练参数量
    trainable_params = sum(p.numel()
                           for p in model_transformer.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    # 计算模型占用的内存大小（以字节为单位）
    param_size = sum(p.numel() * p.element_size()
                     for p in model_transformer.parameters())
    buffer_size = sum(b.numel() * b.element_size()
                      for b in model_transformer.buffers())
    total_size = param_size + buffer_size
    print(f"Model size: {total_size / 1024 ** 2:.2f} MB")

    model = Pipeline(model_transformer, loss, optimizer)
    model.set_seed(seed)
    model.set_loaders(training_loader, test_loader)
    if is_with_tensorboard:
        model.set_tensorboard("transformer", "loss_log")
    model.train(NUM_EPOCHS)

    output_dir = os.path.join("output", datetime.datetime.now().strftime('%m-%d_%H-%M'))
    if os.path.isdir(output_dir):
        output_dir = os.path.join("output", datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    os.makedirs(output_dir, exist_ok=True)

    model.save_checkpoint(os.path.join(output_dir,"transformer.pt"))

    fig = model.plot_losses()
    fig.savefig(os.path.join(output_dir,"transformer_loss.png"))

    # save config
    shutil.copy("config.py", output_dir)


def predict(model_dir, input_file, output_file = None, target_len = None):
    if target_len and target_len > TARGET_LEN:
        raise ValueError("target_len should be less than or equal to TARGET_LEN")

    input = read_data(input_file)
    input_len = input.shape[0]
    n_features = input.shape[1]
    # If a BoolTensor is provided, the positions with the value of True will be ignored while the position with the value of False will be unchanged.
    input_padding_mask = torch.zeros(input_len, dtype=torch.bool)

    if input_len > INPUT_LEN:
        # 截断
        input = input[-INPUT_LEN:]
        input_padding_mask = input_padding_mask[-INPUT_LEN:]
    elif input_len < INPUT_LEN:
        # padding
        input = torch.cat([torch.zeros((INPUT_LEN - input_len, input.shape[1])), input])
        input_padding_mask = torch.cat([torch.ones((INPUT_LEN - input_len), dtype=torch.bool), input_padding_mask])
    
    input.unsqueeze_(0) # (1, L, N_FEATURES)
    input_padding_mask.unsqueeze_(0) # (1, L)
    
    transformer = nn.Transformer(d_model=D_MODEL, nhead=NHEAD,
                                num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
                                dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, batch_first=True)
    model_transformer = TransformerModel(
        transformer, input_len=INPUT_LEN, target_len=TARGET_LEN, n_features=n_features)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model_transformer.parameters(), lr=LEARNING_RATE)

    model = Pipeline(model_transformer, loss, optimizer)
    model.load_checkpoint(os.path.join("output", model_dir, "transformer.pt"))

    output = model.predict(input, input_padding_mask).squeeze(0)
    if target_len:
        output = output[:target_len]

    if output_file:
        np.savetxt(output_file, output.numpy(), delimiter=",")
    else:
        print(output)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.command == "train":
        from config import *
        train(is_with_tensorboard=IS_WITH_TENSORBOARD, seed=SEED)
    elif args.command == "predict":
        os.makedirs("tmp", exist_ok=True)
        shutil.copy(os.path.join("output", args.model_dir, "config.py"), os.path.join("tmp", "config.py"))
        from tmp.config import *
        predict(args.model_dir, args.input_file, args.output_file, args.target_len)
        time.sleep(1)
        shutil.rmtree("tmp")
