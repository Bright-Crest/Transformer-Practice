# lyx项目

## Task2

### 环境配置

pytorch

可选

- tensorboard



### 使用说明

1. 必须在task2目录下运行

```bash
cd task2
```

2. 激活python环境

#### 3. 训练模型

```bash
python main.py train
```

#### 4. 预测数据

查看帮助

```bash
python main.py predict -h
```

示例：

```bash
python main.py predict -m 02-26_23-52 -i test.csv -o predict.csv
```



#### tensorboard

使用tensorboard可以方便地查看loss

```bash
tensorboard --logdir loss_log
```



### 目录结构

```
│   config.py ------- 配置参数
│   data.py --------- 处理数据
│   main.py
│
├───loss_log -------- tensorboard生成的，包含loss（建议不要删）
│
├───output ---------- 训练模型的输出
│   └───02-26_23-52 -- 月份-日期_小时-分钟
│           config.py - 训练此模型时用的参数（不能删）
│           transformer.pt - 训练好的模型（不能删）
│           transformer_loss.png - loss图片
```



### 调参-config.py

与模型复杂度相关的主要调整transformer参数

```python
'''
数据文件路径
相对于main.py的路径或绝对路径
'''
DATA_FILE = "dy.csv"
'''
用于训练的(最长)数据个数
理论上使用的数据越多越好，但是训练速度慢。因此此参数调整使用的数据个数，从而快速地比较模型的好坏。
- None: 使用所有数据
'''
DATA_LENGTH = 50000
'''
训练时的输入序列长度
预测时这是最大的输入序列长度。
增大此参数会导致训练成本很快增大。
在训练数据不变的情况下，增大此参数会导致实际的训练数据减少
'''
INPUT_LEN = 70
'''
训练时的输出序列长度
预测时这是最大的输出序列长度。
增大此参数会导致训练成本很快增大。
在训练数据不变的情况下，增大此参数会导致实际的训练数据减少
'''
TARGET_LEN = 30
'''
训练集占比
'''
TRAINING_RATIO = 0.7
'''
是否要在划分训练集和测试集时打乱顺序
'''
IS_SHUFFLE = True
'''
调整此参数可以改变训练的结果
在其他参数都不变的情况下，此参数不变，则训练结果理论上不变。
'''
SEED = 42
'''
是否使用tensorboard
'''
IS_WITH_TENSORBOARD = True


### hyper-parameters ###

'''
(建议不要改)
建议在16到64之间，建议不要太大
'''
BATCH_SIZE = 16
'''
(要调整)
训练轮数(epochs)
loss下不去，可以增加。模型复杂，可以增加。
'''
NUM_EPOCHS = 200
'''
(影响不大，建议不要改)
学习率
建议在0.1到1e-6之间。
'''
LEARNING_RATE = 0.001


### transformer parameters ###
# 以下参数都**可调整**

'''
一定要比数据的维度高。
建议增大。
'''
D_MODEL = 64
'''
一定要能整除`D_MODEL`。
此参数建议保持适中。
`D_MODEL` / `NHEAD`(商) 建议比数据的维度高，比此参数高，建议不要太小。
'''
NHEAD = 8
'''
增大此参数会很快地使模型变复杂。
建议在1到6之间。
可以与`NUM_DECODER_LAYERS`不同。
'''
NUM_ENCODER_LAYERS = 3
'''
增大此参数会很快地使模型变复杂。
建议在1到6之间。
可以与`NUM_ENCODER_LAYERS`不同。
'''
NUM_DECODER_LAYERS = 3
'''
一定要比`D_MODEL`高。
建议增大。
'''
DIM_FEEDFORWARD = 512
'''
建议在0.1到0.5之间。模型更复杂，建议此参数增大
'''
DROPOUT = 0.5
```