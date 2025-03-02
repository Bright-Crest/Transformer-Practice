import torch
from torch import nn


class StandardScaler(nn.Module):
    def __init__(self, shape=None, dim=0, epsilon=1e-8):
        """
        PyTorch 版本的 StandardScaler（继承自 nn.Module）
        :param shape: 输入数据的形状，包含数据个数的维度(若只有一个数据，则此维度也必须填1)，用于初始化均值和标准差张量
        :param dim: 标准化沿哪个维度进行
        :param epsilon: 防止除零的小值
        """
        super().__init__()
        self.epsilon = epsilon
        self.dim = dim

        if shape:
            shape = list(shape)
            shape[dim] = 1
        # 注册缓冲区（非可训练参数，但会随模型移动设备）
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32) if shape else None)
        self.register_buffer("std", torch.ones(shape, dtype=torch.float32) if shape else None)
    
    def fit(self, X):
        """
        计算输入数据的均值和标准差
        :param X: 输入张量 (形状为 [..., features])
        """
        with torch.no_grad():  # 不追踪梯度
            self.mean = torch.mean(X, dim=self.dim).unsqueeze(self.dim)
            self.std = torch.std(X, dim=self.dim, unbiased=False).unsqueeze(self.dim)  # 与 sklearn 一致（使用有偏标准差）
    
    def forward(self, X, is_inverse=False, update_stats=False):
        """
        前向传播（应用标准化）
        :param X: 输入张量
        :param update_stats: 是否动态更新均值和标准差（默认使用预计算的）
        :return: 标准化后的张量
        """
        if update_stats or (self.mean is None) or (self.std is None):
            # 若未调用 fit 或强制更新，则动态计算
            self.fit(X)
        
        if not is_inverse:
            return self.transform(X)
        else:
            return self.inverse_transform(X)
    
    def transform(self, X):
        """
        变换（标准化）
        """
        return (X - self.mean) / (self.std + self.epsilon)
    
    def inverse_transform(self, X_scaled):
        """
        逆变换（将标准化后的数据还原）
        """
        return X_scaled * (self.std + self.epsilon) + self.mean


class ModuleWithScaler(nn.Module):
    def __init__(self, model, input_scaler=None, output_scaler=None):
        super().__init__()
        self.model = model
        self.input_scaler = input_scaler if input_scaler else StandardScaler()
        self.output_scaler = output_scaler if output_scaler else StandardScaler()
    
    # def init_scaler_for_load_state_dict(self, state_dict):
        
    
    def fit_scaler(self, tensor, is_input, *args, **kwargs):
        '''
        计算训练集的均值和方差。只在训练前使用。
        :param X: 所有训练集
        '''
        if is_input:
            self.input_scaler.fit(tensor, *args, **kwargs)
            return self.input_scaler(tensor)
        else:
            self.output_scaler.fit(tensor, *args, **kwargs)
            return self.output_scaler(tensor)

    def apply_scaler(self, tensor, is_input, is_inverse = False):
        '''
        应用均值和方差进行标准化
        :param X: 输入张量
        '''
        if is_input:
            return self.input_scaler(tensor, is_inverse)
        else:
            return self.output_scaler(tensor, is_inverse)
    
    def forward(self, X, is_scale_input=False, is_predict=False, *args, **kwargs):
        '''
        :param is_predict: 是否是预测阶段，如果是，则进行输出逆标准化
        '''
        if is_scale_input:
            X = self.input_scaler(X)
        X = self.model(X, *args, **kwargs)
        if is_predict:
            X = self.output_scaler(X, is_inverse=True)
        return X
    