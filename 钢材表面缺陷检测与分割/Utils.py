"""
@author:JuferBlue
@file:Utils.py
@date:2024/9/29 8:42
@description:
"""
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_