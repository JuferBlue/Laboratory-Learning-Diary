"""
@author:JuferBlue
@file:2.3.1-简单的实现.py
@date:2024/9/11 8:45
@description:
"""
def AND(x1, x2):
    w1,w2,theta = 0.5,0.5,0.7
    ans = w1*x1+w2*x2
    return 1 if ans>=theta else 0

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))