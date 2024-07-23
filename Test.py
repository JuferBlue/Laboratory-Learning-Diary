"""
@author:JuferBlue
@file:Test.py
@date:2024/7/17 14:33
@description:
"""


import time

start_time = time.time()
for i in range(1000000):
    print(i)

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)


