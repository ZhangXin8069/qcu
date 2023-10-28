import cupy as cp
import numpy as np
import time

# 创建一个大型的随机数组
array_size = int(1e7)
cupy_array = cp.random.rand(array_size)
numpy_array = np.random.rand(array_size)

# 使用CuPy计算平方根
start_time = time.time()
cupy_sqrt = cp.sqrt(cupy_array)
cupy_time = time.time() - start_time

# 使用NumPy计算平方根
start_time = time.time()
numpy_sqrt = np.sqrt(numpy_array)
numpy_time = time.time() - start_time

# 打印时间对比
print("CuPy 平方根计算时间: {:.4f} 秒".format(cupy_time))
print("NumPy 平方根计算时间: {:.4f} 秒".format(numpy_time))

