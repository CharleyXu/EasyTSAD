import numpy as np
import json
import pandas as pd
from Spot import SPOT, biSPOT, bidSPOT, dSPOT
import matplotlib.pyplot as plt

# 步骤1: 读取CSV文件中的数据

## /Users/charliexu/Documents/Bizseer/LLMOps/air_passengers.csv  #Passengers
data = pd.read_csv(
    '/Users/charliexu/Documents/Bizseer/LLMOps/metrics/交易量指标/10.0.210.11.tps.csv')  # 假设CSV文件中有名为'value'的时间序列数据列
time_series_data = data['value'].values  # 获取时间序列数据的numpy数组形式

# 步骤2: 创建SPOT对象并配置检测级别
detection_level = 1e-4  # 检测级别，可根据需要调整
depth = 10
spot = bidSPOT(q=detection_level, depth=depth)

# 步骤3: 使用时间序列数据初始化SPOT对象
# 假设我们使用CSV文件中的全部数据进行SPOT分析
spot.fit(init_data=time_series_data, data=time_series_data)

# 运行初始化步骤
spot.initialize(verbose=True)

# 步骤4: 运行SPOT算法并获取结果
results = spot.run(with_alarm=True)

# 步骤5: 可视化结果
figs = spot.plot(results, with_alarm=True)

# 步骤6: 将结果转换为JSON格式
results_json = {
    'upper_thresholds': results['upper_thresholds'],  # 上阈值列表
    'lower_thresholds': results['lower_thresholds'],  # 下阈值列表
    'alarms': results['alarms']  # 警报索引列表
}

# 步骤7: 打印JSON格式的结果
print(json.dumps(results_json, indent=4))  # 使用indent美化输出

plt.show()
