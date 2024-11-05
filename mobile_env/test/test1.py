import os
import sys

# os.chdir("D:/桌面/NUS/Tham Project/mobile-env-gan")
# sys.path.append(os.path.abspath("D:\\桌面\\NUS\\Tham Project\\mobile-env-gan"))
# print(os.getcwd())  # 确认已经切换到项目根目录
# print("hello")

import matplotlib.pyplot as plt
from IPython import display
from mobile_env.scenarios.custom import MComCustom

# 创建环境实例
env = MComCustom(render_mode="rgb_array")

# env.reset()
# for _ in range(20):
#     # 调用 step 方法，模拟时间推进
#     info = env.step()
#
#     # 调用 features 以记录每一步的数据
#     env.features()

#
# 数据收集主循环
for idx in range(15):
    # 重置环境，并生成新的基站位置
    env.reset()
    done = False

    print("current idx:", idx)
    # 模拟用户移动
    for _ in range(20):
        # 调用 step 方法，模拟时间推进
        info = env.step()

        # 调用 features 以记录每一步的数据
        env.features()

        plt.imshow(env.render())
        display.display(plt.gcf())
        display.clear_output(wait=True)

    # 保存每次得到的数据到 CSV 文件
    env.save_results_to_file(
        datarate_filename=f'Data_Rate_{idx}.csv',
        utility_filename=f'QoE_{idx}.csv'
    )

print("数据收集完成。")
