import os

# 设置目录路径
directory = 'au'

# 列出目录下的所有文件
files = os.listdir(directory)

# 打印文件名
for file in files:
    print(file)
