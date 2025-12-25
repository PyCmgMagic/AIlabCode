import os
import shutil
import random
from pathlib import Path
#配置路径
source_root = '/mnt/nfsdir/DistributionBoxUnclosed'  
#目标路径
target_root = './my_dataset'

#创建目录结构
dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
for d in dirs:
    os.makedirs(os.path.join(target_root, d), exist_ok=True)

#获取所有图片
image_files = [f for f in os.listdir(os.path.join(source_root, 'images')) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files) # 打乱顺序

#划分数据集
val_count = 55 # 要求不少于50个，这里取55
val_files = image_files[:val_count]
train_files = image_files[val_count:]

print(f"训练集数量: {len(train_files)}, 验证集数量: {len(val_files)}")

#复制文件函数
def copy_files(file_list, split_type):
    for img_name in file_list:
        # 构造源文件路径
        src_img = os.path.join(source_root, 'images', img_name)
        # 对应的标签文件 (假设是 .txt)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(source_root, 'labels', label_name)
        
        # 复制图片
        shutil.copy(src_img, os.path.join(target_root, 'images', split_type, img_name))
        
        # 复制标签 (如果标签存在)
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(target_root, 'labels', split_type, label_name))

# 执行复制
copy_files(train_files, 'train')
copy_files(val_files, 'val')

print("数据准备完成！")