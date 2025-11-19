# import os
# import SimpleITK as sitk
# import numpy as np
#
#
# def normalize_nii_files(folder_path):
#     # 获取指定文件夹中的所有.nii文件
#     file_list = [file for file in os.listdir(folder_path) if file.endswith('.gz')]
#
#     for file_name in file_list:
#         file_path = os.path.join(folder_path, file_name)
#
#         # 读取.nii文件
#         image = sitk.ReadImage(file_path)
#
#         # 将图像转换为numpy数组
#         image_array = sitk.GetArrayFromImage(image)
#
#         # 归一化操作
#         max_value = np.max(image_array)
#         min_value = np.min(image_array)
#         normalized_array = (image_array - min_value) / (max_value - min_value)
#
#         # 将归一化后的数组重新转换为sitk图像对象
#         normalized_image = sitk.GetImageFromArray(normalized_array)
#         normalized_image.CopyInformation(image)
#
#         # 保存归一化后的图像
#         normalized_file_path = os.path.join(folder_path, file_name)
#         sitk.WriteImage(normalized_image, normalized_file_path)
#
#
# # 指定文件夹路径
# folder_path = "F:/public_dataset/7points_Prostate/reshape128/imagenom"
#
# # 调用函数进行归一化处理
# normalize_nii_files(folder_path)

import os
import nibabel as nib
import numpy as np


def process_nii_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".nii.gz"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                # 读取.nii.gz文件
                img = nib.load(file_path)

                # 获取分辨率
                zooms = img.header.get_zooms()
                print(f"分辨率: {zooms}")

                # 获取长和宽
                shape = img.shape
                height, width, dethp = shape[0], shape[1], shape[2]
                print(f"长: {height}, 宽: {width}, 高: {dethp}")

                # # 获取像素值分布
                data = img.get_fdata()
                min_value = np.min(data)
                max_value = np.max(data)
                mean_value = np.mean(data)
                std_value = np.std(data)
                print(
                    f"最小像素值: {min_value}, 最大像素值: {max_value}, 平均像素值: {mean_value}, 像素值标准差: {std_value}")
                #

# 指定文件夹路径
folder_path = '/home/zoujie/PA-Seg-main/data/VS/annotation_7points'

# 处理.nii.gz文件
process_nii_files(folder_path)


