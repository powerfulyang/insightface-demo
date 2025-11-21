import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import KMeans

# 初始化 InsightFace 模型
app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 递归扫描 images
image_folder = "faces_folder"
image_files = []

for root, dirs, files in os.walk(image_folder):
    for f in files:
        if f.lower().endswith(('.jpg', '.png')):
            full_path = os.path.join(root, f)
            image_files.append(full_path)

# 输出文件夹
output_folder = "cluster_result"
os.makedirs(output_folder, exist_ok=True)

# 存储人脸特征和对应文件路径
features = []
file_paths = []

# 提取所有人脸特征
for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取文件（可能已损坏或路径异常）：{img_path}")
        continue
    faces = app.get(img)

    if len(faces) > 0:
        feat = faces[0].normed_embedding
        features.append(feat)
        file_paths.append(img_path)
    else:
        print(f"{img_path} 未检测到人脸")

# 转换为 numpy 数组
features = np.array(features)

# 使用 K-Means 聚类
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(features)

def get_unique_filename(dst_folder, filename):
    """
    保证文件名唯一：xxx.jpg → xxx.jpg, xxx_1.jpg, xxx_2.jpg ...
    """
    name, ext = os.path.splitext(filename)
    candidate = filename
    counter = 1

    while os.path.exists(os.path.join(dst_folder, candidate)):
        candidate = f"{name}_{counter}{ext}"
        counter += 1

    return candidate

# 输出分类结果
for i in range(n_clusters):
    print(f"\n类别 {i+1}:")
    cluster_files = [file_paths[j] for j in range(len(labels)) if labels[j] == i]

    # 为每类创建独立目录
    class_folder = os.path.join(output_folder, f"class_{i+1}")
    os.makedirs(class_folder, exist_ok=True)

    for src_path in cluster_files:
        print(f"  - {src_path}")

        img = cv2.imread(src_path)
        cv2.putText(img, f"Class {i+1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 基础文件名（不保留原子路径结构）
        base_name = os.path.basename(src_path)

        # 生成唯一文件名
        unique_name = get_unique_filename(class_folder, base_name)

        # 最终保存路径
        dst_path = os.path.join(class_folder, unique_name)

        cv2.imwrite(dst_path, img)

print("\n🎉 分类完成，所有结果已保存到 cluster_result 目录")
