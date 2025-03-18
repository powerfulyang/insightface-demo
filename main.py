import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import KMeans

# 初始化 InsightFace 模型
app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 文件夹路径，存放多张人脸图片
image_folder = "faces_folder"  # 替换为你的图片文件夹路径
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# 存储人脸特征和对应文件名
features = []
file_names = []

# 提取所有人脸特征
for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    img = cv2.imread(img_path)
    faces = app.get(img)  # 检测人脸

    if len(faces) > 0:
        feat = faces[0].normed_embedding  # 取第一个人脸的特征向量
        features.append(feat)
        file_names.append(img_file)
    else:
        print(f"{img_file} 未检测到人脸")

# 转换为 numpy 数组
features = np.array(features)

# 使用 K-Means 聚类（假设有 3 个不同的人，实际可根据需求调整）
n_clusters = 3  # 可根据实际情况修改
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(features)

# 输出分类结果
for i in range(n_clusters):
    print(f"\n类别 {i + 1}:")
    cluster_files = [file_names[j] for j in range(len(labels)) if labels[j] == i]
    for file in cluster_files:
        print(f"  - {file}")

# 可视化（可选）：将每类人脸显示出来
for i in range(n_clusters):
    cluster_files = [file_names[j] for j in range(len(labels)) if labels[j] == i]
    for file in cluster_files:
        img = cv2.imread(os.path.join(image_folder, file))
        cv2.putText(img, f"Class {i + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"Class {i + 1}", img)
        cv2.waitKey(500)  # 每张图显示 0.5 秒

cv2.destroyAllWindows()
