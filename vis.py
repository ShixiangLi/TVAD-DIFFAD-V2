import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float

def add_noise(image, noise_level=0.1):
    """
    给图像添加高斯噪声
    :param image: 输入图像 (float, 0~1)
    :param noise_level: 噪声强度 (标准差)
    :return: 添加噪声后的图像
    """
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

# ====== 示例：加载图像 ======
# 你可以换成自己的图片路径
img = img_as_float(io.imread("datasets/combustion_dataset/chamber/test/burn/1156.png"))  

# ====== 添加大噪声和小噪声 ======
small_noise_img = add_noise(img, noise_level=0.2)
large_noise_img = add_noise(img, noise_level=10)

# ====== 可视化 ======
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(small_noise_img)
axes[1].set_title("Small Noise")
axes[1].axis("off")

axes[2].imshow(large_noise_img)
axes[2].set_title("Large Noise")
axes[2].axis("off")

plt.tight_layout()
plt.show()
