from keras import models
from keras.models import Sequential
import tkinter as tk
from tkinter import filedialog
# 本脚本属于 单张测试， 如需多张测试 请另行编写脚本。*************************************
# 怎么使用这个脚本呢，启动它就行了，会弹出来一个文件选择器。

# 这里的模型路径，我们采用了硬编码模式。其实手动选择也可以。
from keras import layers
# 载入模型时出错 10.14---0点01分
model_cat_auto_color = Sequential()
"""
卷积层的输出大小你要看输入大小是多少，还要看扫描框多大
第一个参数是输出的深度
第二个参数是扫描框的大小
"""
model_cat_auto_color.add(layers.InputLayer(input_shape=(256, 256, 1)))
model_cat_auto_color.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model_cat_auto_color.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model_cat_auto_color.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model_cat_auto_color.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model_cat_auto_color.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model_cat_auto_color.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model_cat_auto_color.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model_cat_auto_color.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model_cat_auto_color.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model_cat_auto_color.add(layers.UpSampling2D((2, 2)))
model_cat_auto_color.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model_cat_auto_color.add(layers.UpSampling2D((2, 2)))
model_cat_auto_color.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model_cat_auto_color.add(layers.Conv2D(2, (3, 3), activation='tanh', padding='same'))
model_cat_auto_color.add(layers.UpSampling2D((2, 2)))
print(model_cat_auto_color.summary())
# 像我这样载入模型就好了 2019年10月14日--- 17点29分

model_cat_auto_color.load_weights('.\\cats_auto_color.h5')
# -------------------------------------------
import numpy as np
from PIL import Image
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
# -----------------------------------------
image_opened = Image.open(file_path)
image_resized = image_opened.resize((256, 256), Image.ANTIALIAS)
image_arrayed = np.array(image_resized)
image_reshaped = image_arrayed.reshape((256, 256, 1))
print("Image resized and arrayed 's shape:", image_arrayed.shape)
# 就算我们想预测仅仅一张图片，我们也要把图片放进一个数组里面。
# 要注意放入的图片的大小，长宽，要符合神经网络的输入

img_list = np.array([image_reshaped, ])
print("The shape of input of net:", img_list.shape)
result = model_cat_auto_color.predict(img_list)
print("Shape of result:", result.shape)
print("Shape of result[0]:", result[0].shape)
# ************************************************

# 通道值放大
result[0] *= 128

# image_reshaped 只是单通道的灰度图像
# 菜鸟教程，列表截取
cur = np.zeros((256, 256, 3))
cur[:, :, 0] = image_reshaped[:, :, 0]
cur[:, :, 1:] = result[0]


# -------------------------------------------------------
# 虽然有图像了，但是很模糊，完全就是瞎搞，得重新训练 2019年10月14日 17点48分
result_image = Image.fromarray(np.uint8(cur))
print(type(result_image))
result_image.show()

"""自动上色成功，现在解决generator不匹配问题   10.24
   测试的话就选择一张黑白图片就好了
   会返回一张上色了的彩色图片
"""

