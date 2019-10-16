from keras import models
from keras.models import Sequential
import tkinter as tk
from tkinter import filedialog
# 本脚本属于 单张测试， 如需多张测试 请另行编写脚本。*************************************
# 怎么使用这个脚本呢，启动它就行了，会弹出来一个文件选择器。

# 这里的模型路径，我们采用了硬编码模式。其实手动选择也可以。
from keras import layers

model_cat_auto_color = Sequential()

print(type(model_cat_auto_color))
"""
卷积层的输出大小你要看输入大小是多少，还要看扫描框多大
第一个参数是输出的深度
第二个参数是扫描框的大小
"""
model_cat_auto_color.add(layers.InputLayer(input_shape=(256, 256, 3)))
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
model_cat_auto_color.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
model_cat_auto_color.add(layers.UpSampling2D((2, 2)))
print(model_cat_auto_color.summary())


model_cat_auto_color.load_weights('.\\cats_auto_color_V2.h5')
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
print("Image resized and arrayed 's shape:", image_arrayed.shape)
###########
# 我现在把黑白图像装换为3通道，只不过每个通道的值都一样，都是像素点的灰度值
# 这是v2版本
##############
image_arrayed_3road = np.arange(256*256*3).reshape(256, 256, 3)

for i in range(image_arrayed.shape[0]):
    for j in range(image_arrayed.shape[1]):
        image_arrayed_3road[i][j][0] = image_arrayed[i][j]
        image_arrayed_3road[i][j][1] = image_arrayed[i][j]
        image_arrayed_3road[i][j][2] = image_arrayed[i][j]
raw_image = Image.fromarray(np.uint8(image_arrayed_3road))
raw_image.show()
# 就算我们想预测仅仅一张图片，我们也要把图片放进一个数组里面。
# 要注意放入的图片的大小，长宽，要符合神经网络的输入
print(image_arrayed_3road)
img_list = np.array([image_arrayed_3road, ])
print("The shape of input of net:", img_list.shape)
result = model_cat_auto_color.predict(img_list)
print("Shape of result:", result.shape)
print("Shape of result[0]:", result[0].shape)
# ************************************************
print(result[0])
print(result[0]*256)
print(type(result[0]))
# 如果不放大的话，三通道都是零点几的小数，看到的图像是黑的
result[0] *= 256
# -------------------------------------------------------
print(result[0])
result_image = Image.fromarray(np.uint8(result[0]))
print(type(result_image))
result_image.show()