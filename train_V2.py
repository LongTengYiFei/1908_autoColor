'''
这是训练脚本第二版本，我们将输入图片提取为rgb三通道，但实际上我们只有黑白图像
'''

from keras.models import Sequential
from keras import layers

model_cat_auto_color = Sequential()

print(type(model_cat_auto_color))
"""
卷积层的输出大小你要看输入大小是多少，还要看扫描框多大
第一个参数是输出的深度
第二个参数是扫描框的大小
"""
# 黑白图像的三通道输入
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

model_cat_auto_color.compile(optimizer='RMSprop', loss='mse')

import os

if os.path.exists('.\\cats_auto_color_V2.h5') == True:
   print('模型文件已存在。')
   model_cat_auto_color.load_weights('.\\cats_auto_color_V2.h5')
   print('模型参数载入成功')
else:
   print('模型不存在，创建新的模型。')

# train_X_Data这个文件夹下面必须只有一个子文件夹，里面全都是训练图片
# train_Y_Data同理
#train_X_dir = 'D:\\testPicture\\autoColor\\train_X_Data\\'
#train_Y_dir = 'D:\\testPicture\\autoColor\\train_Y_Data\\'

# 这里我进行了单张图片的训练，就只训练一张图片，文件路径根据自己的需要来改
train_X_dir = 'D:\\testPicture\\autoColor\\x1\\'
train_Y_dir = 'D:\\testPicture\\autoColor\\y1\\'
from keras_preprocessing.image import  ImageDataGenerator
train_X_dataGen = ImageDataGenerator(rescale=1. / 255)
train_Y_dataGen = ImageDataGenerator(rescale=1. / 255)

batch_size = int(input('batch的大小是多少？输入数字按回车结束：'))

train_X_generator = train_X_dataGen.flow_from_directory(
    train_X_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='binary',
    color_mode="rgb"
)

train_Y_generator = train_Y_dataGen.flow_from_directory(
    train_Y_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='binary',
    color_mode="rgb"
)

"""
The output of the generator must be a tuple 
(inputs, targets)
This tuple  makes a single batch. 
Different batches may have different sizes. 
For example, the last batch of the epoch is commonly smaller than the others.
"""
"""
  上面两个generator一个x一个y。
  产生的是一个元组，第一个值是一个batch，里面全是图片的三维数组形式
  第二个值是一个batch，是对应的标签向量。
  但是我们这个训练要的不是这些，这些是keras自带的工具，用来搞分类用的
  搞分类我已经做猫狗分类用过了
  现在我要的是输入是图片，输出也是图片
  所以我得自己写一个generator
  产生的元组第一个值和第二个值都是图片的三维数组形式的batch
"""
def image_X_Y_generator():
    for batch_x in train_X_generator:
        for batch_y in train_Y_generator:
            yield (batch_x[0], batch_y[0])


steps_per_epoch = input('一个时代走几个batch？输入数字按回车结束输入：')
epochs = input('走几个时代？输入数字按回车结束输入：')
history = model_cat_auto_color.fit_generator(
    image_X_Y_generator(),
    steps_per_epoch=int(steps_per_epoch),
    epochs=int(epochs),
)
model_cat_auto_color.save('.\\cats_auto_color_V2.h5')
