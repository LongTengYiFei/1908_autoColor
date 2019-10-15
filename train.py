from keras.models import Sequential
from keras import layers
# 此脚本尚未完成 2019-10-12 17:33
# 此脚本完成了！ 其实昨天就完成了，只是忘了打注释 2019-10-14 18点04分
# 此脚本可以直接使用
model_cat_auto_color = Sequential()
# 单通道的图像输入
print(type(model_cat_auto_color))
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

# 哈哈，我终于调通了，这一层的深度应该是3才对 2019.10.13————23点31分
# 其实搞了半天就是因为这里才出错的，应该不是因为generator ---- 2019.10.14————10点14分
model_cat_auto_color.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
model_cat_auto_color.add(layers.UpSampling2D((2, 2)))
print(model_cat_auto_color.summary())

import  some_functions
model_cat_auto_color.compile(optimizer='RMSprop', loss='mse')

# 我能不能预先载入模型，继续训练？ 10月14日 17点57分
model_cat_auto_color.load_weights('.\\cats_auto_color.h5')

train_X_dir = 'D:\\testPicture\\autoColor\\train_X_Data\\'
train_Y_dir = 'D:\\testPicture\\autoColor\\train_Y_Data\\'

from keras_preprocessing.image import  ImageDataGenerator
train_X_dataGen = ImageDataGenerator(rescale=1. / 255)
train_Y_dataGen = ImageDataGenerator(rescale=1. / 255)


batch_size = 20

train_X_generator = train_X_dataGen.flow_from_directory(
    train_X_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='binary',
    color_mode="grayscale"# 提取为灰度图片
)

train_Y_generator = train_Y_dataGen.flow_from_directory(
    train_Y_dir,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='binary',
    color_mode="rgb"# 提取为RGB图片
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
  只不过第一个值是输入值，是单通道图片
  第二个值是三通道RGB图片
"""
def image_X_Y_generator():
    for batch_x in train_X_generator:
        for batch_y in train_Y_generator:
            yield (batch_x[0], batch_y[0])

# 这里总是报错，说输入层shape不对
# 注意，第一个参数是一个生成器，别搞错了，要搞细节！！！！
steps_per_epoch = input('一个时代走几步？输入数字按回车结束输入：')
epochs = input('走几个时代？输入数字按回车结束输入：')
history = model_cat_auto_color.fit_generator(
    image_X_Y_generator(),
    steps_per_epoch=int(steps_per_epoch),
    epochs=int(epochs),
)
model_cat_auto_color.save('.\\cats_auto_color.h5')