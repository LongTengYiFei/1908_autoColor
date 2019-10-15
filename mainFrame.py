"""关于自动上色的话
   步骤1：找大量图片  OK
   步骤2：全部转为灰度图像  OK
   步骤3：分割为训练集，验证集，测试集  OK
   步骤4：训练 OK
   步骤4.1 采集数据集 OK
   步骤4.2 设计架构 OK
   步骤5：测试
"""



from fileSelect import fileSelect
picture_name = fileSelect()

from fileNameGet import  dotAndSuffixGet
print(dotAndSuffixGet(picture_name))

from fileNameGet import  fileNameGet
print(fileNameGet(picture_name))
