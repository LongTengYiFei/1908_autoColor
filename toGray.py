import cv2
from fileNameGet import  fileNameGet
from fileNameGet import  dotAndSuffixGet
# 这个脚本是有关灰化的函数，不能直接运行
def toGrayAndShow(image_path):
    image = cv2.imread(image_path)
    image_grayed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imshow("image_grayed", image_grayed)
    cv2.waitKey()


def toGrayAndSave(image_path):
    image = cv2.imread(image_path)
    image_grayed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grayed_name = fileNameGet(image_path) + "-grayed" + dotAndSuffixGet(image_path)
    cv2.imwrite(grayed_name, image_grayed)
    print("文件成功灰化并保存！")
