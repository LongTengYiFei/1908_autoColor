""" 这个脚本是专门用来大规模灰度化某一些图片的
    图片路径采用硬编码形式

    这个脚本可以直接运行，但是使用前请硬编码文件的路径！！！
"""
import toGray
file_name = "D:\\testPicture\\autoColor\\validationData\\" + "cat.1000" ".jpg"
toGray.toGrayAndSave(file_name)

