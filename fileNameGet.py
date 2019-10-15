import os
# 这个脚本是一些函数，不能直接运行


def dotAndSuffixGet(file_name):
    suffix = os.path.splitext(file_name)[-1]
    return suffix


def fileNameGet(file_name):
    file_name = os.path.splitext(file_name)[0]
    return file_name

