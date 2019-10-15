# 这个脚本是选择文件的函数，不能直接运行
import tkinter as tk
from tkinter import filedialog
def fileSelect():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path