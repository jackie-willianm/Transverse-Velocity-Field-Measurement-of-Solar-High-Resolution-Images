import glob
import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog


def rename_img(soupath, renamepath):
    # 函数输入mat文件路径和转换后输出的flo文件路径
    count = 0
    sous = glob.glob(os.path.join(soupath, '*.jpg')) + \
        glob.glob(os.path.join(soupath, '*.png'))
    sous = sorted(sous)


    for soufile in sous:
        sou = cv2.imread(soufile, cv2.IMREAD_GRAYSCALE)
        # sou = Image.open(soufile)
        # sou = np.array(sou).astype(np.uint8)
        # 获取mat文件的名字，设置需要保存flo文件的路径
        souname = (soufile.split("\\")[-1]).split(".")[0]
        proname = souname[:-8]
        numname = souname[-8:-3]
        aftname = souname[-3:]
        numname = np.int32(numname)
        numname = numname + 2
        numname_str = str(numname)
        wei = list(numname_str)
        neednum = [-2, -4]
        for i in neednum:
            if (i == -2 and wei[i] == '6'):
                numname += 40
                wei = list(str(numname))
            if (i == -4 and wei[i] == '6'):
                numname += 4000
        numname = str(numname)
        souname = proname + numname + aftname
        # if ((numname+2)%100) >= 60:
        #     numname = numname + 2 - 60 + 100
        #     if ((numname%1000)//100) == 9:

        rpath = renamepath + "/" + souname + ".jpg"
        cv2.imwrite(rpath, sou)
        count = count + 1
        print("已经完成 ", count, " 个文件的重命名")

    print("一共将 ", count, " 个jpg数据重命名jpg数据。")


root = tk.Tk()
root.withdraw()
soupath = filedialog.askdirectory()  # askdirectory() 获取选择的文件夹   askopenfilename() 获取选择的文件
renamepath = filedialog.askdirectory()  # 这种方式获取的绝对路径是用 "/"连接的
rename_img(soupath, renamepath)
