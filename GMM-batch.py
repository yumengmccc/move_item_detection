import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
from GMM_demo import optimized_motion_detection
from tqdm import tqdm, trange

def change_to_batch():
    # 进入到 batch 框架
    index_frame.pack_forget()
    batch_frame.pack()

def change_to_index():
    # 进入到 index 框架
    batch_frame.pack_forget()
    index_frame.pack()

    # 进行 batch 框架的初始化

def batch_operation():
    fn_list = filedialog.askdirectory()

    # 显示文件夹信息于 Text 框中
    batch_text.insert(tk.END, fn_list)

    video_path_list = []
    for fn in os.listdir(fn_list):
        if fn[-4:] == ".MP4": # 为 mp4 视频后缀，而过滤掉其他文件
            video_path_list.append(fn) # 视频完整路径

    # 实时进度条显示
    for idx, video in tqdm(enumerate(video_path_list)):
        optimized_motion_detection(os.path.join(fn_list,video), scale=0.25, result_path=f'./result/{video}/')

    print('finish')

def create_root():
    # 建立 index 框架
    tk.Label(index_frame, text='主菜单').grid(row=0, column=0, padx=5, pady=5)

    # 开始程序的按键实现
    begin_button = tk.Button(index_frame, text='开始选择')
    begin_button.grid(row=1, column=0, padx=5, pady=5)
    begin_button.config(command=change_to_batch)

    # 退出程序的按键实现
    quit_button = tk.Button(index_frame, text='退出程序')
    quit_button.grid(row=3, column=0, padx=5, pady=5)
    quit_button.config(command=root.quit)

    # 建立 batch 框架
    tk.Label(batch_frame, text='批处理菜单，请选择文件').grid(row=0, column=0, padx=5, pady=5)
    batch_text = tk.Text(batch_frame, width=30, height=1)
    batch_text.grid(row=1, column=0, padx=5, pady=5)

    choose_button = tk.Button(batch_frame, text='选择文件')
    choose_button.grid(row=1, column=1, padx=5, pady=5)
    choose_button.config(command=batch_operation)

    return begin_button, quit_button, batch_text, choose_button

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('500x300')
    root.title('GMM - batch')
    index_frame = tk.Frame(root)
    batch_frame = tk.Frame(root)

    # 创建 root 的框架
    begin_button, quit_button, batch_text, choose_button = create_root()

    # 初始化打开 index_frame
    index_frame.pack()
    root.mainloop()

