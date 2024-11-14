import tkinter as tk
from tkinter import colorchooser
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageTk
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("绘画板")
        
        # 创建画布
        self.canvas = tk.Canvas(root, bg='white', width=600, height=400)
        self.canvas.pack()

        # 初始化变量
        self.last_x = None
        self.last_y = None
        self.pen_color = "black"
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # 完成按钮
        self.complete_button = tk.Button(root, text="完成", command=self.complete)
        self.complete_button.pack()

        # 颜色选择按钮
        self.color_button = tk.Button(root, text="选择颜色", command=self.choose_color)
        self.color_button.pack()

        # 创建一个空白图像用于保存绘图
        self.image = Image.new("RGB", (600, 400), "white")
        self.draw = ImageDraw.Draw(self.image)

        # 显示生成的图像
        self.result_label = tk.Label(root)
        self.result_label.pack()

    def choose_color(self):
        # 打开颜色选择器并更新画笔颜色
        color = colorchooser.askcolor()[1]
        if color:
            self.pen_color = color

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # 画线
            self.canvas.create_line((self.last_x, self.last_y, x, y), fill=self.pen_color, width=2)
            # 在图像上画线
            self.draw.line((self.last_x, self.last_y, x, y), fill=self.pen_color, width=2)
        self.last_x = x
        self.last_y = y

    def reset(self, event):
        # 重置最后的坐标
        self.last_x = None
        self.last_y = None

    def complete(self):
        # ControlNet
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        canny_image=self.image

        # 创建一个可以输入文字的地方，把prompt改成用户输入的文字
        prompt="the mona lisa"
        result_image = pipe(prompt, image=canny_image).images[0]

        result_image.save("output.jpg")

        self.display_result(result_image)

    def display_result(self, image):
        # 将PIL图像转换为PhotoImage格式
        image.thumbnail((600, 400))
        photo = ImageTk.PhotoImage(image)

        # 更新Label显示图像
        self.result_label.config(image=photo)
        self.result_label.image = photo  # 保持引用

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
