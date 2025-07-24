#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import serial
import serial.tools.list_ports
import threading
import time
from collections import Counter

class XceptionTrashClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("智能垃圾分类系统 - Xception测试版")
        self.root.geometry("1200x800")
        
        # 设备检测
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else 
                                  "cpu")
        print(f"使用设备: {self.device}")
        
        # 类别名称
        self.class_names = ['battery', 'Garbage', 'Organics', 'Recyclables']
        
        # 模型相关
        self.model = None
        self.model_loaded = False
        
        # 串口相关
        self.serial_port = None
        self.serial_connected = False
        
        # 图像预处理 - 针对 EfficientNet 优化
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="智能垃圾分类系统 - Xception版本", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 模型状态
        self.model_status_label = ttk.Label(control_frame, text="模型状态: 未加载", 
                                           foreground="red")
        self.model_status_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # 文件选择按钮
        ttk.Button(control_frame, text="选择图片", 
                  command=self.select_image).grid(row=1, column=0, pady=5, sticky=tk.W+tk.E)
        
        # 摄像头按钮
        ttk.Button(control_frame, text="打开摄像头", 
                  command=self.open_camera).grid(row=2, column=0, pady=5, sticky=tk.W+tk.E)
        
        ttk.Button(control_frame, text="关闭摄像头", 
                  command=self.close_camera).grid(row=3, column=0, pady=5, sticky=tk.W+tk.E)
        
        # 串口控制
        serial_frame = ttk.LabelFrame(control_frame, text="硬件控制", padding="5")
        serial_frame.grid(row=4, column=0, pady=(20, 0), sticky=tk.W+tk.E)
        
        # 串口选择
        ttk.Label(serial_frame, text="串口:").grid(row=0, column=0, sticky=tk.W)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(serial_frame, textvariable=self.port_var, width=15)
        self.port_combo.grid(row=0, column=1, padx=(5, 0))
        self.refresh_ports()
        
        ttk.Button(serial_frame, text="刷新", 
                  command=self.refresh_ports).grid(row=0, column=2, padx=(5, 0))
        
        # 连接按钮
        self.connect_btn = ttk.Button(serial_frame, text="连接", 
                                     command=self.toggle_connection)
        self.connect_btn.grid(row=1, column=0, columnspan=2, pady=(5, 0), sticky=tk.W+tk.E)
        
        # 手动控制按钮
        control_buttons_frame = ttk.Frame(serial_frame)
        control_buttons_frame.grid(row=2, column=0, columnspan=3, pady=(10, 0))
        
        ttk.Button(control_buttons_frame, text="电池", 
                  command=lambda: self.send_command('battery')).grid(row=0, column=0, padx=2)
        ttk.Button(control_buttons_frame, text="垃圾", 
                  command=lambda: self.send_command('garbage')).grid(row=0, column=1, padx=2)
        ttk.Button(control_buttons_frame, text="有机", 
                  command=lambda: self.send_command('organics')).grid(row=1, column=0, padx=2)
        ttk.Button(control_buttons_frame, text="可回收", 
                  command=lambda: self.send_command('recyclables')).grid(row=1, column=1, padx=2)
        
        # 中间显示区域
        display_frame = ttk.LabelFrame(main_frame, text="图像显示", padding="10")
        display_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 图像显示标签
        self.image_label = ttk.Label(display_frame, text="请选择图片或打开摄像头")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 右侧结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="分析结果", padding="10")
        result_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # 结果显示
        self.result_text = tk.Text(result_frame, height=10, width=30)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        # 置信度显示
        self.confidence_label = ttk.Label(result_frame, text="置信度: --")
        self.confidence_label.grid(row=1, column=0, pady=(10, 0), sticky=tk.W)
        
        # 底部图表区域
        chart_frame = ttk.LabelFrame(main_frame, text="分类概率分布", padding="10")
        chart_frame.grid(row=2, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0), pady=(10, 0))
        
        # 创建matplotlib图表
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 摄像头相关
        self.camera = None
        self.camera_running = False
        
        # 配置网格权重
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
    
    def load_model(self):
        """加载 train_xception.py 训练的模型"""
        try:
            # 专门查找 Xception 训练的模型文件
            model_files = [
                'best_model_garbage_xception.pth',
                'final_model_garbage_xception.pth'
            ]
            
            model_path = None
            for file in model_files:
                if os.path.exists(file):
                    model_path = file
                    break
            
            if model_path is None:
                self.log_message("错误: 未找到 Xception 模型文件")
                self.log_message("请先运行 train_xception.py 训练模型")
                return
            
            # 加载checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # train_xception.py 使用的是 EfficientNet-B0
            backbone = 'efficientnet_b0'
            
            # 获取类别数量（从模型权重推断）
            if 'model_state_dict' in checkpoint:
                # 从分类器权重推断类别数量
                classifier_key = None
                for key in checkpoint['model_state_dict'].keys():
                    if 'classifier' in key and 'weight' in key:
                        classifier_key = key
                        break
                
                if classifier_key:
                    num_classes = checkpoint['model_state_dict'][classifier_key].shape[0]
                    self.log_message(f"检测到类别数量: {num_classes}")
                else:
                    num_classes = len(self.class_names)
            else:
                num_classes = len(self.class_names)
            
            # 创建 EfficientNet-B0 模型（与 train_xception.py 一致）
            self.model = models.efficientnet_b0(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 更新类别名称（如果类别数量不匹配，使用默认名称）
            if num_classes == len(self.class_names):
                pass  # 使用默认类别名称
            else:
                self.log_message(f"警告: 类别数量不匹配，使用默认类别名称")
            
            self.model_loaded = True
            self.model_status_label.config(text=f"模型状态: 已加载 (EfficientNet-B0)", foreground="green")
            self.log_message(f"成功加载 Xception 模型: {model_path}")
            self.log_message(f"模型架构: EfficientNet-B0")
            self.log_message(f"类别数量: {num_classes}")
            self.log_message(f"类别: {self.class_names}")
            
            # 显示模型信息
            if 'val_accuracy' in checkpoint:
                self.log_message(f"验证准确率: {checkpoint['val_accuracy']:.2%}")
            if 'test_accuracy' in checkpoint:
                self.log_message(f"测试准确率: {checkpoint['test_accuracy']:.2%}")
            if 'epoch' in checkpoint:
                self.log_message(f"训练轮数: {checkpoint['epoch']}")
            
        except Exception as e:
            self.log_message(f"加载模型失败: {str(e)}")
            self.model_loaded = False
    
    def select_image(self):
        """选择图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.analyze_image(file_path)
    
    def analyze_image(self, image_path):
        """分析图片"""
        if not self.model_loaded:
            messagebox.showerror("错误", "模型未加载")
            return
        
        try:
            # 加载和预处理图片
            image = Image.open(image_path).convert('RGB')
            
            # 显示原图
            display_image = image.copy()
            display_image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(display_image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            # 预处理用于推理
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # 显示结果
            result = self.class_names[predicted_class]
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"分类结果: {result}\n")
            self.result_text.insert(tk.END, f"置信度: {confidence:.2%}\n\n")
            
            # 显示所有类别的概率
            self.result_text.insert(tk.END, "各类别概率:\n")
            for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities[0])):
                self.result_text.insert(tk.END, f"{class_name}: {prob:.2%}\n")
            
            self.confidence_label.config(text=f"置信度: {confidence:.2%}")
            
            # 更新图表
            self.update_chart(probabilities[0].cpu().numpy())
            
            # 发送命令到硬件
            self.send_command(result)
            
        except Exception as e:
            self.log_message(f"分析图片失败: {str(e)}")
    
    def update_chart(self, probabilities):
        """更新概率分布图表"""
        self.ax.clear()
        bars = self.ax.bar(self.class_names, probabilities, color='lightgreen')
        self.ax.set_ylabel('概率')
        self.ax.set_title('Xception 分类概率分布')
        self.ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.2%}', ha='center', va='bottom')
        
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def open_camera(self):
        """打开摄像头"""
        if self.camera_running:
            return
        
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("错误", "无法打开摄像头")
                return
            
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
        except Exception as e:
            self.log_message(f"打开摄像头失败: {str(e)}")
    
    def camera_loop(self):
        """摄像头循环"""
        while self.camera_running:
            ret, frame = self.camera.read()
            if ret:
                # 转换颜色空间
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 显示图像
                image = Image.fromarray(frame_rgb)
                display_image = image.copy()
                display_image.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(display_image)
                
                # 在主线程中更新UI
                self.root.after(0, lambda: self.update_camera_display(photo))
                
                # 分析图像
                if self.model_loaded:
                    self.analyze_camera_frame(frame_rgb)
            
            time.sleep(0.1)  # 控制帧率
    
    def update_camera_display(self, photo):
        """更新摄像头显示"""
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def analyze_camera_frame(self, frame):
        """分析摄像头帧"""
        try:
            # 预处理
            image = Image.fromarray(frame)
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # 如果置信度足够高，更新结果
            if confidence > 0.7:  # 阈值可调整
                result = self.class_names[predicted_class]
                self.root.after(0, lambda: self.update_camera_result(result, confidence, probabilities[0]))
        
        except Exception as e:
            pass  # 忽略摄像头分析错误
    
    def update_camera_result(self, result, confidence, probabilities):
        """更新摄像头分析结果"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"实时分类结果: {result}\n")
        self.result_text.insert(tk.END, f"置信度: {confidence:.2%}\n\n")
        
        self.confidence_label.config(text=f"置信度: {confidence:.2%}")
        self.update_chart(probabilities.cpu().numpy())
        
        # 发送命令到硬件
        self.send_command(result)
    
    def close_camera(self):
        """关闭摄像头"""
        self.camera_running = False
        if self.camera:
            self.camera.release()
        self.image_label.config(image="", text="摄像头已关闭")
    
    def refresh_ports(self):
        """刷新串口列表"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.set(ports[0])
    
    def toggle_connection(self):
        """切换串口连接状态"""
        if self.serial_connected:
            self.disconnect_serial()
        else:
            self.connect_serial()
    
    def connect_serial(self):
        """连接串口"""
        try:
            port = self.port_var.get()
            if not port:
                messagebox.showerror("错误", "请选择串口")
                return
            
            self.serial_port = serial.Serial(port, 9600, timeout=1)
            self.serial_connected = True
            self.connect_btn.config(text="断开")
            self.log_message(f"串口连接成功: {port}")
            
        except Exception as e:
            self.log_message(f"串口连接失败: {str(e)}")
    
    def disconnect_serial(self):
        """断开串口连接"""
        if self.serial_port:
            self.serial_port.close()
        self.serial_connected = False
        self.connect_btn.config(text="连接")
        self.log_message("串口连接已断开")
    
    def send_command(self, command):
        """发送命令到硬件"""
        if not self.serial_connected:
            return
        
        try:
            # 映射命令
            command_map = {
                'battery': '1',
                'garbage': '2', 
                'organics': '3',
                'recyclables': '4'
            }
            
            if command.lower() in command_map:
                cmd = command_map[command.lower()]
                self.serial_port.write(cmd.encode())
                self.log_message(f"发送命令: {command} -> {cmd}")
        
        except Exception as e:
            self.log_message(f"发送命令失败: {str(e)}")
    
    def log_message(self, message):
        """记录日志消息"""
        print(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.result_text.insert(tk.END, f"{message}\n")
        self.result_text.see(tk.END)

def main():
    root = tk.Tk()
    app = XceptionTrashClassifier(root)
    root.mainloop()

if __name__ == "__main__":
    main() 