"""
Smart Trash Classification System
AI-Powered Waste Sorting Application with Hardware Integration
Version 1.0.0
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import serial
import serial.tools.list_ports
import time
import threading
from datetime import datetime
from pathlib import Path
import random
import glob
import queue

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

class TransferLearningModel(nn.Module):
    """AI Model for Trash Classification"""
    
    def __init__(self, num_classes=4, backbone='efficientnet_b0', use_pretrained=False):
        super(TransferLearningModel, self).__init__()
        
        # Create model based on backbone
        if backbone == 'resnet18':
            self.model = models.resnet18(weights=None if not use_pretrained else 'DEFAULT')
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif backbone == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=None if not use_pretrained else 'DEFAULT')
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        elif backbone == 'efficientnet_b3':
            self.model = models.efficientnet_b3(weights=None if not use_pretrained else 'DEFAULT')
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        elif backbone == 'mobilenet_v2':
            self.model = models.mobilenet_v2(weights=None if not use_pretrained else 'DEFAULT')
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        else:
            # Default to EfficientNet-B0
            self.model = models.efficientnet_b0(weights=None if not use_pretrained else 'DEFAULT')
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

class SmartTrashClassifier:
    """Main Application Class for Smart Trash Classification System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Trash Classification System v1.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Application configuration
        self.APP_NAME = "Smart Trash Classifier"
        self.VERSION = "1.0.0"
        
        # Classification categories - will be updated from model
        self.class_names = ['Garbage', 'Organics', 'Recyclables', 'battery']  # Default, will be updated from model
        self.class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        self.hardware_commands = {'Garbage': '2', 'Organics': '1', 'Recyclables': '3', 'battery': '4'}
        
        # System configuration
        self.device = self.get_device()
        self.model = None
        self.serial_conn = None
        
        # Dataset path for random testing
        self.dataset_path = "garbage_dataset"
        
        # Camera and prediction variables
        self.cap = None
        self.camera_active = False
        self.camera_thread = None
        self.queue = queue.Queue()
        
        # Photo mode variables
        self.photo_mode = False
        self.photo_cap = None
        self.photo_thread = None
        self.captured_frame = None
        
        # Prediction stability settings
        self.prediction_history = []
        self.stable_prediction = None
        self.stable_start_time = None
        self.stability_threshold = 1.5  # seconds
        self.confidence_threshold = 0.65  # 65% confidence required
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Current state
        self.current_image = None
        self.current_prediction = None
        
        # Initialize application
        self.setup_ui()
        self.load_model()
        self.connect_hardware()
        self.start_queue_processing()
        self.load_placeholder_image()
        
        # Application startup message
        self.log("Smart Trash Classification System started successfully")
        self.log(f"Version: {self.VERSION}")
        self.log(f"Device: {self.device}")
        
    def get_device(self):
        """Detect and return the best available computing device"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Configure root window
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left control panel
        self.setup_control_panel(main_frame)
        
        # Right display area
        self.setup_display_area(main_frame)
        
        # Status bar
        self.setup_status_bar()
    
    def setup_control_panel(self, parent):
        """Setup the left control panel"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="15")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))
        
        # Image input section
        input_frame = ttk.LabelFrame(control_frame, text="Image Input", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(input_frame, text="📁 Select Image File", 
                  command=self.load_image).pack(fill=tk.X, pady=2)
        
        self.camera_btn = ttk.Button(input_frame, text="📹 Start Live Camera", 
                                   command=self.toggle_camera)
        self.camera_btn.pack(fill=tk.X, pady=2)
        
        self.photo_btn = ttk.Button(input_frame, text="📸 Photo Mode", 
                                  command=self.toggle_photo_mode)
        self.photo_btn.pack(fill=tk.X, pady=2)
        
        # Photo capture button (initially hidden)
        self.capture_btn = ttk.Button(input_frame, text="📷 Capture Photo", 
                                    command=self.capture_photo, state=tk.DISABLED)
        self.capture_btn.pack(fill=tk.X, pady=2)
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(control_frame, text="AI Analysis", padding="10")
        analysis_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.analyze_btn = ttk.Button(analysis_frame, text="🔍 Analyze Image", 
                                    command=self.analyze_image, state=tk.DISABLED)
        self.analyze_btn.pack(fill=tk.X, pady=2)
        
        self.send_btn = ttk.Button(analysis_frame, text="📤 Send to Sorter", 
                                 command=self.send_to_hardware, state=tk.DISABLED)
        self.send_btn.pack(fill=tk.X, pady=2)
        
        # Random test button
        ttk.Button(analysis_frame, text="🎲 Random Test", 
                  command=self.start_random_test).pack(fill=tk.X, pady=2)
        
        # Auto-processing option
        self.auto_send_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(analysis_frame, text="🤖 Auto-Sort Mode", 
                       variable=self.auto_send_var).pack(fill=tk.X, pady=2)
        
        # Manual sorting section
        manual_frame = ttk.LabelFrame(control_frame, text="Manual Sorting", padding="10")
        manual_frame.pack(fill=tk.X, pady=(0, 10))
        
        for i, name in enumerate(self.class_names):
            command = self.hardware_commands[name]
            
            ttk.Button(manual_frame, 
                      text=name,
                      command=lambda cmd=command, n=name: self.manual_sort(cmd, n)).pack(fill=tk.X, pady=1)
        
        # System status section
        status_frame = ttk.LabelFrame(control_frame, text="System Status", padding="10")
        status_frame.pack(fill=tk.X)
        
        self.hardware_status = ttk.Label(status_frame, text="🔴 Hardware Disconnected", 
                                       foreground="red")
        self.hardware_status.pack(anchor=tk.W)
        
        ttk.Button(status_frame, text="Reconnect Hardware", 
                  command=self.connect_hardware).pack(fill=tk.X, pady=(5, 0))
    
    def setup_display_area(self, parent):
        """Setup the right display area"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Notebook for different views
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Image preview tab
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="📷 Image Preview")
        
        self.image_label = ttk.Label(self.image_frame, text="No image loaded", 
                                   anchor="center", font=('Arial', 12))
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        # Analysis results tab
        self.result_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.result_frame, text="📊 Analysis Results")
        self.setup_result_chart()
        
        # Activity log tab
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="📝 Activity Log")
        self.setup_activity_log()
    
    def setup_result_chart(self):
        """Setup the analysis results chart"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize empty charts
        self.ax1.text(0.5, 0.5, 'Ready for analysis...', ha='center', va='center', 
                     transform=self.ax1.transAxes, fontsize=14)
        self.ax1.set_title('Classification Result', fontsize=16, fontweight='bold')
        self.ax1.axis('off')
        
        self.ax2.text(0.5, 0.5, 'Confidence levels will appear here', ha='center', va='center', 
                     transform=self.ax2.transAxes, fontsize=12)
        self.ax2.set_title('Confidence Distribution', fontsize=16, fontweight='bold')
        self.ax2.axis('off')
    
    def setup_activity_log(self):
        """Setup the activity log"""
        log_container = ttk.Frame(self.log_frame)
        log_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = tk.Text(log_container, height=20, wrap=tk.WORD, 
                               font=('Consolas', 10), bg='#f8f8f8')
        log_scrollbar = ttk.Scrollbar(log_container, orient="vertical", 
                                     command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_status_bar(self):
        """Setup the bottom status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=(0, 10))
        
        self.status_text = ttk.Label(self.status_bar, text=f"{self.APP_NAME} v{self.VERSION} - Ready")
        self.status_text.pack(side=tk.LEFT)
    
    def load_model(self):
        """Load the AI classification model"""
        try:
            # Search for model files
            model_files = [
                'best_model_garbage_resnet18_adam_cross_entropy.pth',
                'final_model_garbage_resnet18_adam_cross_entropy.pth',
                'best_model_garbage_resnet18_adamw_cross_entropy.pth',
                'final_model_garbage_resnet18_adamw_cross_entropy.pth',
                'best_model_garbage_resnet18_sgd_cross_entropy.pth',
                'final_model_garbage_resnet18_sgd_cross_entropy.pth',
                'best_model_garbage_xception.pth',
                'final_model_garbage_xception.pth',
                'best_model_garbage_efficientnet_b0.pth',
                'final_model_garbage_efficientnet_b0.pth',
                'best_model_garbage_efficientnet_b3.pth',
                'final_model_garbage_efficientnet_b3.pth',
                'best_model_mobilenet_v2_adam_cross_entropy.pth',
                'final_model_mobilenet_v2_adam_cross_entropy.pth',
                'best_model_xception_pytorch.pth',
                'final_model_garbage_xception_pytorch.pth',
                'best_model_garbage_simplified_xception_pytorch.pth',
                'final_model_garbage_simplified_xception_pytorch.pth'
            ]
            
            model_path = None
            for file in model_files:
                if os.path.exists(file):
                    model_path = file
                    break
            
            if not model_path:
                raise FileNotFoundError("AI model file not found")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Determine backbone from model path
            if 'xception' in model_path.lower() and 'pytorch' in model_path.lower():
                if 'simplified' in model_path.lower():
                    backbone = 'simplified_xception'
                else:
                    backbone = 'xception'
            elif 'xception' in model_path.lower():
                backbone = 'efficientnet_b0'  # train_xception.py actually uses EfficientNet
            elif 'efficientnet_b0' in model_path.lower():
                backbone = 'efficientnet_b0'
            elif 'efficientnet_b3' in model_path.lower():
                backbone = 'efficientnet_b3'
            elif 'mobilenet' in model_path.lower():
                backbone = 'mobilenet_v2'
            elif 'resnet18' in model_path.lower():
                backbone = 'resnet18'
            else:
                backbone = checkpoint.get('backbone', 'resnet18')  # Fallback
            
            # Get class names from checkpoint or use default
            if 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']
            else:
                # Infer from model state dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    # Find the classifier layer to determine num_classes
                    for key in state_dict.keys():
                        if 'classifier' in key or 'fc' in key:
                            if 'weight' in key:
                                num_classes = state_dict[key].shape[0]
                                break
                    else:
                        num_classes = 4  # Default fallback
                else:
                    num_classes = 4  # Default fallback
                
                # Use default class names
                self.class_names = ['Garbage', 'Organics', 'Recyclables', 'battery']
            
            # Update hardware commands based on class names
            self.hardware_commands = {}
            for i, class_name in enumerate(self.class_names):
                if class_name == 'Garbage':
                    self.hardware_commands[class_name] = '2'
                elif class_name == 'Organics':
                    self.hardware_commands[class_name] = '1'
                elif class_name == 'Recyclables':
                    self.hardware_commands[class_name] = '3'
                elif class_name == 'battery':
                    self.hardware_commands[class_name] = '4'
                else:
                    # Default mapping for unknown classes
                    self.hardware_commands[class_name] = str(i + 1)
            
            # Create model based on backbone
            if backbone in ['xception', 'simplified_xception']:
                # These are custom models from pytorch_train.py
                self.log(f"ERROR: Custom Xception models require Xception_test.py")
                messagebox.showerror("Model Error", 
                    f"Model {model_path} is a custom Xception model.\n"
                    f"Please use Xception_test.py to load this model.")
                return
            
            # Create torchvision model
            if backbone == 'resnet18':
                self.model = models.resnet18(weights=None)
                self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
            elif backbone == 'efficientnet_b0':
                self.model = models.efficientnet_b0(weights=None)
                self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(self.class_names))
            elif backbone == 'efficientnet_b3':
                self.model = models.efficientnet_b3(weights=None)
                self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(self.class_names))
            elif backbone == 'mobilenet_v2':
                self.model = models.mobilenet_v2(weights=None)
                self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(self.class_names))
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
            
            # Load model state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                accuracy = checkpoint.get('accuracy', checkpoint.get('test_accuracy', 'N/A'))
                epoch = checkpoint.get('epoch', 'N/A')
                self.log(f"AI model loaded: {model_path}")
                self.log(f"Model architecture: {backbone}")
                self.log(f"Classes: {self.class_names}")
                self.log(f"Model accuracy: {accuracy}")
                self.log(f"Training epoch: {epoch}")
            else:
                self.model.load_state_dict(checkpoint)
                self.log(f"AI model loaded: {model_path}")
                self.log(f"Model architecture: {backbone}")
                self.log(f"Classes: {self.class_names}")
            
            self.model.eval()
            self.model = self.model.to(self.device)
            self.status_text.config(text=f"{self.APP_NAME} - AI Model Ready")
            
        except Exception as e:
            self.log(f"ERROR: Failed to load AI model - {e}")
            messagebox.showerror("Model Error", f"Failed to load AI model:\n{e}")
    
    def connect_hardware(self):
        """Connect to hardware sorting system"""
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            
            # Find hardware port
            port = self.find_hardware_port()
            if not port:
                raise Exception("Hardware not found")
            
            self.serial_conn = serial.Serial(port, 115200, timeout=1)
            time.sleep(2)  # Wait for connection
            
            self.hardware_status.config(text="🟢 Hardware Connected", foreground="green")
            self.log(f"Hardware connected: {port}")
            
        except Exception as e:
            self.hardware_status.config(text="🔴 Hardware Disconnected", foreground="red")
            self.log(f"Hardware connection failed: {e}")
    
    def find_hardware_port(self):
        """Find the hardware communication port"""
        # Primary port (adjust as needed)
        specific_port = "/dev/cu.usbmodem21101"
        if os.path.exists(specific_port):
            return specific_port
        
        # Search for compatible ports
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if 'usbmodem' in port.device or 'USB' in port.description:
                return port.device
        
        return None
    
    def load_image(self):
        """Load an image file for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image = Image.open(file_path).convert('RGB')
                self.display_image(self.current_image)
                self.analyze_btn.config(state=tk.NORMAL)
                self.log(f"Image loaded: {os.path.basename(file_path)}")
            except Exception as e:
                self.log(f"ERROR: Failed to load image - {e}")
                messagebox.showerror("Image Error", f"Failed to load image:\n{e}")
    
    def display_image(self, image):
        """Display image in the preview area"""
        # Resize for display while maintaining aspect ratio
        display_size = (700, 500)
        
        # Calculate scaling
        img_width, img_height = image.size
        scale = min(display_size[0] / img_width, display_size[1] / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        
        # Resize image
        display_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create canvas with centered image
        canvas = Image.new('RGB', display_size, color='#f0f0f0')
        paste_pos = ((display_size[0] - new_size[0]) // 2, 
                    (display_size[1] - new_size[1]) // 2)
        canvas.paste(display_image, paste_pos)
        
        # Convert to PhotoImage and display
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(canvas)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep reference
        
        # Switch to image tab
        self.notebook.select(0)
    
    def analyze_image(self):
        """Analyze the current image using AI"""
        if not self.current_image:
            self.log("ERROR: No image loaded for analysis")
            return
        
        if not self.model:
            self.log("ERROR: AI model not loaded")
            messagebox.showerror("Model Error", "AI model not loaded. Please restart the application.")
            return
        
        try:
            # Preprocess image
            input_tensor = self.transform(self.current_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Run AI inference
            with torch.no_grad():
                outputs = self.model(input_batch)
                probabilities = F.softmax(outputs[0], dim=0).cpu().numpy()
                predicted_idx = np.argmax(probabilities)
            
            # Store prediction results
            self.current_prediction = {
                'class_idx': predicted_idx,
                'class_name': self.class_names[predicted_idx],
                'probabilities': {self.class_names[i]: float(probabilities[i]) 
                                for i in range(len(self.class_names))},
                'confidence': probabilities[predicted_idx]
            }
            
            # Display results
            self.display_analysis_results()
            self.send_btn.config(state=tk.NORMAL)
            
            # Log results
            class_name = self.current_prediction['class_name']
            confidence = self.current_prediction['confidence']
            self.log(f"Analysis complete: {class_name} ({confidence:.1%} confidence)")
            
            # For uploaded images, auto-send regardless of confidence
            if self.auto_send_var.get():
                self.log("Auto-sort triggered for uploaded image (no confidence threshold)")
                self.send_to_hardware()
            
        except Exception as e:
            self.log(f"ERROR: Analysis failed - {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            messagebox.showerror("Analysis Error", f"Analysis failed:\n{e}")
    
    def display_analysis_results(self):
        """Display the analysis results in charts"""
        if not self.current_prediction:
            return
        
        pred = self.current_prediction
        
        # Clear previous results
        self.ax1.clear()
        self.ax2.clear()
        
        # Main result display
        class_name = pred['class_name']
        confidence = pred['confidence']
        
        result_text = f"{class_name}\n\n{confidence:.1%} Confidence"
        color = self.class_colors[pred['class_idx']]
        
        self.ax1.text(0.5, 0.5, result_text, ha='center', va='center',
                     transform=self.ax1.transAxes, fontsize=18, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        self.ax1.set_title('Classification Result', fontsize=16, fontweight='bold')
        self.ax1.axis('off')
        
        # Confidence distribution
        prob_values = [pred['probabilities'][name] for name in self.class_names]
        colors = [self.class_colors[i] for i in range(len(self.class_names))]
        
        bars = self.ax2.barh(range(len(self.class_names)), prob_values, color=colors, alpha=0.7)
        self.ax2.set_yticks(range(len(self.class_names)))
        self.ax2.set_yticklabels(self.class_names)
        self.ax2.set_xlabel('Confidence Level')
        self.ax2.set_title('All Categories Confidence', fontsize=16, fontweight='bold')
        self.ax2.set_xlim(0, 1)
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            self.ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                         f'{prob_values[i]:.1%}', va='center', fontweight='bold')
        
        # Highlight the predicted class
        bars[pred['class_idx']].set_alpha(1.0)
        bars[pred['class_idx']].set_edgecolor('black')
        bars[pred['class_idx']].set_linewidth(2)
        
        plt.tight_layout()
        self.canvas.draw()
        
        # Switch to results tab
        self.notebook.select(1)
    
    def send_to_hardware(self):
        """Send classification result to hardware sorting system"""
        if not self.current_prediction:
            return
        
        class_name = self.current_prediction['class_name']
        command = self.hardware_commands[class_name]
        
        self.send_hardware_command(command, f"AI Classification: {class_name}")
    
    def manual_sort(self, command, category_name):
        """Manually trigger sorting for a specific category"""
        self.send_hardware_command(command, f"Manual Sort: {category_name}")
    
    def send_hardware_command(self, command, description):
        """Send command to hardware sorting system"""
        if not self.serial_conn or not self.serial_conn.is_open:
            self.log("ERROR: Hardware not connected")
            messagebox.showwarning("Hardware Error", "Hardware not connected")
            return
        
        try:
            self.serial_conn.write(command.encode('utf-8'))
            
            # Get category information
            idx = int(command) - 1
            if 0 <= idx < len(self.class_names):
                category = self.class_names[idx]
                action = "Sorting mechanism activated" if command != '4' else "Special handling mode"
            else:
                category = "Unknown"
                action = "Unknown action"
            
            self.log(f"✅ {description}")
            self.log(f"   Command: {command} → {category}")
            self.log(f"   Action: {action}")
            
            # Show completion in separate thread
            threading.Thread(target=self.wait_for_hardware_completion, 
                           args=(command, category), daemon=True).start()
            
        except Exception as e:
            self.log(f"ERROR: Command failed - {e}")
            messagebox.showerror("Hardware Error", f"Command failed:\n{e}")
    
    def wait_for_hardware_completion(self, command, category):
        """Wait for hardware operation to complete"""
        if command != '4':  # Regular sorting operations
            time.sleep(3)
            self.log(f"   ✅ Sorting complete: {category}")
        else:  # Special handling for hazardous waste
            time.sleep(1)
            self.log(f"   ⚠️ Hazardous waste flagged for special handling")
    
    def toggle_photo_mode(self):
        """Toggle photo mode on/off"""
        if self.photo_mode:
            self.stop_photo_mode()
        else:
            self.start_photo_mode()
    
    def start_photo_mode(self):
        """Start photo mode with live preview"""
        try:
            # Make sure camera mode is off
            if self.camera_active:
                self.stop_camera()
            
            self.photo_cap = cv2.VideoCapture(0)
            
            if not self.photo_cap.isOpened():
                self.log("ERROR: Failed to access camera for photo mode")
                return
            
            self.photo_mode = True
            self.photo_btn.config(text="📸 Stop Photo Mode")
            self.capture_btn.config(state=tk.NORMAL)
            self.log("Photo mode started - Position object in analysis box and click Capture")
            
            # Start photo preview thread
            self.photo_thread = threading.Thread(target=self.photo_preview_loop, daemon=True)
            self.photo_thread.start()
            
        except Exception as e:
            self.log(f"ERROR: Photo mode start failed - {e}")
            messagebox.showerror("Photo Error", f"Photo mode start failed:\n{e}")
    
    def stop_photo_mode(self):
        """Stop photo mode"""
        self.photo_mode = False
        self.photo_btn.config(text="📸 Photo Mode")
        self.capture_btn.config(state=tk.DISABLED)
        
        if self.photo_cap is not None and self.photo_cap.isOpened():
            self.photo_cap.release()
        
        self.log("Photo mode stopped")
        self.load_placeholder_image()
    
    def photo_preview_loop(self):
        """Photo mode preview loop"""
        try:
            while self.photo_mode:
                ret, frame = self.photo_cap.read()
                
                if not ret:
                    self.queue.put(("error", "Photo preview failed"))
                    break
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                # Add analysis box overlay
                annotated_frame = self.draw_photo_overlay(frame)
                
                # Store current frame for capture
                self.captured_frame = frame.copy()
                
                # Update UI
                self.queue.put(("photo_frame", annotated_frame))
                
                time.sleep(0.1)  # Limit to ~10 FPS
        
        except Exception as e:
            self.queue.put(("error", f"Photo preview error: {str(e)}"))
        finally:
            if self.photo_cap is not None and self.photo_cap.isOpened():
                self.photo_cap.release()
            self.queue.put(("photo_stopped", None))
    
    def draw_photo_overlay(self, frame):
        """Draw analysis box overlay for photo mode"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        roi_size = min(w, h) // 3
        
        # Draw analysis box
        color = (0, 255, 0)  # Green box
        cv2.rectangle(frame, 
                     (center_x - roi_size, center_y - roi_size),
                     (center_x + roi_size, center_y + roi_size), 
                     color, 3)
        
        # Add instruction text
        instruction_text = "Position object in green box, then click Capture"
        text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (10, 10), (text_size[0] + 20, 50), (0, 0, 0), -1)
        cv2.putText(frame, instruction_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    
    def capture_photo(self):
        """Capture photo from current preview and analyze"""
        if not self.photo_mode or self.captured_frame is None:
            self.log("ERROR: No photo preview available")
            return
        
        try:
            # Extract analysis region from captured frame
            h, w = self.captured_frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            roi_size = min(w, h) // 3
            
            # Extract ROI (Region of Interest)
            roi = self.captured_frame[center_y - roi_size:center_y + roi_size,
                                   center_x - roi_size:center_x + roi_size]
            
            # Convert to PIL Image
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(roi_rgb)
            
            # Stop photo mode
            self.stop_photo_mode()
            
            # Display captured image
            self.display_image(self.current_image)
            
            # Enable analyze button and auto-analyze
            self.analyze_btn.config(state=tk.NORMAL)
            self.log("📸 Photo captured from analysis box")
            
            # Auto-analyze the captured region
            self.analyze_captured_photo()
            
        except Exception as e:
            self.log(f"ERROR: Photo capture failed - {e}")
            messagebox.showerror("Capture Error", f"Photo capture failed:\n{e}")
    
    def analyze_captured_photo(self):
        """Analyze the captured photo region"""
        if not self.current_image or not self.model:
            return
        
        try:
            # Preprocess image
            input_tensor = self.transform(self.current_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Run AI inference
            with torch.no_grad():
                outputs = self.model(input_batch)
                probabilities = F.softmax(outputs[0], dim=0).cpu().numpy()
                predicted_idx = np.argmax(probabilities)
            
            # Store prediction results
            self.current_prediction = {
                'class_idx': predicted_idx,
                'class_name': self.class_names[predicted_idx],
                'probabilities': {self.class_names[i]: float(probabilities[i]) 
                                for i in range(len(self.class_names))},
                'confidence': probabilities[predicted_idx]
            }
            
            # Display results
            self.display_analysis_results()
            self.send_btn.config(state=tk.NORMAL)
            
            # Log results
            class_name = self.current_prediction['class_name']
            confidence = self.current_prediction['confidence']
            self.log(f"Captured photo analysis: {class_name} ({confidence:.1%} confidence)")
            
            # For captured photos, auto-send regardless of confidence
            if self.auto_send_var.get():
                self.log("Auto-sort triggered for captured photo (no confidence threshold)")
                self.send_to_hardware()
            
        except Exception as e:
            self.log(f"ERROR: Photo analysis failed - {e}")
            messagebox.showerror("Analysis Error", f"Photo analysis failed:\n{e}")
    
    def toggle_camera(self):
        """Toggle live camera feed on/off"""
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start live camera feed"""
        try:
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                self.log("ERROR: Failed to access camera")
                return
            
            self.camera_active = True
            self.camera_btn.config(text="📹 Stop Camera")
            self.log("Live camera started")
            
            # Reset prediction tracking
            self.prediction_history = []
            self.stable_prediction = None
            self.stable_start_time = None
            
            # Start camera processing thread
            self.camera_thread = threading.Thread(target=self.camera_processing_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            self.log(f"ERROR: Camera start failed - {e}")
            messagebox.showerror("Camera Error", f"Camera start failed:\n{e}")
    
    def stop_camera(self):
        """Stop live camera feed"""
        self.camera_active = False
        self.camera_btn.config(text="📹 Start Live Camera")
        
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        # Reset tracking
        self.prediction_history = []
        self.stable_prediction = None
        self.stable_start_time = None
        
        self.log("Live camera stopped")
        self.load_placeholder_image()
    
    def camera_processing_loop(self):
        """Main camera processing loop"""
        try:
            while self.camera_active:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.queue.put(("error", "Camera frame capture failed"))
                    break
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                # Extract center region for analysis
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                roi_size = min(w, h) // 3
                
                roi = frame[center_y - roi_size:center_y + roi_size,
                           center_x - roi_size:center_x + roi_size]
                
                # Run AI analysis on ROI
                if self.model:
                    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    input_tensor = self.transform(roi_pil)
                    input_batch = input_tensor.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model(input_batch)
                        probabilities = F.softmax(outputs[0], dim=0).cpu().numpy()
                        predicted_idx = np.argmax(probabilities)
                    
                    # Create prediction result
                    prediction = {
                        'class_idx': predicted_idx,
                        'class_name': self.class_names[predicted_idx],
                        'probabilities': {self.class_names[i]: float(probabilities[i])
                                        for i in range(len(self.class_names))},
                        'confidence': probabilities[predicted_idx]
                    }
                    
                    # Draw overlay on frame
                    annotated_frame = self.draw_camera_overlay(frame, prediction, center_x, center_y, roi_size)
                    
                    # Update UI
                    self.queue.put(("camera_frame", annotated_frame))
                    self.queue.put(("camera_prediction", prediction))
                    
                    # Check for stable prediction
                    self.check_prediction_stability(prediction)
                
                time.sleep(0.1)  # Limit to ~10 FPS
        
        except Exception as e:
            self.queue.put(("error", f"Camera processing error: {str(e)}"))
        finally:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.queue.put(("camera_stopped", None))
    
    def draw_camera_overlay(self, frame, prediction, center_x, center_y, roi_size):
        """Draw prediction overlay on camera frame"""
        # Convert to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw analysis box
        color = (0, 255, 0) if prediction['confidence'] > self.confidence_threshold else (255, 255, 0)
        cv2.rectangle(frame, 
                     (center_x - roi_size, center_y - roi_size),
                     (center_x + roi_size, center_y + roi_size), 
                     color, 3)
        
        # Add prediction text
        class_name = prediction['class_name']
        confidence = prediction['confidence']
        
        text = f"{class_name}: {confidence:.1%}"
        
        # Text background
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (10, 10), (text_size[0] + 20, 50), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Stability indicator
        if self.stable_prediction and self.stable_start_time:
            elapsed = time.time() - self.stable_start_time
            if elapsed < self.stability_threshold:
                stability_text = f"Stabilizing: {elapsed:.1f}s"
                cv2.putText(frame, stability_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "STABLE - AUTO-SORTING", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    def check_prediction_stability(self, prediction):
        """Check if prediction is stable enough for auto-sorting"""
        current_time = time.time()
        class_name = prediction['class_name']
        confidence = prediction['confidence']
        
        # Only consider high-confidence predictions
        if confidence < self.confidence_threshold:
            self.reset_stability_tracking()
            return
        
        # Add to history
        self.prediction_history.append({
            'class_name': class_name,
            'confidence': confidence,
            'timestamp': current_time
        })
        
        # Keep only recent predictions
        self.prediction_history = [p for p in self.prediction_history 
                                 if current_time - p['timestamp'] <= 5.0]
        
        # Check stability
        if len(self.prediction_history) >= 15:  # ~1.5 seconds of consistent predictions
            recent_classes = [p['class_name'] for p in self.prediction_history[-15:]]
            
            if all(cls == class_name for cls in recent_classes):
                if self.stable_prediction != class_name:
                    # New stable prediction
                    self.stable_prediction = class_name
                    self.stable_start_time = current_time
                    self.log(f"Stable prediction detected: {class_name}")
                else:
                    # Check if stability threshold is reached
                    if current_time - self.stable_start_time >= self.stability_threshold:
                        # Auto-sort
                        self.trigger_auto_sort(prediction)
                        self.reset_stability_tracking()
            else:
                self.reset_stability_tracking()
        else:
            if self.stable_prediction and class_name != self.stable_prediction:
                self.reset_stability_tracking()
    
    def reset_stability_tracking(self):
        """Reset prediction stability tracking"""
        if self.stable_prediction:
            self.stable_prediction = None
            self.stable_start_time = None
    
    def trigger_auto_sort(self, prediction):
        """Trigger automatic sorting based on stable prediction"""
        class_name = prediction['class_name']
        confidence = prediction['confidence']
        
        self.log(f"🚀 Auto-sort triggered: {class_name} ({confidence:.1%})")
        
        # Update current prediction for hardware sending
        self.current_prediction = prediction
        
        # Send to hardware if auto-sort is enabled
        if self.auto_send_var.get():
            command = self.hardware_commands[class_name]
            self.send_hardware_command(command, f"Auto-Sort: {class_name}")
    
    def start_queue_processing(self):
        """Start processing the update queue"""
        try:
            self.process_queue()
        except Exception as e:
            print(f"Queue processing error: {e}")
    
    def process_queue(self):
        """Process updates from camera thread"""
        try:
            while True:
                message_type, data = self.queue.get_nowait()
                
                if message_type == "camera_frame":
                    self.display_image(data)
                elif message_type == "photo_frame":
                    self.display_image(data)
                elif message_type == "error":
                    self.log(f"ERROR: {data}")
                elif message_type == "camera_stopped":
                    self.camera_active = False
                    self.camera_btn.config(text="📹 Start Live Camera")
                elif message_type == "photo_stopped":
                    self.photo_mode = False
                    self.photo_btn.config(text="📸 Photo Mode")
                    self.capture_btn.config(state=tk.DISABLED)
        
        except queue.Empty:
            pass
        
        # Schedule next check - use a safer method
        try:
            self.root.after(50, self.process_queue)
        except tk.TclError:
            # Window was destroyed, stop processing
            pass
    
    def load_placeholder_image(self):
        """Load placeholder when no image is selected"""
        placeholder = Image.new('RGB', (700, 500), color='#e8e8e8')
        self.display_image(placeholder)
    
    def log(self, message):
        """Add entry to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Update UI in main thread
        self.root.after(0, self._update_log, log_entry)
    
    def _update_log(self, message):
        """Update the log display"""
        try:
            self.log_text.insert(tk.END, message)
            self.log_text.see(tk.END)
        except tk.TclError:
            # Widget was destroyed, ignore
            pass
    
    def get_random_training_images(self, num_images=1):
        """Get random images from training dataset"""
        all_images = []
        class_counts = {}
        
        train_path = os.path.join(self.dataset_path, 'TRAIN')
        if not os.path.exists(train_path):
            self.log(f"Dataset path not found: {train_path}")
            return []
        
        # Traverse all class directories
        for class_dir in os.listdir(train_path):
            class_path = os.path.join(train_path, class_dir)
            if os.path.isdir(class_path):
                # Get all images in this class
                images = glob.glob(os.path.join(class_path, '*.jpg')) + \
                        glob.glob(os.path.join(class_path, '*.jpeg')) + \
                        glob.glob(os.path.join(class_path, '*.png'))
                
                if len(images) > 0:
                    class_counts[class_dir] = len(images)
                    all_images.extend([(img, class_dir) for img in images])
        
        if not all_images:
            self.log("No images found in dataset")
            return []
        
        # Randomly select specified number of images
        if len(all_images) > num_images:
            # Try to select images evenly from each class
            selected_images = []
            images_per_class = max(1, num_images // len(class_counts))
            
            # Organize images by class
            class_images = {}
            for img_path, class_name in all_images:
                if class_name not in class_images:
                    class_images[class_name] = []
                class_images[class_name].append((img_path, class_name))
            
            # Randomly select from each class
            for class_name, imgs in class_images.items():
                selected_count = min(images_per_class, len(imgs))
                selected_images.extend(random.sample(imgs, selected_count))
            
            # If still not enough, select more randomly
            if len(selected_images) < num_images:
                remaining = all_images
                for img in selected_images:
                    if img in remaining:
                        remaining.remove(img)
                additional = random.sample(remaining, min(num_images - len(selected_images), len(remaining)))
                selected_images.extend(additional)
            
            # If too many, randomly remove some
            if len(selected_images) > num_images:
                selected_images = random.sample(selected_images, num_images)
                
            return selected_images
        return all_images
    
    def start_random_test(self):
        """Start random test with a randomly selected image"""
        self.log("🎲 Selecting random image for testing...")
        
        # Get random image - select only 1
        test_images = self.get_random_training_images(num_images=1)
        
        if not test_images:
            self.log("❌ No images found in dataset")
            messagebox.showwarning("Warning", f"No images found in path {self.dataset_path}")
            return
        
        # Get selected image
        img_path, true_class = test_images[0]
        
        try:
            # Load image
            self.current_image = Image.open(img_path).convert('RGB')
            
            if self.current_image is None:
                self.log(f"❌ Failed to load image: {img_path}")
                return
            
            # Automatically analyze image
            self.analyze_image_for_random_test()
            
            # Get prediction results
            if self.current_prediction:
                # Draw prediction results on image
                annotated_image = self.draw_prediction_on_image(self.current_image, self.current_prediction)
                
                # Display image with prediction results
                self.display_image(annotated_image)
                
                predicted_class = self.current_prediction['class_name']
                confidence = self.current_prediction['confidence']
                is_correct = (predicted_class == true_class)
                
                # Log test results
                status = "✅ Correct" if is_correct else "❌ Incorrect"
                self.log(f"Random test result: {status}")
                self.log(f"   File: {os.path.basename(img_path)}")
                self.log(f"   True class: {true_class}")
                self.log(f"   Predicted: {predicted_class} (confidence: {confidence:.1%})")
                
                # If prediction is correct and high confidence, suggest hardware sending
                if is_correct and confidence > 0.8:
                    self.log(f"   ✨ High confidence correct prediction, ready for hardware")
                
                # Enable send to hardware button
                self.send_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.log(f"❌ Error processing image: {str(e)}")
    
    def analyze_image_for_random_test(self):
        """Analyze image for random test without displaying result charts"""
        if not self.current_image or not self.model:
            return
        
        try:
            # Preprocess image
            input_tensor = self.transform(self.current_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Run AI inference
            with torch.no_grad():
                outputs = self.model(input_batch)
                probabilities = F.softmax(outputs[0], dim=0).cpu().numpy()
                predicted_idx = np.argmax(probabilities)
            
            # Store prediction results
            self.current_prediction = {
                'class_idx': predicted_idx,
                'class_name': self.class_names[predicted_idx],
                'probabilities': {self.class_names[i]: float(probabilities[i]) 
                                for i in range(len(self.class_names))},
                'confidence': probabilities[predicted_idx]
            }
            
            # Log results
            class_name = self.current_prediction['class_name']
            confidence = self.current_prediction['confidence']
            self.log(f"Analysis complete: {class_name} (confidence: {confidence:.1%})")
            
            # For random test, auto-send regardless of confidence
            if self.auto_send_var.get():
                self.log("Auto-sort triggered for random test (no confidence threshold)")
                self.send_to_hardware()
            
        except Exception as e:
            self.log(f"ERROR: Random test failed - {e}")
            messagebox.showerror("Error", f"Random test failed:\n{e}")
    
    def draw_prediction_on_image(self, image, result):
        """Draw prediction results on image - optimized version"""
        # Convert to OpenCV format
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Copy image to avoid modifying original
        frame = image.copy()
        h, w = frame.shape[:2]
        
        # Add class name and confidence - using larger font
        class_name = result['class_name']
        confidence = result['confidence']
        main_text = f"{class_name}"
        confidence_text = f"Confidence: {confidence:.1%}"
        
        # Set color based on confidence
        if confidence > 0.7:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.4:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Main title area - top center
        main_font_scale = 1.2
        conf_font_scale = 0.8
        
        # Get text size
        main_text_size, _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, main_font_scale, 3)
        conf_text_size, _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, conf_font_scale, 2)
        
        # Calculate text position (top center)
        main_x = (w - main_text_size[0]) // 2
        main_y = 50
        conf_x = (w - conf_text_size[0]) // 2
        conf_y = main_y + main_text_size[1] + 10
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add main text
        cv2.putText(frame, main_text, (main_x, main_y), cv2.FONT_HERSHEY_SIMPLEX, main_font_scale, color, 3)
        cv2.putText(frame, confidence_text, (conf_x, conf_y), cv2.FONT_HERSHEY_SIMPLEX, conf_font_scale, (255, 255, 255), 2)
        
        # Add rectangle for prediction area
        center_x, center_y = w // 2, h // 2
        box_size = min(w, h) // 4
        
        # Draw rectangle
        cv2.rectangle(frame, 
                     (center_x - box_size, center_y - box_size), 
                     (center_x + box_size, center_y + box_size), 
                     color, 3)
        
        # Add "Analysis Area" label
        area_text = "Analysis Area"
        area_text_size, _ = cv2.getTextSize(area_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        area_x = center_x - area_text_size[0] // 2
        area_y = center_y + box_size + 30
        
        # Draw label background
        cv2.rectangle(frame, (area_x - 5, area_y - area_text_size[1] - 5), 
                     (area_x + area_text_size[0] + 5, area_y + 5), (0, 0, 0), -1)
        cv2.putText(frame, area_text, (area_x, area_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Right side probability bar chart - simplified version
        if w > 400:  # Only display when image is wide enough
            bar_start_x = w - 250
            bar_start_y = 150
            bar_height = 25
            max_bar_width = 120
            
            # Draw background
            cv2.rectangle(frame, (bar_start_x - 10, bar_start_y - 20), 
                         (w - 10, bar_start_y + len(self.class_names) * (bar_height + 15) + 10), 
                         (0, 0, 0), -1)
            
            # Title
            cv2.putText(frame, "Probabilities", (bar_start_x, bar_start_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Sort categories by probability
            sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
            
            for i, (class_name_prob, prob) in enumerate(sorted_probs):
                y_pos = bar_start_y + i * (bar_height + 15) + 30
                
                # Draw class name
                cv2.putText(frame, class_name_prob[:8], (bar_start_x, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw probability bar
                bar_width = int(prob * max_bar_width)
                prob_color = color if class_name_prob == class_name else (100, 100, 100)
                cv2.rectangle(frame, (bar_start_x + 80, y_pos - 15), 
                             (bar_start_x + 80 + bar_width, y_pos), prob_color, -1)
                
                # Show probability value
                cv2.putText(frame, f"{prob:.1%}", (bar_start_x + 80 + bar_width + 5, y_pos - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Convert OpenCV BGR image to PIL RGB image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        return pil_image

def main():
    """Main application entry point"""
    try:
        root = tk.Tk()
        
        # Set application icon and properties
        root.iconname("Smart Trash Classifier")
        
        # Create and run application
        app = SmartTrashClassifier(root)
        
        try:
            root.mainloop()
        except KeyboardInterrupt:
            print("Application terminated by user")
        except Exception as e:
            print(f"Application error: {e}")
        finally:
            # Clean up resources
            try:
                if hasattr(app, 'camera_active') and app.camera_active:
                    app.stop_camera()
                if hasattr(app, 'serial_conn') and app.serial_conn and app.serial_conn.is_open:
                    app.serial_conn.close()
            except Exception as e:
                print(f"Cleanup error: {e}")
            print("Smart Trash Classification System shutdown complete")
            
    except Exception as e:
        print(f"Failed to start application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 