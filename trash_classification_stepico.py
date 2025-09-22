"""
智能垃圾分类系统 - 硬件控制模块
基于MicroPython的嵌入式控制系统

功能：
- ST7789 TFT显示屏驱动
- 舵机控制系统（3个垃圾桶）- 节能模式
- 串口通信管理（支持 USB 虚拟串口 与 硬件 UART0）
- 系统状态管理

舵机动作：收到指令时打开到 155°，保持约 3 秒，然后关闭到 108° 并断电（无初始角度设定）

通信协议：
- '1' = Organic Waste (有机垃圾)
- '2' = General Waste (普通垃圾) 
- '3' = Recyclables (可回收垃圾)
- '4' = Hazardous Waste (危险垃圾)

舵机GPIO配置：
- Organic (有机垃圾): GPIO 17
- Recycle (可回收垃圾): GPIO 16  
- Garbage (普通垃圾): GPIO 20
"""

import machine
import time
import json
from machine import Pin, PWM, UART
import st7789
import vga1_16x32 as font

# 可选：MicroPython 下的非阻塞 stdin 检测
try:
    import uselect as select  # MicroPython
except Exception:
    try:
        import select  # CPython/Thonny
    except Exception:
        select = None

import sys

class TrashClassificationSystem:
    """智能垃圾分类系统硬件控制类"""
    
    def __init__(self):
        """初始化系统"""
        # 优先尝试 USB 虚拟串口，其次硬件 UART0
        self.usb = None
        try:
            USB_VCP = getattr(machine, 'USB_VCP', None)
            if USB_VCP is not None:
                self.usb = USB_VCP()
        except Exception:
            self.usb = None
        
        # 硬件 UART0（仅当使用外部USB-TTL连接到 GPIO0/1 时生效）
        self.uart = None
        try:
            self.uart = UART(0, baudrate=115200, tx=Pin(0), rx=Pin(1))
        except Exception:
            self.uart = None
        
        # 输入解析缓冲
        self.input_buffer = ''
        
        self.setup_display()
        
        # 舵机角度控制参数（可根据舵机特性微调）
        self.servo_min_us = 500     # 0° 脉宽（保留，用于参考，不参与计算）
        self.servo_max_us = 2500    # 180° 脉宽（保留，用于参考，不参与计算）
        self.servo_freq_hz = 50     # 50Hz
        # 直接使用 duty_u16 区间进行角度映射（来自用户标定）
        self.servo_min_duty_u16 = 1802   # 0° 对应 duty_u16
        self.servo_max_duty_u16 = 7864   # 180° 对应 duty_u16
        self.open_angle_deg = 155    # 打开角度
        self.closed_angle_deg = 108   # 关闭角度
        self.hold_time_s = 2.0      # 保持时间
        
        self.setup_servos()
        self.setup_status_led()
        self.system_status = "ready"
        self.current_classification = None
        
        # 舵机电源控制
        self.servo_enable_pin = Pin(14, Pin.OUT)
        self.servo_enable_pin.value(0)
        
        # 分类映射
        self.class_mapping = {
            '1': {'name': 'Organic Waste', 'chinese': '有机垃圾', 'servo': 'organic', 'color': st7789.GREEN},
            '2': {'name': 'General Waste', 'chinese': '普通垃圾', 'servo': 'garbage', 'color': st7789.RED},
            '3': {'name': 'Recyclables', 'chinese': '可回收垃圾', 'servo': 'recyclable', 'color': st7789.BLUE},
            '4': {'name': 'Hazardous Waste', 'chinese': '危险垃圾', 'servo': 'recyclable', 'color': st7789.YELLOW}
        }
        
    def setup_display(self):
        """初始化ST7789显示屏"""
        try:
            self.display = st7789.ST7789(
                machine.SPI(1, baudrate=40000000, sck=Pin(10), mosi=Pin(11)),
                135, 240,
                reset=Pin(12, Pin.OUT),
                cs=Pin(9, Pin.OUT),
                dc=Pin(8, Pin.OUT),
                backlight=Pin(13, Pin.OUT),
                rotation=0
            )
            self.display.init()
            self.display.fill(st7789.BLACK)
            self.show_startup_screen()
            print("显示屏初始化成功")
        except Exception as e:
            print(f"显示屏初始化失败: {e}")
            
    def setup_servos(self):
        """初始化舵机控制系统（仅配置信号引脚，不常驻PWM，避免空闲噪音）"""
        try:
            # 仅准备信号引脚；不创建常驻PWM，避免在空闲时输出脉冲
            self.servo_pin_garbage = Pin(20, Pin.OUT)
            self.servo_pin_organic = Pin(17, Pin.OUT)
            self.servo_pin_recyclable = Pin(16, Pin.OUT)
            # 默认拉低，防止悬空抖动
            self.servo_pin_garbage.value(0)
            self.servo_pin_organic.value(0)
            self.servo_pin_recyclable.value(0)

            # 运行期PWM句柄（动作时才创建，动作后释放）
            self.servo_garbage = None
            self.servo_organic = None
            self.servo_recyclable = None

            print("舵机系统初始化成功")
            print("舵机GPIO配置:\n  - 普通垃圾桶: GPIO 20\n  - 有机垃圾桶: GPIO 17\n  - 可回收垃圾桶: GPIO 16")
        except Exception as e:
            print(f"舵机系统初始化失败: {e}")
            
    def setup_status_led(self):
        """初始化状态LED"""
        try:
            self.status_led = Pin(25, Pin.OUT)
            self.status_led.value(1)
        except Exception as e:
            print(f"状态LED初始化失败: {e}")
            
    def enable_servos(self):
        """开启舵机电源，并在需要时创建PWM（动作期）"""
        self.servo_enable_pin.value(1)
        time.sleep(0.1)
        print("舵机电源已开启")
        # 惰性创建PWM
        try:
            if self.servo_garbage is None:
                self.servo_garbage = PWM(self.servo_pin_garbage, freq=self.servo_freq_hz)
            if self.servo_organic is None:
                self.servo_organic = PWM(self.servo_pin_organic, freq=self.servo_freq_hz)
            if self.servo_recyclable is None:
                self.servo_recyclable = PWM(self.servo_pin_recyclable, freq=self.servo_freq_hz)
        except Exception as e:
            print(f"创建PWM失败: {e}")
        
    def disable_servos(self):
        """关闭舵机电源，并释放PWM（空闲零脉冲，避免噪音）"""
        # 尽量先停止PWM输出
        try:
            for pwm, pin in (
                (self.servo_garbage, self.servo_pin_garbage),
                (self.servo_organic, self.servo_pin_organic),
                (self.servo_recyclable, self.servo_pin_recyclable),
            ):
                if pwm is not None:
                    try:
                        pwm.duty_u16(0)
                    except Exception:
                        pass
                    try:
                        pwm.deinit()
                    except Exception:
                        pass
                # 将信号线拉低，避免悬空
                try:
                    pin.init(Pin.OUT)
                    pin.value(0)
                except Exception:
                    pass
            # 句柄置空，等待下次动作再创建
            self.servo_garbage = None
            self.servo_organic = None
            self.servo_recyclable = None
        except Exception as e:
            print(f"释放PWM失败: {e}")

        # 最后切断舵机电源
        self.servo_enable_pin.value(0)
        print("舵机电源已关闭")
        
    def angle_to_duty_u16(self, angle_deg: float) -> int:
        """将角度(0~180°)映射为 duty_u16，按给定整数公式计算。

        origin equation:
            duty = min_duty + (max_duty - min_duty) * angle // 180

        optimized (optional):
            # duty = min_duty + int(math.fabs((max_duty - min_duty) * (math.sin(angle // 180 * math.pi - math.pi/4))))
        """
        # 角度限制并取整
        a = int(angle_deg)
        if a < 0:
            a = 0
        elif a > 180:
            a = 180

        min_duty = self.servo_min_duty_u16
        max_duty = self.servo_max_duty_u16
        span = max_duty - min_duty

        # 整数公式（避免浮点运算）
        duty = min_duty + (span * a) // 180

        # 边界保护
        if duty < 0:
            duty = 0
        elif duty > 65535:
            duty = 65535
        return duty

    def set_servo_angle(self, servo: PWM, angle_deg: float) -> None:
        """设置单个舵机角度。"""
        servo.duty_u16(self.angle_to_duty_u16(angle_deg))

    def show_startup_screen(self):
        """显示启动屏幕"""
        self.display.fill(st7789.BLACK)
        self.display.text(font, "智能垃圾分类", 20, 60, st7789.WHITE)
        self.display.text(font, "系统启动中...", 20, 100, st7789.GREEN)
        self.display.text(font, "等待上位机连接", 20, 140, st7789.YELLOW)
        
    def show_classification_result(self, command):
        """显示分类结果"""
        if command not in self.class_mapping:
            self.show_error_screen("未知命令")
            return
        class_info = self.class_mapping[command]
        self.display.fill(st7789.BLACK)
        self.display.text(font, "识别结果", 60, 20, st7789.WHITE)
        self.display.text(font, f"类别: {class_info['chinese']}", 20, 60, class_info['color'])
        self.display.text(font, "正在开启对应垃圾桶...", 20, 140, st7789.YELLOW)
        
    def show_error_screen(self, message):
        """显示错误屏幕"""
        self.display.fill(st7789.BLACK)
        self.display.text(font, "错误", 60, 80, st7789.RED)
        self.display.text(font, message, 20, 120, st7789.YELLOW)
        
    def show_system_status(self, status):
        """显示系统状态"""
        self.display.fill(st7789.BLACK)
        self.display.text(font, "系统状态", 60, 20, st7789.WHITE)
        self.display.text(font, f"状态: {status}", 20, 60, st7789.GREEN)
        self.display.text(font, "等待指令...", 20, 100, st7789.YELLOW)
        
    def open_bin(self, command):
        """打开指定垃圾桶"""
        if command not in self.class_mapping:
            print(f"未知命令: {command}")
            return False
        try:
            self.enable_servos()
            class_info = self.class_mapping[command]
            servo_type = class_info['servo']
            target_angle = self.open_angle_deg   # 55°
            close_angle = self.closed_angle_deg  # 0°
            
            if servo_type == "garbage":
                self.set_servo_angle(self.servo_garbage, target_angle)
                print(f"开启普通垃圾桶 (GPIO 20) - {class_info['chinese']} → {target_angle}°")
            elif servo_type == "organic":
                self.set_servo_angle(self.servo_organic, target_angle)
                print(f"开启有机垃圾桶 (GPIO 17) - {class_info['chinese']} → {target_angle}°")
            elif servo_type == "recyclable":
                self.set_servo_angle(self.servo_recyclable, target_angle)
                print(f"开启可回收垃圾桶 (GPIO 16) - {class_info['chinese']} → {target_angle}°")
            else:
                print(f"未知舵机类型: {servo_type}")
                self.disable_servos()
                return False
            
            # 保持 3 秒
            time.sleep(self.hold_time_s)
            
            # 关闭到 0°
            self.close_all_bins()
            
            # 回到关闭位后断电
            self.disable_servos()
            return True
        except Exception as e:
            print(f"开启垃圾桶失败: {e}")
            self.disable_servos()
            return False
        
    def close_all_bins(self):
        """关闭所有垃圾桶"""
        try:
            # 统一回到关闭角度 0°
            self.set_servo_angle(self.servo_garbage, self.closed_angle_deg)
            self.set_servo_angle(self.servo_organic, self.closed_angle_deg)
            self.set_servo_angle(self.servo_recyclable, self.closed_angle_deg)
            time.sleep(0.2)
        except Exception as e:
            print(f"关闭垃圾桶失败: {e}")
            
    def test_servos(self):
        """测试舵机"""
        print("开始舵机测试...")
        self.show_system_status("舵机测试中")
        self.enable_servos()
        for command in ['1', '2', '3', '4']:
            self.open_bin(command)
            time.sleep(0.5)
        self.disable_servos()
        print("舵机测试完成")
        self.show_system_status("就绪")
        
    def send_status(self, status_data):
        """发送状态信息到上位机（print 到 USB，同步尝试 usb/uart 写入）"""
        try:
            # 1) 打印到 USB CDC（Thonny/串口工具可见）
            try:
                print(json.dumps(status_data))
            except Exception:
                pass
            # 2) 以字节写到 USB_VCP/UART
            message = (json.dumps(status_data) + '\n').encode('utf-8')
            if self.usb:
                try:
                    self.usb.write(message)
                except Exception:
                    pass
            if self.uart:
                try:
                    self.uart.write(message)
                except Exception:
                    pass
        except Exception as e:
            print(f"发送状态失败: {e}")
        
    def _parse_incoming(self, text):
        """解析输入命令"""
        for ch in text:
            if ch in ('\r', '\n', '\t', ' '):
                if self.input_buffer:
                    token = self.input_buffer
                    self.input_buffer = ''
                    if token in ('test', 'ping', 'status'):
                        return token
                continue
            if ch in ('1', '2', '3', '4'):
                self.input_buffer = ''
                return ch
            if ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'):
                self.input_buffer += ch.lower()
                if self.input_buffer in ('test', 'ping', 'status'):
                    token = self.input_buffer
                    self.input_buffer = ''
                    return token
        return None
        
    def _receive_from_usb_vcp(self):
        """从 USB_VCP 接收命令"""
        if self.usb and getattr(self.usb, 'any', None) and self.usb.any():
            try:
                data = self.usb.read()
                if data:
                    text = ''
                    try:
                        text = data.decode('utf-8', 'ignore')
                    except Exception:
                        pass
                    cmd = self._parse_incoming(text)
                    if cmd:
                        return cmd
            except Exception:
                return None
        return None
    
    def _receive_from_uart(self):
        """从 UART0 接收命令"""
        if self.uart and self.uart.any():
            try:
                data = self.uart.read()
                if data:
                    try:
                        text = data.decode('utf-8', 'ignore')
                    except Exception:
                        text = ''
                    cmd = self._parse_incoming(text)
                    if cmd:
                        return cmd
            except Exception:
                return None
        return None
    
    def _receive_from_stdin(self):
        """Thonny/REPL 友好：基于 stdin 的读取（换行结束）"""
        try:
            if select is not None:
                try:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0)
                except Exception:
                    rlist = []
                if rlist:
                    try:
                        line = sys.stdin.readline()
                    except Exception:
                        line = ''
                    if line:
                        return self._parse_incoming(line)
            else:
                # 无 select 时，尝试非阻塞读取（可能在部分固件不可用）
                if hasattr(sys.stdin, 'readline'):
                    try:
                        line = sys.stdin.readline()
                        if line:
                            return self._parse_incoming(line)
                    except Exception:
                        return None
        except Exception:
            return None
        return None
        
    def receive_command(self):
        """接收上位机命令（优先 USB stdin，其次 USB_VCP，最后 UART0）"""
        cmd = self._receive_from_stdin()
        if cmd:
            return cmd
        cmd = self._receive_from_usb_vcp()
        if cmd:
            return cmd
        cmd = self._receive_from_uart()
        if cmd:
            return cmd
        return None
            
    def process_command(self, command):
        """处理上位机命令"""
        try:
            print(f"收到命令: {command}")
            if command in self.class_mapping:
                self.show_classification_result(command)
                success = self.open_bin(command)
                status_data = {
                    'type': 'classification_result',
                    'command': command,
                    'class_name': self.class_mapping[command]['name'],
                    'bin_opened': success,
                    'timestamp': time.time()
                }
                self.send_status(status_data)
                time.sleep(0.5)
                self.show_system_status("就绪")
            elif command == 'test':
                self.test_servos()
            elif command == 'status':
                status_data = {
                    'type': 'system_status',
                    'status': self.system_status,
                    'current_classification': self.current_classification
                }
                self.send_status(status_data)
            elif command == 'ping':
                self.send_status({'type': 'pong'})
            else:
                print(f"未知命令: {command}")
                self.show_error_screen(f"未知命令: {command}")
        except Exception as e:
            print(f"处理命令失败: {e}")
            
    def run(self):
        """主运行循环"""
        print("智能垃圾分类系统启动")
        print("舵机GPIO配置:\n  - 普通垃圾桶: GPIO 20\n  - 有机垃圾桶: GPIO 17\n  - 可回收垃圾桶: GPIO 16")
        if self.usb:
            print("通信: USB_VCP 可用")
        if self.uart:
            print("通信: UART0 可用 (GPIO0/1)")
        self.show_system_status("就绪")
        
        while True:
            try:
                command = self.receive_command()
                if command:
                    self.process_command(command)
                time.sleep(0.05)
            except KeyboardInterrupt:
                print("系统停止")
                self.disable_servos()
                break
            except Exception as e:
                print(f"运行错误: {e}") 
                time.sleep(0.5)

# 主程序入口
if __name__ == "__main__":
    try:
        system = TrashClassificationSystem()
        system.run()
    except Exception as e:
        print(f"系统启动失败: {e}")


