import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryVisualizer3D:
    def __init__(self):
        # 开启 matplotlib 的交互模式 (Interactive Mode)
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置工业风暗色背景 (可选，显得更高级)
        self.fig.patch.set_facecolor('#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.tick_params(axis='z', colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('white')
        
        self.ax.set_title("DeepVO 3D Trajectory (Draggable)", color='white', pad=20)
        self.ax.set_xlabel("X (Right)")
        self.ax.set_ylabel("Z (Forward)")
        self.ax.set_zlabel("Y (Down/Up)")
        
        # 存储历史 3D 轨迹点
        self.xs = []
        self.ys = []
        self.zs = []
        
        # 初始化一个空的 3D 曲线对象 (红色，线宽 2)
        self.line, = self.ax.plot([], [], [], color='#ff0055', linewidth=2, label="Camera Path")
        
        # 画一个原点标记
        self.ax.scatter([0], [0], [0], color='cyan', marker='*', s=100, label="Start (0,0,0)")
        
        legend = self.ax.legend(facecolor='#2b2b2b', edgecolor='white')
        for text in legend.get_texts():
            text.set_color("white")

    def update(self, x, y, z):
        """接收新的 3D 坐标并实时刷新图表"""
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        
        # 更新曲线数据
        # 注意物理坐标系映射：相机的 Z 是前进方向，我们在图表中把它映射到平面的 Y 轴
        # 相机的 Y 是上下方向，我们在图表中把它映射到高度 Z 轴
        self.line.set_data(self.xs, self.zs) 
        self.line.set_3d_properties(self.ys)
        
        # 动态调整坐标轴的范围，让视野始终包裹着最新的轨迹
        margin = 2.0
        self.ax.set_xlim(min(self.xs) - margin, max(self.xs) + margin)
        self.ax.set_ylim(min(self.zs) - margin, max(self.zs) + margin) 
        self.ax.set_zlim(min(self.ys) - margin, max(self.ys) + margin) 
        
        # 暂停极短的时间 (1毫秒)，让 GUI 渲染画面，同时处理你的鼠标拖拽事件
        plt.pause(0.001)

    def close(self, save_path=None):
            """视频跑完后，关闭交互模式并保持窗口打开，支持自动保存截图"""
            plt.ioff()
            # 如果传了保存路径，就在展示前自动截一张超高清的图
            if save_path:
                self.fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.fig.get_facecolor())
                print(f"📸 3D 轨迹超清截图已自动保存至: {save_path}")
                
            plt.show()