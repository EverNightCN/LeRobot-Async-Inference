# LeRobot Async Inference

An optimized asynchronous inference engine for LeRobot, designed to run VLA models (like SmolVLA) smoothly on SO-101 arms.

> ⚠️ **Compatibility Note**: This script is designed for **LeRobot v3.x**. It is **NOT** compatible with LeRobot v4.0 or higher due to API changes.

## Key Features

*   🚀 **Asynchronous Architecture**: Decouples inference (3Hz) from control (25Hz) to prevent stuttering.
*   🌊 **Temporal Aggregation**: Uses Exponential Weighted Averaging for fluid, jitter-free motion.
*   ⚡ **Performance Tuned**: Integrated `torch.compile` and optimized loop timings.
*   🛡️ **Robust**: Handles camera timeouts and hardware instabilities gracefully.

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/EverNightCN/LeRobot-Async-Inference
    ```
2.  **Important**: Move the `inference_async.py` file to the **root directory** of your existing `lerobot` project (e.g., `lerobot/inference_async.py`). It relies on relative imports from the `lerobot` package.

## Usage

1.  **Install Dependencies**:
    Ensure you have the `lerobot` environment set up.

2.  **Configure**:
    Edit the configuration section in `inference_async.py` to match your setup:
    ```python
    # --- 配置区域 ---
    FPS = 25                         # Control frequency
    RESET_TIME_SEC = 1.0             # Reset duration
    TASK_DESCRIPTION = "Grub the shoe" # Task description for VLA models
    ROBOT_PORT = "COM5"              # Robot serial port
    ```

3.  **Run**:
    ```bash
    python inference_async.py
    ```

4.  **Controls**:
    *   `s`: Start / Resume Inference
    *   `p`: Pause Inference
    *   `r`: Reset Robot (Stop & Home)
    *   `q`: Quit Program

---

# LeRobot 异步推理引擎 (中文说明)

专为 LeRobot 设计的优化版异步推理引擎，旨在让 VLA 模型（如 SmolVLA）在 SO-101 机械臂上流畅运行。

> ⚠️ **兼容性注意**: 本脚本专为 **LeRobot v3.x** 设计。由于 API 变更，**不支持** LeRobot v4.0 及以上版本。

## 核心特性

*   🚀 **异步架构**: 将推理 (3Hz) 与控制 (25Hz) 解耦，防止卡顿。
*   🌊 **时序聚合**: 使用指数加权平均 (Exponential Weighted Averaging) 实现如流体般顺滑、无抖动的运动。
*   ⚡ **性能调优**: 集成 `torch.compile` 并优化了循环时序。
*   🛡️ **鲁棒性**: 优雅地处理摄像头超时和硬件不稳定情况。

## 安装

1.  克隆本仓库:
    ```bash
    git clone https://github.com/EverNightCN/LeRobot-Async-Inference
    ```
2.  **重要**: 将 `inference_async.py` 文件移动到你现有的 `lerobot` 项目的 **根目录** 下 (例如 `lerobot/inference_async.py`)。它依赖于 `lerobot` 包的相对导入。

## 使用方法

1.  **安装依赖**:
    确保已配置好 `lerobot` 环境。

2.  **配置**:
    编辑 `inference_async.py` 中的配置区域以匹配你的设置:
    ```python
    # --- 配置区域 ---
    FPS = 25                         # 控制频率
    RESET_TIME_SEC = 1.0             # 复位耗时
    TASK_DESCRIPTION = "Grub the shoe" # VLA 模型的任务描述
    ROBOT_PORT = "COM5"              # 机械臂串口
    ```

3.  **运行**:
    ```bash
    python inference_async.py
    ```

4.  **控制**:
    *   `s`: 开始 / 恢复推理
    *   `p`: 暂停推理
    *   `r`: 复位机械臂 (停止并回零)
    *   `q`: 退出程序
