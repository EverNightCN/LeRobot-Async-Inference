import time
import logging
import torch
import sys
import json
import threading
import collections
import numpy as np
from dataclasses import dataclass
from unittest.mock import patch
from pathlib import Path
from contextlib import nullcontext

# LeRobot Imports
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame, load_stats, cast_stats_to_numpy
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.utils import log_say, get_safe_torch_device
from lerobot.utils.robot_utils import busy_wait

# --- 跨平台键盘输入处理 ---
if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty
    import select

class NonBlockingInput:
    """
    跨平台非阻塞键盘输入上下文管理器
    支持 Windows (msvcrt) 和 Linux (termios)
    """
    def __enter__(self):
        if sys.platform != 'win32':
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        if sys.platform != 'win32':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self):
        if sys.platform == 'win32':
            if msvcrt.kbhit():
                return msvcrt.getch().decode('utf-8').lower()
        else:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1).lower()
        return None

# --- 配置区域 ---
FPS = 25                         # 尝试提高 FPS 到 25 (受限于摄像头读取速度 ~40ms，实际约 24fps)
RESET_TIME_SEC = 1.0             # 复位动作耗时
TASK_DESCRIPTION = "Grub the shoe"
ROBOT_PORT = "COM5" if sys.platform == 'win32' else "/dev/ttyACM0"
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

CAMERA_CONFIG = {
    "handeye": {"index": 0, "width": 640, "height": 480, "fps": 30}, # 摄像头仍可保持较高 FPS
    "fixed":   {"index": 1, "width": 640, "height": 480, "fps": 30}
}

HOME_POSITION = [-0.90, -99.92, 99.82, 72.19, 2.85, 0.00]

# --- 状态定义 ---
class State:
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    RESETTING = "RESETTING"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 辅助函数 ---

def get_joint_state(obs):
    if "observation.state" in obs:
        val = obs["observation.state"]
        return val.cpu().numpy() if isinstance(val, torch.Tensor) else val
    if "state" in obs:
        val = obs["state"]
        return val.cpu().numpy() if isinstance(val, torch.Tensor) else val
    try:
        return np.array([obs[f"{name}.pos"] for name in MOTOR_NAMES])
    except KeyError:
        return None

def move_to_home(robot, duration=3.0):
    logging.info("Moving to Home position...")
    obs = robot.get_observation()
    current_pos = get_joint_state(obs)
    
    if current_pos is None:
        logging.error("Could not determine current robot state. Aborting Reset.")
        return

    target_pos = np.array(HOME_POSITION)
    if len(current_pos) != len(target_pos):
        logging.error(f"State dimension mismatch: Current {len(current_pos)} vs Home {len(target_pos)}")
        return

    steps = int(duration * FPS)
    for i in range(steps):
        alpha = (i + 1) / steps
        interp_pos = current_pos + alpha * (target_pos - current_pos)
        action = {f"{name}.pos": val for name, val in zip(MOTOR_NAMES, interp_pos)}
        robot.send_action(action)
        time.sleep(1.0 / FPS)
    logging.info("Arrived at Home.")

# --- 异步推理核心类 ---

class AsyncInferenceEngine:
    def __init__(self, policy, robot_type, task, device, use_amp):
        self.policy = policy
        
        # [优化1] 尝试编译模型内部的 model 以加速推理 (而不是编译整个策略对象)
        # 编译整个策略对象会导致自定义方法 (如 predict_action_chunk) 丢失或无法访问
        if hasattr(self.policy, "model"):
            try:
                self.policy.model = torch.compile(self.policy.model, mode="reduce-overhead")
                logging.info("Policy model compiled with torch.compile (mode='reduce-overhead')")
            except Exception as e:
                logging.warning(f"Could not compile policy model: {e}")
        
        self.robot_type = robot_type
        self.task = task
        self.device = device
        self.use_amp = use_amp
        
        # 共享数据
        self.latest_obs_frame = None
        self.latest_obs_step = 0  # 新增：记录观测对应的时间步
        self.chunks = []          # 新增：存储 (start_step, action_chunk) 的列表，用于时序聚合
        
        # 线程控制
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.new_obs_event = threading.Event()
        self.reset_event = threading.Event()
        
        self.thread = threading.Thread(target=self._worker, name="InferenceThread")
        self.thread.daemon = True
        
        # 性能监控
        self.last_infer_time = 0
        self.total_steps_consumed = 0 # 新增：用于同步时间步

    def start(self):
        self.thread.start()
        logging.info("Inference thread started.")

    def stop(self):
        self.stop_event.set()
        self.new_obs_event.set() # 唤醒线程以便退出
        self.thread.join()
        logging.info("Inference thread stopped.")

    def reset(self):
        """清空动作队列并重置策略状态"""
        with self.lock:
            self.chunks = []
            self.total_steps_consumed = 0
        self.reset_event.set()
        self.new_obs_event.set() # 唤醒线程处理重置

    def update_obs(self, obs_frame):
        """主线程调用：更新最新的观测数据"""
        with self.lock:
            self.latest_obs_frame = obs_frame
            self.latest_obs_step = self.total_steps_consumed # 绑定当前时间步
        self.new_obs_event.set() # 通知工作线程有新数据

    def get_action(self):
        """主线程调用：获取下一个动作 (使用时序聚合 Temporal Aggregation)"""
        with self.lock:
            current_step = self.total_steps_consumed
            
            # 1. 聚合所有覆盖当前步的 Chunk
            action_sum = None
            total_weight = 0
            active_chunks = []
            
            # 指数加权参数 k
            # k=0.01 是 ACT 的默认值。
            # [优化2] 增大 k 到 0.05 以提升响应速度
            k = 0.01

            for start_step, chunk_data in self.chunks:
                rel_idx = current_step - start_step
                
                # 检查当前步是否在该 Chunk 的覆盖范围内
                if 0 <= rel_idx < len(chunk_data):
                    val = np.array(chunk_data[rel_idx])
                    
                    # 计算权重：rel_idx 越小（越接近 Chunk 的起始点），权重越大
                    # 这意味着基于"最近"观测生成的动作（rel_idx 小）会有更高权重
                    weight = np.exp(-k * rel_idx)

                    if action_sum is None:
                        action_sum = val * weight
                    else:
                        action_sum += val * weight
                    
                    total_weight += weight
                    active_chunks.append((start_step, chunk_data)) # 保留活跃 Chunk
                elif rel_idx < 0:
                    # 未来的 Chunk
                    active_chunks.append((start_step, chunk_data))
                # else: 过期的 Chunk，自动丢弃
            
            self.chunks = active_chunks # 清理过期数据
            
            if total_weight > 0:
                self.total_steps_consumed += 1
                return action_sum / total_weight # 加权平均
            
        return None

    def has_actions(self):
        with self.lock:
            return len(self.chunks) > 0

    def _worker(self):
        while not self.stop_event.is_set():
            # 等待新数据或重置信号
            if not self.new_obs_event.wait(timeout=0.1):
                continue
            
            # 处理重置
            if self.reset_event.is_set():
                if hasattr(self.policy, "reset"):
                    self.policy.reset()
                self.reset_event.clear()
                self.new_obs_event.clear()
                continue

            # 获取观测数据
            with self.lock:
                obs = self.latest_obs_frame
                start_step_count = self.latest_obs_step # 使用观测捕获时的准确时间步
                # 检查是否需要推理：如果当前已经有很多覆盖未来的 Chunk，可以稍缓
                # 但为了 Temporal Aggregation 的平滑性，我们希望尽可能多地叠加 Chunk
                # 所以只要有新数据就跑
            
            if obs is None:
                continue

            self.new_obs_event.clear() # 标记为已处理

            # 执行推理
            t0 = time.perf_counter()
            try:
                action_chunk = self._predict_chunk(obs)
            except Exception as e:
                logging.error(f"Inference error: {e}")
                continue
            t1 = time.perf_counter()
            self.last_infer_time = (t1 - t0) * 1000

            # 更新 Chunks
            if action_chunk is not None:
                with self.lock:
                    chunk_cpu = action_chunk.squeeze(0).cpu().tolist()
                    # 存储 (开始时间步, 动作数据)
                    # 注意：这里的 start_step_count 是观测时刻。
                    # 策略输出的第 0 个动作通常对应观测后的第 1 步 (或第 0 步，取决于具体实现)
                    # 在 LeRobot/ACT 中，通常 chunk[0] 是当前步或下一步。
                    # 我们假设 chunk[0] 对应 start_step_count (即观测时刻的动作，虽然已经过去了)
                    # Temporal Aggregation 会自动处理过期数据。
                    self.chunks.append((start_step_count, chunk_cpu))
                    
                    # 限制 Chunks 数量防止内存无限增长 (虽然自动清理已经处理了大部分)
                    if len(self.chunks) > 100:
                        self.chunks = self.chunks[-50:]

    def _predict_chunk(self, observation):
        """内部函数：处理数据格式并调用策略"""
        # 复制并转换为 Tensor
        batch = {}
        with torch.inference_mode(), torch.autocast(device_type=self.device.type) if self.device.type == "cuda" and self.use_amp else nullcontext():
            for name, value in observation.items():
                if isinstance(value, np.ndarray):
                    val_tensor = torch.from_numpy(value)
                    if "image" in name:
                        val_tensor = val_tensor.type(torch.float32) / 255
                        val_tensor = val_tensor.permute(2, 0, 1).contiguous()
                    val_tensor = val_tensor.unsqueeze(0) # Add batch dim
                    batch[name] = val_tensor.to(self.device)
                else:
                    batch[name] = value

            batch["task"] = self.task if self.task else ""
            batch["robot_type"] = self.robot_type if self.robot_type else ""

            # [兼容性修复] 某些版本的 SmolVLA 可能直接查找 'observation.language.tokens'
            # 如果策略有 tokenizer，我们手动进行 tokenization 并添加到 batch 中
            # 即使 self.policy 被编译过，只要我们没有覆盖 self.policy 变量，这里应该能访问到
            if self.task and "observation.language.tokens" not in batch:
                tokenizer = getattr(self.policy, "language_tokenizer", None)
                if tokenizer:
                    try:
                        # 使用策略配置中的最大长度
                        max_len = getattr(self.policy.config, "tokenizer_max_length", 48)
                        
                        tokenized = tokenizer(
                            [self.task], 
                            return_tensors="pt", 
                            padding="max_length", 
                            max_length=max_len,
                            truncation=True
                        )
                        batch["observation.language.tokens"] = tokenized["input_ids"].to(self.device)
                        batch["observation.language.attention_mask"] = tokenized["attention_mask"].to(self.device)
                    except Exception as e:
                        logging.warning(f"Manual tokenization failed: {e}")
                else:
                    # 如果没有 tokenizer，可能是旧版本或不同架构，尝试构造 dummy token 以防万一
                    # 但通常如果模型需要 token，它应该有 tokenizer
                    pass

            # 调用策略的 predict_action_chunk (如果存在)
            # SmolVLAPolicy 有这个方法
            if hasattr(self.policy, "predict_action_chunk"):
                return self.policy.predict_action_chunk(batch)
            else:
                # 回退到 select_action (单步)
                action = self.policy.select_action(batch)
                return action.unsqueeze(0) # 模拟 chunk

# --- 主程序 ---

def main():
    # 1. 初始化配置
    camera_config = {
        name: OpenCVCameraConfig(index_or_path=cfg["index"], width=cfg["width"], height=cfg["height"], fps=cfg["fps"])
        for name, cfg in CAMERA_CONFIG.items()
    }
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id="R12254106", 
        cameras=camera_config,
        disable_torque_on_disconnect=True
    )

    # 2. 初始化机器人
    logging.info(f"Initializing robot on port {ROBOT_PORT}...")
    robot = SO101Follower(robot_config)
    
    # 3. 加载策略
    logging.info("Loading policy...")
    policy_path = Path(r"D:\Document\College\Lerobot\outputs\train\smolvla_so101_shoeflat_15\checkpoints\020000\pretrained_model")
    
    # 4. 构建 Features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # 构造 Mock MetaData
    @dataclass
    class DatasetMeta:
        features: dict
        fps: int
        robot_type: str
        stats: dict | None = None

    # Load stats
    stats_path = policy_path / "dataset_stats.json"
    stats = None
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
        stats = cast_stats_to_numpy(stats)
    else:
        stats = load_stats(policy_path)
    
    if stats is None:
        logging.warning("Could not find dataset stats. Normalization might be incorrect.")

    meta = DatasetMeta(features=dataset_features, fps=FPS, robot_type=robot.robot_type, stats=stats)
    
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    
    policy = make_policy(policy_cfg, ds_meta=meta)
    policy.eval()
    
    # CUDA Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        logging.info("Using CUDA for inference. (GPU Accelerated)")
    else:
        device = get_safe_torch_device(policy.config.device)
        logging.info(f"CUDA not available, using {device}. (CPU Mode)")
    
    policy.to(device)
    
    # 5. 初始化异步推理引擎
    engine = AsyncInferenceEngine(
        policy=policy,
        robot_type=robot.robot_type,
        task=TASK_DESCRIPTION,
        device=device,
        use_amp=policy.config.use_amp
    )
    engine.start()

    # 6. 主循环
    with NonBlockingInput() as kb:
        try:
            logging.info("Connecting to robot...")
            with patch('builtins.input', return_value=""):
                robot.connect()
            log_say("System Ready (Async Mode).")
            
            current_state = State.IDLE
            
            print("\n" + "="*40)
            print("  ASYNC CONTROL KEYS:")
            print("  [s] : Start / Resume Inference")
            print("  [p] : Pause Inference")
            print("  [r] : Reset Robot (Stop & Home)")
            print("  [q] : Quit Program")
            print("="*40 + "\n")

            while True:
                loop_start_time = time.perf_counter()
                
                # --- 输入处理 ---
                key = kb.get_key()
                if key == 'q':
                    if current_state in [State.RUNNING, State.PAUSED]:
                        logging.info("Stopping... Resetting to Home.")
                        move_to_home(robot, duration=RESET_TIME_SEC)
                    break
                elif key == 'r':
                    current_state = State.RESETTING
                    logging.info("Command: RESET")
                elif key == 'p':
                    if current_state == State.RUNNING:
                        current_state = State.PAUSED
                        logging.info("Command: PAUSE")
                elif key == 's':
                    if current_state in [State.IDLE, State.PAUSED]:
                        current_state = State.RUNNING
                        logging.info("Command: START/RESUME")
                        engine.reset() # 重置推理引擎状态

                # --- 状态机 ---
                if current_state == State.RESETTING:
                    move_to_home(robot, duration=RESET_TIME_SEC)
                    current_state = State.IDLE
                    engine.reset()
                    logging.info("System IDLE.")

                elif current_state == State.RUNNING:
                    # 1. 获取最新观测 (增加重试机制以应对摄像头超时)
                    try:
                        observation = robot.get_observation()
                    except TimeoutError as e:
                        logging.warning(f"Camera timeout: {e}. Skipping this frame.")
                        # 即使摄像头超时，也应继续循环，避免程序崩溃
                        # 可以选择复用上一帧的观测，或者仅跳过本次推理更新
                        busy_wait(1 / FPS - dt)
                        continue
                    
                    observation_frame = build_dataset_frame(dataset_features, observation, prefix="observation")
                    
                    # 2. 发送给推理引擎 (非阻塞)
                    engine.update_obs(observation_frame)
                    
                    # 3. 尝试从队列获取动作
                    action_values = engine.get_action()
                    
                    if action_values is not None:
                        # 执行动作
                        action = {key: action_values[i] for i, key in enumerate(robot.action_features)}
                        robot.send_action(action)
                    else:
                        # 队列为空 (Starvation)
                        # 对于 VLA，刚开始可能需要等待几百毫秒
                        # logging.warning("Action queue empty! Waiting for inference...")
                        pass

                # --- 循环控制 ---
                dt = time.perf_counter() - loop_start_time
                
                # 状态显示
                if robot.is_connected:
                    obs = robot.get_observation()
                    current_pos = get_joint_state(obs)
                    if current_pos is not None:
                        pos_str = " ".join([f"{p:6.2f}" for p in current_pos])
                        active_chunks = len(engine.chunks)
                        infer_ms = int(engine.last_infer_time)
                        sys.stdout.write(f"\rState: {current_state:<8} | Chunks: {active_chunks:<2} | Infer: {infer_ms}ms | Joints: [{pos_str}]")
                        sys.stdout.flush()

                busy_wait(1 / FPS - dt)

        except Exception as e:
            logging.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            engine.stop()
            if robot.is_connected:
                robot.disconnect()
            logging.info("Disconnected.")

if __name__ == "__main__":
    main()
