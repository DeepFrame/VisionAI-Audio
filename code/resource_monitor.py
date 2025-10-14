import psutil
import time
import threading

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False


class ResourceMonitor:
    """
    Monitors CPU, RAM, and GPU (if available) usage during model inference.
    Automatically adapts the sampling interval for short GPU bursts.
    """
    def __init__(self, interval=0.1, expected_duration=None):
        """
        interval: Sampling frequency (seconds). If None, it adapts automatically.
        expected_duration: Estimated inference duration in seconds (optional).
        """
        # # Auto-tune interval based on expected duration
        # if interval is None:
        #     if expected_duration and expected_duration < 10:
        #         interval = 0.1    # Fast sampling for short runs
        #     elif expected_duration and expected_duration < 60:
        #         interval = 0.1
        #     else:
        #         interval = 0.1    # Long runs -> reduce overhead
        self.interval = interval

        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_mem = []
        self.running = False
        self.thread = None

    def _monitor(self):
        while self.running:
            # CPU + RAM
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            self.cpu_usage.append(cpu)
            self.memory_usage.append(mem)

            # GPU
            if HAS_NVML:
                try:
                    device_count = pynvml.nvmlDeviceGetCount()
                    util_list, mem_list = [], []
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util_list.append(util.gpu)
                        mem_list.append(mem_info.used / mem_info.total * 100)
                    # Average across all GPUs
                    self.gpu_usage.append(sum(util_list) / len(util_list))
                    self.gpu_mem.append(sum(mem_list) / len(mem_list))
                except Exception:
                    pass

            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def get_summary(self):
        summary = {
            "avg_cpu": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            "max_cpu": max(self.cpu_usage, default=0),
            "avg_mem": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            "max_mem": max(self.memory_usage, default=0),
        }

        if self.gpu_usage:
            summary.update({
                "avg_gpu": sum(self.gpu_usage) / len(self.gpu_usage),
                "max_gpu": max(self.gpu_usage, default=0),
                "avg_gpu_mem": sum(self.gpu_mem) / len(self.gpu_mem),
                "max_gpu_mem": max(self.gpu_mem, default=0),
            })
        return summary
