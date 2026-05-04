import time
import psutil
import os

def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # MB

class Timer:
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self):
        return time.time() - self.start_time