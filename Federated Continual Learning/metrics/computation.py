from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import numpy as np
import os
import psutil
# from ..evaluation.metric_utils import bytes2human
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
import PIL.Image
from torchvision.transforms import ToTensor
import io
import queue
import subprocess
import threading
import time
import logging
class GPUUsage:
    """
        GPU usage metric measured as average usage percentage over time.

        :param gpu_id: GPU device ID
        :param every: time delay (in seconds) between measurements
    """

    def __init__(self, gpu_id, every=10):
        # 'nvidia-smi --loop=1 --query-gpu=utilization.gpu --format=csv'
        cmd = ['nvidia-smi', f'--loop={every}', '--query-gpu=utilization.gpu,memory.used',
               '--format=csv', f'--id={gpu_id}']
        # something long running
        try:
            self.p = subprocess.Popen(cmd, bufsize=1, stdout=subprocess.PIPE)
        except NotADirectoryError:
            raise ValueError('No GPU available: nvidia-smi command not found.')

        self.lines_queue = queue.Queue()
        self.read_thread = threading.Thread(target=GPUUsage.push_lines,
                                            args=(self,), daemon=True)
        self.read_thread.start()

        self.n_measurements = 0
        self.avg_usage = 0
        self.log = logging.getLogger("avalanche")

    def compute(self, t):
        """
        Compute CPU usage measured in seconds.

        :param t: task id
        :return: float: average GPU usage
        """
        init_mem = None
        peak_mem = 0 
        while not self.lines_queue.empty():
            line = self.lines_queue.get()
            if line[0] == 'u':  # skip first line 'utilization.gpu [%]'
                continue
            usage, mem = line.strip().split(", ")
            usage = int(usage[:-1])
            mem = int(mem[:-3]) 
            if init_mem is None: 
                init_mem = mem 
            peak_mem = max(peak_mem, mem) 
            self.n_measurements += 1
            self.avg_usage += usage

        if self.n_measurements > 0:
            self.avg_usage /= float(self.n_measurements)
        self.log.info(f"Train Task {t} - average GPU usage: {self.avg_usage}%")

        print("Peak mem and init mem:", peak_mem, init_mem)
        if init_mem is None:
            init_mem = 0
        return (self.avg_usage, peak_mem - init_mem)

    def push_lines(self):
        while True:
            line = self.p.stdout.readline()
            self.lines_queue.put(line.decode('ascii'))

    def close(self):
        self.p.terminate()


class CPUUsage:
    """
        CPU usage metric measured in seconds.
    """
    def __init__(self):
        self.log = logging.getLogger("avalanche")

    def compute(self, t):
        """
        Compute CPU usage measured in seconds.

        :param t: task id
        :return: tuple (float, float): (user CPU time, system CPU time)
        """
        p = psutil.Process(os.getpid())
        times = p.cpu_times()
        user, sys = times.user, times.system
        self.log.info("Train Task {:} - CPU usage: user {} system {}"
                      .format(t, user, sys))
        return user, sys
    


class RAMU(object):

    def __init__(self):
        """
        RAM Usage metric.
        """
        self.log = logging.getLogger("avalanche")

    def compute(self, t):

        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss  # in bytes

        self.log.info("Train Task {:} - MU: {:.3f} GB"
                      .format(t, mem / (1024 * 1024 * 1024)))

        return mem / (1024 * 1024 * 1024)




class TimeUsage:

    """
        Time usage metric measured in seconds.
    """

    def __init__(self):
        self._start_time = time.perf_counter()
        self.log = logging.getLogger("avalanche")

    def compute(self, t):
        elapsed_time = time.perf_counter() - self._start_time
        self.log.info(f"Elapsed time: {elapsed_time:0.4f} seconds")
    