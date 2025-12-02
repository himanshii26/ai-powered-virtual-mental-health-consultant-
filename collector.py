# Data collector for OS performance

import time
import psutil
import pandas as pd
from datetime import datetime

def sample_processes(interval=2, duration=20, out_file="data/sample.parquet"):
    rows = []
    end_time = time.time() + duration

    while time.time() < end_time:
        timestamp = datetime.utcnow().isoformat()

        for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'io_counters', 'num_threads']):
            info = p.info
            rows.append({
                "timestamp": timestamp,
                "pid": info['pid'],
                "name": info['name'],
                "cpu_percent": info['cpu_percent'],
                "mem_rss": getattr(info['memory_info'], 'rss', None),
                "io_read": getattr(info['io_counters'], 'read_bytes', None),
                "io_write": getattr(info['io_counters'], 'write_bytes', None),
                "threads": info['num_threads']
            })

        time.sleep(interval)

    df = pd.DataFrame(rows)
    df.to_parquet(out_file, index=False)
    print(f"Data saved to {out_file}")

if __name__ == "__main__":
    sample_processes()
