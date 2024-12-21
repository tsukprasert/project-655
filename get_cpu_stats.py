import os 
import pandas as pd
import numpy as np

np.random.seed(42)



datadir = 'mm1_trace_v2'

files = os.listdir(datadir)

cpu_df = pd.DataFrame(columns=['source', 'max_cpus', 'min_cpus', 'mean_cpus'])
for file in files:
    trace_df = pd.read_csv(os.path.join(datadir, file))

    cpu_stats = trace_df['cpus']
    max_cpu = max(cpu_stats)
    min_cpu = min(cpu_stats)
    mean_cpu = cpu_stats.mean()

    # print(f"File: {file}")
    # print(f"Max CPUs: {max_cpu}")
    # print(f"Min CPUs: {min_cpu}\n")


    name = file.replace('.csv', '')
    print(file)
    if name == 'pai':
        name = 'Alibaba'
    name = name.capitalize()
    cpu_df = cpu_df._append({'source': name, 'max_cpus': max_cpu, 'min_cpus': min_cpu, 'mean_cpus':mean_cpu}, ignore_index=True)
print(cpu_df)
cpu_df.to_csv('stats/cpu_stats_v2.csv', index=False)