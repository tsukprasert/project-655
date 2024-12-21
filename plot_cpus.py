import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

datadir = 'stats'
save_dir = 'plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


filepath = os.path.join(datadir, 'cpu_stats_v2.csv')
stat_df = pd.read_csv(filepath, index_col='source').rename_axis(None)
stat_df = stat_df.reindex(['Mustang', "Azure", "Alibaba", "Google"])
stat_df = stat_df.reset_index()
stat_df.rename(columns={'index': 'source'}, inplace=True)
print(stat_df)
# exit()


sns.set_theme(style="whitegrid", font_scale=1.5)
sns.set_style("ticks")
sns.set_style({'font.family': 'Times New Roman'})
plt.rc('mathtext',**{'default':'regular'})

plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rc('axes', titlesize=18, titleweight=1)
plt.rc('legend',fontsize='18')
plt.rcParams['legend.title_fontsize'] = '18'

# sns.set_palette('Set2')

fig, ax = plt.subplots(figsize=(8,4.5))
stat_df.plot(x='source', y='mean_cpus', kind='bar', legend=False, edgecolor='black', ax=ax, color='lightskyblue')
# stat_df.columns = stat_df.columns.str.replace('_', ' ')

for bar, max_cpu in zip(ax.patches, stat_df['max_cpus']):
    ax.errorbar(stat_df['source'], stat_df['mean_cpus'], 
                yerr=[stat_df['mean_cpus'] - stat_df['min_cpus'], stat_df['max_cpus'] - stat_df['mean_cpus']], 
                fmt='none', ecolor='black', capsize=5)

for bar, mean_cpu, max_cpu in zip(ax.patches, stat_df['mean_cpus'], stat_df['max_cpus']):
    ax.text(bar.get_x() + bar.get_width() / 2, max_cpu , f'({mean_cpu:.0f}, {max_cpu:.0f})', ha='center', va='bottom', fontsize=16, color='k')

    ax.annotate('(Mean, Max) CPUs', xy=(0.70, 0.97), xycoords='axes fraction', fontsize=18, color='k', ha='left', va='top')
    # ax.annotate('Mean CPUs', xy=(0.81, 0.90), xycoords='axes fraction', fontsize=18, color='blue', ha='left', va='top')

# plt.title(r'CPUs Distribution for Different Workload Traces')
plt.xlabel('Workload Trace Sources')


plt.ylim([0, 10])
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(os.path.join(save_dir, 'cpus_v2.jpg'), dpi=300)
