import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "results_rat_moves.csv"
df = pd.read_csv(file_path)

# Compute overall averages for bar plot
avg_metrics = df[['ping_tries_1', 'ping_tries_2',
                  'moves_1', 'moves_2',
                  'total_actions_1', 'total_actions_2']].mean()

print(avg_metrics)

# Bar plot for overall averages
plt.figure(figsize=(10, 6))
avg_metrics.plot(kind='bar')
plt.title('Average Values of Ping Tries, Moves, and Total Actions')
plt.ylabel('Average Count')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Group by alpha and calculate average of ping, move, and total actions
grouped = df.groupby('alpha').agg({
    'ping_tries_1': 'mean',
    'ping_tries_2': 'mean',
    'moves_1': 'mean',
    'moves_2': 'mean',
    'total_actions_1': 'mean',
    'total_actions_2': 'mean'
}).reset_index()


grouped = grouped.sort_values(by='alpha')

xticks = sorted(df['alpha'].unique())

# Ping Tries
plt.figure(figsize=(10, 6))
plt.plot(grouped['alpha'], grouped['ping_tries_1'], marker='o', label='Ping Tries 1')
plt.plot(grouped['alpha'], grouped['ping_tries_2'], marker='s', label='Ping Tries 2')
plt.title('Ping Tries vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Average Ping Tries')
plt.xticks(xticks)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Move Tries
plt.figure(figsize=(10, 6))
plt.plot(grouped['alpha'], grouped['moves_1'], marker='o', label='Moves 1')
plt.plot(grouped['alpha'], grouped['moves_2'], marker='s', label='Moves 2')
plt.title('Moves vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Average Move Tries')
plt.xticks(xticks)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Total Actions
plt.figure(figsize=(10, 6))
plt.plot(grouped['alpha'], grouped['total_actions_1'], marker='o', label='Total Actions 1')
plt.plot(grouped['alpha'], grouped['total_actions_2'], marker='s', label='Total Actions 2')
plt.title('Total Actions vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Average Total Actions')
plt.xticks(xticks)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
