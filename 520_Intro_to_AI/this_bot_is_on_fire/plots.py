import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("results.csv")

# Extract unique q values and sort them
q_values = sorted(df[" q"].unique())

# Generate success rates for each bot
success_rates = {f"Bot {i}": df.groupby(" q")[f" result{i}"].mean() for i in range(1, 5)}

# Plot success rates vs. q
plt.figure(figsize=(8, 6))
for bot, rates in success_rates.items():
    plt.plot(q_values, rates, marker="o", linestyle="-", label=bot)
plt.xlabel("Flammability (q)")
plt.ylabel("Success Frequency")
plt.ylim(0.44, 1.02)
plt.title("Bot Success Frequency vs. Flammability")
plt.legend()
plt.grid(True)
plt.savefig("writeup_plots/success_rate_vs_q.png")
# plt.show()

# Compute and plot winnability frequency
winnability_rate = df.groupby(" q")["Winnability"].mean()
plt.figure(figsize=(8, 6))
plt.plot(q_values, winnability_rate, marker="o", linestyle="-", color="red", label="Winnability")
plt.xlabel("Flammability (q)")
plt.ylabel("Winnability Frequency")
plt.ylim(0.44, 1.02)
plt.title("Frequency of Winnable Simulations vs. Flammability")
plt.legend()
plt.grid(True)
plt.savefig("writeup_plots/winnability_vs_q.png")
# plt.show()

# Filter winnable simulations and compute success rates
winnable_df = df[df["Winnability"] == 1]
winnable_success_rates = {f"Bot {i}": winnable_df.groupby(" q")[f" result{i}"].mean() for i in range(1, 5)}

# Plot success rates in winnable cases
plt.figure(figsize=(8, 6))
for bot, rates in winnable_success_rates.items():
    plt.plot(q_values, rates, marker="o", linestyle="-", label=bot)
plt.xlabel("Flammability (q)")
plt.ylabel("Success Frequency (Winnable Cases)")
plt.ylim(0.79, 1.01)
plt.title("Bot Success Frequency in Winnable Simulations vs. Flammability")
plt.legend()
plt.grid(True)
plt.savefig("writeup_plots/winnable_success_rate_vs_q.png")
# plt.show()

# Compute total success frequency for each bot
total_success_rates = df[[" result1", " result2", " result3", " result4"]].mean()

# Plot total success frequency per bot
plt.figure(figsize=(8, 6))
plt.bar(["Bot 1", "Bot 2", "Bot 3", "Bot 4"], total_success_rates, color=["blue", "orange", "green", "red"])
plt.ylabel("Success Frequency")
plt.ylim(0.58, 1.02)
plt.title("Average Success Frequency Across All Simulations")
plt.grid(True, axis="y")
plt.savefig("writeup_plots/total_success_frequency_per_bot.png")
# plt.show()

# Compute total success frequency among winnable simulations for each bot
total_winnable_success_rates = winnable_df[[" result1", " result2", " result3", " result4"]].mean()

# Plot total success frequency among winnable simulations per bot
plt.figure(figsize=(8, 6))
plt.bar(["Bot 1", "Bot 2", "Bot 3", "Bot 4"], total_winnable_success_rates, color=["blue", "orange", "green", "red"])
plt.ylabel("Success Frequency")
plt.ylim(0.58, 1.02)
plt.title("Average Success Frequency Across Winnable Simulations")
plt.grid(True, axis="y")
plt.savefig("writeup_plots/total_success_frequency_winnable_per_bot.png")
# plt.show()
