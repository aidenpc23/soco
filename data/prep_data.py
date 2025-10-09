import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("AI_workload.csv")
df["Time (UTC)"] = pd.to_datetime(df["Time (UTC)"])
df = df.sort_values("Time (UTC)").reset_index(drop=True)

df["Hour Index"] = np.arange(len(df))

scaler = MinMaxScaler(feature_range=(0, 1))
df["Power (Norm)"] = scaler.fit_transform(df[["Power (Watt)"]])

window_size = 24
step_size = 2

wind_num = 0
sequences = []
timestamps = []

for start_idx in range(0, len(df) - window_size + 1, step_size):
    end_idx = start_idx + window_size
    seq = df["Power (Norm)"].iloc[start_idx:end_idx].values
    time_seq = df["Hour Index"].iloc[start_idx:end_idx].values
    sequences.append(seq)
    timestamps.append(time_seq)

    print(f"processed window {wind_num} : hour {time_seq[0]}-{time_seq[-1]}")
    wind_num += 1

sequences = np.array(sequences)

# normally input and output but instead im just using to split powers and times
X_train, X_test, time_train, time_test = train_test_split(
    sequences, timestamps, test_size=0.2, shuffle=False
)


def save_sequences_to_csv(X, times, filename):
    rows = []
    for seq, t in zip(X, times):
        row = {"start hour": t[0], "end hour": t[-1]}
        for i, val in enumerate(seq):
            row[f"t+{i}"] = val
        rows.append(row)
    pd.DataFrame(rows).to_csv(filename, index=False)


save_sequences_to_csv(X_train, time_train, "train.csv")
save_sequences_to_csv(X_test, time_test, "test.csv")

print("sequences saved to 'train_sequences.csv' and 'test_sequences.csv'")
