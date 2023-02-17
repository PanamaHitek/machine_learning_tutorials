import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the saved model
filename = 'trained_model.joblib'
loaded_model = joblib.load(filename)

# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv("dataset.csv", header=None, names=["Time", "Value"])

# Extract the "Value" column as the observations for the HMM
observations = df["Value"].values.reshape(-1, 1)

# Predict the hidden states for each observation using the loaded model
hidden_states = loaded_model.predict(observations)

# Filter the hidden states
filtered_hidden_states = np.full_like(hidden_states, 0)
current_state = hidden_states[0]
cluster_length = 1
for i in range(1, len(hidden_states)):
    if hidden_states[i] == current_state:
        cluster_length += 1
    else:
        if cluster_length >= 75:
            filtered_hidden_states[i-cluster_length:i] = 3
        current_state = hidden_states[i]
        cluster_length = 1

for i in range(0, len(filtered_hidden_states)):
    if (filtered_hidden_states[i]==0):
        observations[i]=0

# Print the filtered hidden states
print("Filtered Hidden States:", filtered_hidden_states)

# Plot the observations and filtered hidden states
fig, ax = plt.subplots()
ax.plot(observations, label="Observations")
ax.plot(filtered_hidden_states, label="Filtered Hidden States")
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Value")
plt.show()
