import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
import joblib

# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv("dataset.csv", header=None, names=["Time", "Value"])

# Extract the "Value" column as the observations for the HMM
observations = df["Value"].values.reshape(-1, 1)

# Define the number of states for the HMM
n_states = 2

# Initialize the HMM model
model = hmm.GaussianHMM(n_components=n_states, covariance_type="full")

# Fit the HMM model to the observations
model.fit(observations)

# Predict the hidden states for each observation
hidden_states = model.predict(observations)

# Filter the hidden states
filtered_hidden_states = np.full_like(hidden_states, -1) # Create an array of the same shape as hidden_states with all elements set to -1
current_state = hidden_states[0] # Set the initial state to the first element of hidden_states
cluster_length = 1 # Initialize the length of the current cluster to 1

# Loop through all elements in hidden_states
for i in range(1, len(hidden_states)):
    if hidden_states[i] == current_state: # If the current state is still the same as the previous state
        cluster_length += 1 # Increase the length of the current cluster by 1
    else:
        if cluster_length >= 75: # If the length of the current cluster is greater than or equal to 75
            filtered_hidden_states[i-cluster_length:i] = 3 # Assign the value of 3 to the corresponding range of filtered_hidden_states
        current_state = hidden_states[i] # Set the current state to the current element of hidden_states
        cluster_length = 1 # Reset the length of the current cluster to 1

# Print the filtered hidden states
print("Filtered Hidden States:", filtered_hidden_states)
print("Number of States:", model.n_components)

# Save the model
filename = 'trained_model.joblib'
joblib.dump(model, filename)

# Load the saved model
loaded_model = joblib.load(filename)

# Plot the observations and filtered hidden states
fig, ax = plt.subplots()
ax.plot(observations, label="Observations")
ax.plot(filtered_hidden_states, label="Filtered Hidden States")
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Value")
plt.show()
