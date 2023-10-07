import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('../../../../../datasets/red_wine_quality/dataset.csv')

# Check for missing values and handle them
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(), inplace=True)  # Fill missing values with column mean. Adjust this as needed.

# Split the data into features (X) and target variable (y)
X = df.drop('quality', axis=1)
y = df['quality']

# Splitting the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the RandomForestRegressor model
regr = RandomForestRegressor()

# Train the model
regr.fit(X_train, y_train)

# Test the model sample by sample and round the predicted values
rounded_predictions = []
n=0
for i, row in X_test.iterrows():
    n = n+1
    predicted_value = float(regr.predict(row.to_frame().T))
    rounded_predicted_value = round(predicted_value)
    rounded_predictions.append(rounded_predicted_value)
    print(f"Nº {n} | Expected Value: {y_test.loc[i]} | Predicted Value: {rounded_predicted_value}")

# Calculate the accuracy after rounding
accuracy = accuracy_score(y_test, rounded_predictions)
print(f"Accuracy after rounding: {accuracy * 100:.2f}%")
