import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt




print("SYSTEM RUNNING, DO NOT SPAM BUTTONS")
start_time = time.time()

# Simulated data for demonstration purposes
def generate_data(num_samples=1000):
    # Generate system metrics
    node_availability = np.random.uniform(0.5, 1, num_samples)
    network_latency = np.random.uniform(0, 100, num_samples)
    resource_usage = np.random.uniform(0, 1, num_samples)
    failure_patterns = np.random.uniform(0, 1, num_samples)

    # Generate target replication levels based on system metrics
    target_replication = (1 - node_availability) * network_latency * resource_usage * failure_patterns
    target_replication = np.round(target_replication).astype(int)

    data = pd.DataFrame({
        'node_availability': node_availability,
        'network_latency': network_latency,
        'resource_usage': resource_usage,
        'failure_patterns': failure_patterns,
        'replication_level': target_replication
    })

    return data

data = generate_data()



# Split data into train and test sets
X = data.drop(columns='replication_level')
y = data['replication_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

def adaptive_replication(node_availability, network_latency, resource_usage, failure_patterns, model, scaler):
    features = np.array([[node_availability, network_latency, resource_usage, failure_patterns]])
    features_scaled = scaler.transform(features)
    replication_level = model.predict(features_scaled)
    return int(np.round(replication_level[0]))

# Example usage
node_availability = 0.9
network_latency = 50
resource_usage = 0.5
failure_patterns = 0.2
replication_level = adaptive_replication(node_availability, network_latency, resource_usage, failure_patterns, model, scaler)
print(f"Adaptive replication level: {replication_level}")


def plot_results(y_test, y_pred):
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Replication Level")
    plt.ylabel("Predicted Replication Level")
    plt.title("Actual vs. Predicted Replication Levels")
    plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")
plot_results(y_test, y_pred)
