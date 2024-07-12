import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('ETH_USD_2017_2024.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values('Date')

# Set Date as the index
df.set_index('Date', inplace=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Create lag features
df['Lag_Close'] = df['Close'].shift(1)
df['Lag_Volume'] = df['Volume'].shift(1)

# Create moving averages
df['MA7'] = df['Close'].rolling(window=7).mean()
df['MA21'] = df['Close'].rolling(window=21).mean()

# Create price changes
df['Price_Change'] = df['Close'] - df['Lag_Close']

# Drop missing values
df = df.dropna()

# Define features and target
features = ['Lag_Close', 'Lag_Volume', 'MA7', 'MA21', 'Price_Change']
target = 'Close'

# Prepare data for preprocessing
window_size = 7
# Prepare data for preprocessing
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])
scaler_target = MinMaxScaler()
scaled_target = scaler_target.fit_transform(df[[target]])


# Function to create sequences
def create_sequences(data, target, time_steps=window_size):
    xs, ys = [], []
    for i in range(len(data) - time_steps):
        xs.append(data[i:(i + time_steps)])
        ys.append(target[i + time_steps])
    return np.array(xs), np.array(ys)


X, y = create_sequences(scaled_features, scaled_target, window_size)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the model
input_layer = Input(shape=(window_size, len(features)))
x1 = Dense(128, activation='relu')(input_layer)
x1 = Dropout(0.2)(x1)
x2 = Dense(64, activation='relu')(x1)
x2 = Dropout(0.2)(x2)
output_layer = Dense(1)(x2)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

model.summary()

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(-1, 1)  # change the shape to 2 dimensional.
y_pred_inverse = scaler_target.inverse_transform(y_pred)  # changed to scaler_target
y_test = scaler_target.inverse_transform(y_test)

print(f'Predictions: {y_pred_inverse}')
print(f'Actual: {y_test}')

"""# Save the model
model.save('eth_model3.h5')"""

# create a dataframe from the predictions and actual values
df_predictions = pd.DataFrame({'Predictions': y_pred_inverse.flatten()})
df_predictions.to_csv('predictions.csv', index=False)

print('Dataframes merged and saved to csv.')

