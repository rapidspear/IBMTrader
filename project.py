import numpy as np
import tensorflow as tf

# Example data: 6080 rows and 5 columns
data = np.array([
    [0.678778052, 0.675125599, 0.67889452, 0.674920678, 0.0568903312],
    [0.680084586, 0.673743725, 0.682450533, 0.676165, 0.0458001904],
    [0.674982905, 0.671733677, 0.677958727, 0.674360752, 0.0686043203],
    # ... (more rows)
    [0.256454915, 0.246482417, 0.246365964, 0.244509429, 0.2481139],
    [0.261929959, 0.251947224, 0.247551307, 0.247246936, 0.266066104],
    [0.272817761, 0.264510036, 0.264271021, 0.259316862, 0.228186786]
])

# Number of timesteps
timesteps = 5

# Reshape the data to (number_of_sequences, timesteps, features)
sequences = []
for i in range(len(data) - timesteps + 1):
    sequences.append(data[i:i + timesteps])
    print(sequences[-1])
    print("\n")

sequences = np.array(sequences)
print("Sequences shape:", sequences.shape)  # Should be (number_of_sequences, timesteps, features)

# Example labels (this should be your actual labels)
# Ensure the number of labels matches the number of sequences
labels = np.random.randint(0, 10, len(sequences))

# Convert labels to categorical if necessary
labels = tf.keras.utils.to_categorical(labels, num_classes=10)

# Define the LSTM model
print("------------------------=================-----------------------------")
print(data.shape[1])
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(timesteps, data.shape[1])),
    tf.keras.layers.Dense(10, activation='softmax')  # Example output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(sequences, labels, epochs=10, batch_size=32)