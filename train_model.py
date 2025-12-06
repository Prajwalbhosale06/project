from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
import os

DATA_PATH = os.path.join('MP_Data') 
# Make sure these match Step 1 exactly
actions = np.array(['Hello', 'NO']) 
no_sequences = 30 
sequence_length = 30
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

# Load Data Safely
for action in actions:
    for sequence in range(no_sequences):
        try:
            path = os.path.join(DATA_PATH, action, "{}.npy".format(sequence))
            res = np.load(path)
            # 258 is the shape of (Pose + Left Hand + Right Hand)
            if res.shape == (30, 258): 
                sequences.append(res)
                labels.append(label_map[action])
        except:
            pass

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- ADVANCED MODEL ARCHITECTURE ---
model = Sequential()

# Bidirectional LSTM 1
model.add(Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=(30, 258)))
model.add(Dropout(0.2)) # Prevent memorization

# Bidirectional LSTM 2
model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
model.add(Dropout(0.2))

# Bidirectional LSTM 3 (Last one, return_sequences=False)
model.add(Bidirectional(LSTM(64, return_sequences=False, activation='relu')))

# Dense Layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Early Stopping (Stop if not improving for 15 epochs)
early_stop = EarlyStopping(monitor='categorical_accuracy', patience=15, restore_best_weights=True)

# Train
model.fit(X_train, y_train, epochs=200, callbacks=[TensorBoard(log_dir='Logs'), early_stop])

model.save('action.h5')
print("Model trained and saved as action.h5")