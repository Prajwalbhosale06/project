from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt

DATA_PATH = os.path.join('MP_Data') 
# ⚠️ CRITICAL: Must match collection script EXACTLY (case-sensitive!)
actions = np.array(['Hello', 'NO'])  # Keep uppercase 'NO'
no_sequences = 30 
sequence_length = 30
label_map = {label:num for num, label in enumerate(actions)}

print("\n" + "="*60)
print("SIGN LANGUAGE MODEL TRAINING")
print("="*60)
print(f"Actions: {actions}")
print(f"Expected sequences per action: {no_sequences}")
print(f"Frames per sequence: {sequence_length}")
print("="*60 + "\n")

sequences, labels = [], []

# Load Data with validation
print("Loading training data...\n")
for action in actions:
    action_sequences_loaded = 0
    for sequence in range(no_sequences):
        try:
            path = os.path.join(DATA_PATH, action, f"{sequence}.npy")
            res = np.load(path)
            
            # Validate shape
            if res.shape == (30, 258): 
                sequences.append(res)
                labels.append(label_map[action])
                action_sequences_loaded += 1
            else:
                print(f"⚠️  Skipping {path}: Wrong shape {res.shape}, expected (30, 258)")
        except FileNotFoundError:
            print(f"⚠️  Missing file: {path}")
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
    
    print(f"✅ Loaded {action_sequences_loaded}/{no_sequences} sequences for '{action}'")

print(f"\n📊 Total sequences loaded: {len(sequences)}")

if len(sequences) == 0:
    print("❌ ERROR: No valid sequences found! Please run data collection first.")
    exit()

# Check class balance
unique, counts = np.unique(labels, return_counts=True)
print("\n📊 Class distribution:")
for action_idx, count in zip(unique, counts):
    print(f"   {actions[action_idx]}: {count} sequences ({count/len(labels)*100:.1f}%)")

if len(unique) < len(actions):
    print(f"⚠️  WARNING: Only {len(unique)}/{len(actions)} classes have data!")

# Prepare data
X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"\n📐 Data shapes:")
print(f"   X: {X.shape} (samples, frames, features)")
print(f"   y: {y.shape} (samples, classes)")

# Split data - using 20% for testing (better than 5%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=labels)

print(f"\n📊 Data split:")
print(f"   Training: {X_train.shape[0]} sequences")
print(f"   Testing: {X_test.shape[0]} sequences")

# --- IMPROVED MODEL ARCHITECTURE ---
print("\n🏗️  Building model architecture...")

model = Sequential([
    # First Bidirectional LSTM layer
    Bidirectional(LSTM(64, return_sequences=True, activation='relu'), 
                  input_shape=(30, 258)),
    BatchNormalization(),
    Dropout(0.3),
    
    # Second Bidirectional LSTM layer
    Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
    BatchNormalization(),
    Dropout(0.3),
    
    # Third Bidirectional LSTM layer (final sequence layer)
    Bidirectional(LSTM(64, return_sequences=False, activation='relu')),
    BatchNormalization(),
    Dropout(0.3),
    
    # Dense layers
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    
    # Output layer
    Dense(actions.shape[0], activation='softmax')
])

# Print model summary
print("\n" + "="*60)
model.summary()
print("="*60 + "\n")

# Compile with custom learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, 
    loss='categorical_crossentropy', 
    metrics=['categorical_accuracy']
)

# --- CALLBACKS ---
callbacks = [
    # TensorBoard for visualization
    TensorBoard(log_dir='Logs'),
    
    # Early stopping - stop if accuracy doesn't improve for 20 epochs
    EarlyStopping(
        monitor='categorical_accuracy', 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='categorical_accuracy',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )
]

# --- TRAIN ---
print("🚀 Starting training...\n")
print("💡 Tips:")
print("   - Training will stop early if accuracy plateaus")
print("   - Best weights will be restored automatically")
print("   - Monitor the Logs/ folder with TensorBoard\n")

history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test),
    epochs=200, 
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# --- EVALUATE ---
print("\n" + "="*60)
print("📊 EVALUATION ON TEST SET")
print("="*60)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Detailed per-class accuracy
print("\n📊 Per-class performance:")
predictions = model.predict(X_test, verbose=0)
pred_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

for i, action in enumerate(actions):
    mask = true_labels == i
    if mask.sum() > 0:
        class_accuracy = (pred_labels[mask] == true_labels[mask]).mean()
        print(f"   {action}: {class_accuracy*100:.2f}% ({mask.sum()} samples)")

# Confusion analysis
print("\n🔍 Confusion analysis:")
for true_idx, true_action in enumerate(actions):
    for pred_idx, pred_action in enumerate(actions):
        if true_idx != pred_idx:
            confused = ((true_labels == true_idx) & (pred_labels == pred_idx)).sum()
            total = (true_labels == true_idx).sum()
            if confused > 0:
                print(f"   '{true_action}' misclassified as '{pred_action}': {confused}/{total} ({confused/total*100:.1f}%)")

# --- SAVE MODEL ---
model.save('action.h5')
print("\n✅ Model saved as 'action.h5'")

# --- PLOT TRAINING HISTORY ---
print("\n📈 Generating training plots...")

plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("✅ Training plots saved as 'training_history.png'")

# --- FINAL RECOMMENDATIONS ---
print("\n" + "="*60)
print("🎯 RECOMMENDATIONS")
print("="*60)

if test_accuracy < 0.85:
    print("\n⚠️  Accuracy is below 85%. Consider:")
    print("   1. Collecting more data (50+ sequences per action)")
    print("   2. Making signs more distinct from each other")
    print("   3. Ensuring consistent sign execution")
    print("   4. Improving lighting conditions")

if test_accuracy > 0.95:
    print("\n✅ Excellent accuracy! Your model is ready to use.")
    
# Check if any class is performing poorly
for i, action in enumerate(actions):
    mask = true_labels == i
    if mask.sum() > 0:
        class_accuracy = (pred_labels[mask] == true_labels[mask]).mean()
        if class_accuracy < 0.8:
            print(f"\n⚠️  '{action}' has low accuracy ({class_accuracy*100:.1f}%)")
            print(f"   Consider collecting more diverse samples for this sign")

print("\n💡 Next steps:")
print("   1. Review training_history.png")
print("   2. Test the model in your app")
print("   3. If accuracy is low, collect more diverse data")
print("   4. Use 'tensorboard --logdir=Logs' to see detailed metrics\n")