"""
SignSpeak — CNN Model Architecture

Deep CNN for ASL fingerspelling recognition on Sign MNIST (24 classes).
Sign MNIST skips J (label 9) and Z (label 25) — those require motion.
Labels in the CSV are 0–24 with 9 absent → 24 unique classes.
"""

from tensorflow.keras import layers, models, regularizers

NUM_CLASSES = 24
IMG_SIZE = 28

# Sign MNIST raw labels: 0,1,...,8,10,11,...,24  (9 is absent)
# Mapping: raw_label → letter is simply chr(65 + raw_label)
#   0→A, 1→B, ..., 8→I,  10→K, 11→L, ..., 24→Y
RAW_LABELS = [i for i in range(25) if i != 9]  # 24 values
RAW_TO_INDEX = {raw: idx for idx, raw in enumerate(RAW_LABELS)}
INDEX_TO_LETTER = {idx: chr(65 + raw) for idx, raw in enumerate(RAW_LABELS)}
LETTERS = [INDEX_TO_LETTER[i] for i in range(NUM_CLASSES)]


def build_model(num_classes: int = NUM_CLASSES) -> models.Model:
    """Build a deep CNN for 28x28 grayscale sign language recognition."""
    model = models.Sequential([
        # --- Block 1 ---
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # --- Block 2 ---
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # --- Block 3 ---
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # --- Classifier Head ---
        layers.Flatten(),
        layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax'),
    ])
    return model
