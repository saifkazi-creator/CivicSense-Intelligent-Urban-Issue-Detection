import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 3
EPOCHS = 10

# =========================
# LOAD DATASETS
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    r"C:/Users/Saif Kazi/OneDrive/Desktop/SIH/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    r"C:/Users/Saif Kazi/OneDrive/Desktop/SIH/valid",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    r"C:/Users/Saif Kazi/OneDrive/Desktop/SIH/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# =========================
# NORMALIZATION
# =========================
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds  = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Performance boost
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =========================
# MODEL
# =========================
model = keras.Sequential([
    layers.Conv2D(32, 3, activation="relu", input_shape=(224, 224, 3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# =========================
# EVALUATE
# =========================
test_loss, test_acc = model.evaluate(test_ds)
print("✅ Test accuracy:", test_acc)

# =========================
# SAVE
# =========================
model.save("SIH_model.h5")
print("✅ Model saved as SIH_model.h5")
