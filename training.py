import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory('testdata/', target_size=(64, 64), batch_size=32, subset='training')
val_data = datagen.flow_from_directory('testdata/', target_size=(64, 64), batch_size=32, subset='validation')

print(f"Found {train_data.samples} training images belonging to {train_data.num_classes} classes.")
print(f"Found {val_data.samples} validation images belonging to {val_data.num_classes} classes.")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)

# Save the Model
model.save('pixel_pattern_model.h5')