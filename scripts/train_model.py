import os
import math
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, f1_score

train_dir = r'C:\Users\bussa\OneDrive\Documents\Desktop\miniporj\Driver-Drowsiness-ML\data\prepared\train'
test_dir = r'C:\Users\bussa\OneDrive\Documents\Desktop\miniporj\Driver-Drowsiness-ML\data\prepared\test'

img_width, img_height = 64, 64

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)
print("Class Indices:", train_generator.class_indices)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

os.makedirs("models", exist_ok=True)

steps = math.ceil(test_generator.samples / test_generator.batch_size)
Y_pred = model.predict(test_generator, steps=steps)
y_pred = np.argmax(Y_pred, axis=1)

y_true = test_generator.classes[:len(y_pred)]

print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys()))
print(report)

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

f1 = f1_score(y_true, y_pred, average='weighted')
print("Weighted F1 Score:", f1)

with open("models/evaluation_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
    f.write("\nWeighted F1 Score: " + str(f1))

model.save('models/drowsiness_model.h5')
print("Model saved to models/drowsiness_model.h5")



# import os
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam

# # Set paths
# train_dir = R'C:\Users\bussa\OneDrive\Documents\Desktop\miniporj\Driver-Drowsiness-ML\data\prepared\train'
# test_dir = R'C:\Users\bussa\OneDrive\Documents\Desktop\miniporj\Driver-Drowsiness-ML\data\prepared\test'

# # Image dimensions
# img_width, img_height = 64, 64

# # Image augmentation for training
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# # Only rescaling for testing
# test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# # Load training data
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_width, img_height),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=True
# )
# print(train_generator.class_indices)


# # Load testing data
# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_width, img_height),
#     batch_size=32,
#     class_mode='categorical'
# )

# # Define the CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
#     MaxPooling2D(pool_size=(2, 2)),

#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(2, activation='softmax')  # 2 classes: drowsy and alert
# ])

# # Compile model
# model.compile(
#     optimizer=Adam(learning_rate=0.001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Train model
# model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     epochs=20,
#     validation_data=test_generator,
#     validation_steps=test_generator.samples // test_generator.batch_size
# )

# # Save the model
# os.makedirs("models", exist_ok=True)
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np

# # Predict on the test set
# Y_pred = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
# y_pred = np.argmax(Y_pred, axis=1)

# # Get true labels
# y_true = test_generator.classes[:len(y_pred)]  # Ensure matching lengths

# # Print classification report
# print("\nClassification Report:")
# print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

# # Print confusion matrix
# print("Confusion Matrix:")
# print(confusion_matrix(y_true, y_pred))

# model.save('models/drowsiness_model.h5')
# print("Model saved to models/drowsiness_model.h5")



# import sys

# import os
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# sys.stdout.reconfigure(encoding='utf-8')
# os.makedirs('models', exist_ok=True)
# train_dir = R'C:\Users\bussa\OneDrive\Documents\Desktop\miniporj\Driver-Drowsiness-ML\data\prepared\train'
# test_dir = R'C:\Users\bussa\OneDrive\Documents\Desktop\miniporj\Driver-Drowsiness-ML\data\prepared\test'

# img_width, img_height = 64, 64

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(img_width, img_height),
#     batch_size=32,
#     class_mode='categorical'
# )

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_width, img_height),
#     batch_size=32,
#     class_mode='categorical'
# )


# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
#     MaxPooling2D(pool_size=(2, 2)),

#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(2, activation='softmax')  
# ])

# model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(
#     train_generator,
#     steps_per_epoch=len(train_generator),
#     epochs=20,
#     validation_data=test_generator,
#     validation_steps=len(test_generator)
# )

# model.save('models/drowsiness_model.h5')
# print("Model saved to models/drowsiness_model.h5")
