import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from src.LoadDogs import create_training_data
import matplotlib.pyplot as plt
import numpy as np

#fit by (112, 112)data
(train_images112, train_labels112), (test_images112, test_labels112) = create_training_data(IMG_SIZE = 112)

model9 = models.Sequential()
model9.add(layers.ZeroPadding2D(padding=(1, 1), input_shape=(112, 112, 3)))
model9.add(layers.Conv2D(32, (3, 3), activation='relu'))
model9.add(layers.ZeroPadding2D(padding=(1, 1)))
model9.add(layers.Conv2D(64, (3, 3), activation='relu'))
model9.add(layers.MaxPooling2D((2, 2)))

model9.add(layers.ZeroPadding2D(padding=(1, 1)))
model9.add(layers.Conv2D(64, (3, 3), activation='relu'))
model9.add(layers.ZeroPadding2D(padding=(1, 1)))
model9.add(layers.Conv2D(64, (3, 3), activation='relu'))
model9.add(layers.ZeroPadding2D(padding=(1, 1)))
model9.add(layers.Conv2D(64, (3, 3), activation='relu'))
model9.add(layers.MaxPooling2D((2, 2)))

model9.add(layers.Conv2D(64, (3, 3), activation='relu'))
model9.add(layers.ZeroPadding2D(padding=(1, 1)))
model9.add(layers.Conv2D(64, (3, 3), activation='relu'))
model9.add(layers.ZeroPadding2D(padding=(1, 1)))
model9.add(layers.Conv2D(64, (3, 3), activation='relu'))
model9.add(layers.MaxPooling2D((2, 2)))

model9.add(layers.Conv2D(128, (5, 5), activation='relu'))


model9.add(layers.Flatten())
model9.add(layers.Dense(64, activation='relu'))
model9.add(layers.Dense(2,activation='softmax'))

#model9.summary()

#compile, fit
model9.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model9.fit(train_images112, train_labels112, epochs=10)


test_loss, test_acc = model9.evaluate(test_images112, test_labels112)

#evaluation of the model9
probability_model9 = tf.keras.Sequential([model5, tf.keras.layers.Softmax()])
predictions9 = probability_model9.predict(test_images112)

num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

batch_mask = np.random.choice(len(test_images112), num_images)
img_batch = test_images[batch_mask]
label_batch = test_labels[batch_mask]

for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions9[i], label_batch, img_batch)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions9[i], label_batch)
plt.tight_layout()
plt.show()