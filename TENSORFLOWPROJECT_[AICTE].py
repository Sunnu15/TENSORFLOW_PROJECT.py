import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers, models

# Load the IMDB dataset
train_data, validation_data, test_data = tfds.load(
    name='imdb_reviews',
    split=['train[:60%]', 'train[60%:]', 'test'],
    as_supervised=True
)

# Define a custom Keras layer to wrap the TensorFlow Hub layer
class HubLayer(tf.keras.layers.Layer):
    def __init__(self, hub_url):
        super(HubLayer, self).__init__()
        self.hub_layer = hub.KerasLayer(hub_url, input_shape=[], dtype=tf.string, trainable=True)

    def call(self, inputs):
        return self.hub_layer(inputs)

# Define the embedding URL
embedding_url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

# Build the model using the custom layer
inputs = tf.keras.Input(shape=(), dtype=tf.string)
x = HubLayer(embedding_url)(inputs)
x = layers.Dense(16, activation='relu')(x)
outputs = layers.Dense(1)(x)

model = tf.keras.Model(inputs, outputs)

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data.shuffle(10000).batch(512),
    epochs=25,
    validation_data=validation_data.batch(512),
    verbose=1
)

# Evaluate the model
results = model.evaluate(test_data.batch(512), verbose=2)

# Print evaluation results
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
