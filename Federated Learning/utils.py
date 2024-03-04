import tensorflow as tf


# def model():
#   mod=tf.keras.models.Sequential(
#       [
#               tf.keras.layers.Dense(16,input_shape=(29,), activation='linear'),
#               tf.keras.layers.Dense(16, activation='linear'),
          
#               tf.keras.layers.Dense(8, activation="linear"),
#       ]
#   )
#   return mod

def model():
  mod=tf.keras.models.Sequential(
      [
              tf.keras.layers.Dense(16,input_shape=(29,), activation='linear'),
              tf.keras.layers.BatchNormalization(),
              tf.keras.layers.Dense(16, activation='linear'),
              tf.keras.layers.BatchNormalization(),
              tf.keras.layers.Dense(8, activation="linear"),
      ]
  )
  return mod

datapath='/path/to/csv'
