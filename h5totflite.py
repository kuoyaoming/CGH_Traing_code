import tensorflow as tf
model = tf.keras.models.load_model("C:\TWCC\weight\InceptionResNetV2_0204_1906.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
tflite_model = converter.convert()
open("C:\TWCC\weight\InceptionResNetV2_0204_1906.tflite", "wb").write(tflite_model)
