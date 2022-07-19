import pathlib
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import tensorflow as tf
import tempfile

tmpdir = tempfile.mkdtemp()
mobilenet_save_path = os.path.join(tmpdir, "/home/power703/work/cgh/tmp/")
LOAD_WEIGHT = '/home/power703/work/cgh/weight/DenseNet121_paper_e1_c4_p4_bs128_data_1227_1241.14-0.92-0.27.hdf5'
model = tf.keras.models.load_model(LOAD_WEIGHT)

mobilenet_save_path = os.path.join(tmpdir, "test/1/")
tf.saved_model.save(model, mobilenet_save_path)

converter = tf.lite.TFLiteConverter.from_saved_model(mobilenet_save_path) # path to the SavedModel directory
tflite_model = converter.convert()

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
# tflite_quant_model = converter.convert()

# Save the model.
with open('e1_DenseNet121.tflite', 'wb') as f:
  f.write(tflite_model)

# tflite_models_dir = pathlib.Path("/work/power703/cgh/tflite/")
# tflite_models_dir.mkdir(exist_ok=True, parents=True)
# tflite_model_file = tflite_models_dir/"test.tflite"
# tflite_quant_model.write_bytes(tflite_model_file)
