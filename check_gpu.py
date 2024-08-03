import tensorflow as tf

def print_num_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs detected: {len(gpus)}")

print_num_gpus()