import tensorflow as tf
import tensorflow_datasets as tfds

# Code mostly from https://github.com/GoogleCloudPlatform/practical-ml-vision-book/blob/master/04_detect_segment/04b_unet_segmentation.ipynb

def read_and_preprocess(data):
  input_image = tf.image.resize(data['image'], (128, 128))
  input_mask = tf.image.resize(data['segmentation_mask'], (128, 128))

  input_image = tf.image.convert_image_dtype(input_image, tf.float32) # [0,1]
  input_mask -= 1 # {1,2,3} to {0,1,2}
  return input_image, input_mask


def augment(img, mask):
  if tf.random.uniform(()) > 0.5:
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)

  # Just testing different rotations
  if tf.random.uniform(()) < 0.33:
    # Rotate 90 degrees
    img = tf.image.rot90(img, k=1)
    mask = tf.image.rot90(mask, k=1)
  elif tf.random.uniform(()) < 0.66:
    # Rotate 270 degrees
    img = tf.image.rot90(img, k=3)
    mask = tf.image.rot90(mask, k=3)
  return img, mask


def get_datasets(batch_size):
    dataset = tfds.load('oxford_iiit_pet:3.*.*')
    train = dataset['train'].map(read_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test = dataset['test'].map(read_and_preprocess)
    train_dataset = train.cache().map(augment).shuffle(1000).batch(batch_size).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test.batch(batch_size)
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_data, test_data = get_datasets(16)
    train_data_iterator = iter(train_data)
    train_sample_input, train_sample_target = next(train_data_iterator)
    print(train_sample_input.shape, train_sample_target.shape)