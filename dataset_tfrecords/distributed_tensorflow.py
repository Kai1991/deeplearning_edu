import argparse
import sys
import numpy as np
import os

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = None

def get_tfrecord_filenames():
    train_tfrecord_filenames = []
    train_tfrecord_folder = 'tfrecords/train_tfrecords'

    for filename in os.listdir(train_tfrecord_folder):
        train_tfrecord_filenames.append(os.path.join(train_tfrecord_folder, filename))
    return train_tfrecord_filenames

def create_dataset():
    filename_placeholder = tf.placeholder(tf.string, shape=[None])

    def _parse_function(example_proto):
        features = {"img_raw": tf.FixedLenFeature((), tf.string, default_value=""),
                    "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
        parsed_features = tf.parse_single_example(example_proto, features)
        img_decoded = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
        return img_decoded, parsed_features['label']

    dataset = tf.data.TFRecordDataset(filename_placeholder)
    dataset = dataset.map(_parse_function)

    batch_size = 20
    dataset = dataset.prefetch(buffer_size=2000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.repeat()
    # iterator = dataset.make_one_shot_iterator()
    iterator = dataset.make_initializable_iterator()
    batch_imgs, batch_labels = iterator.get_next()
    batch_imgs = tf.cast(batch_imgs, tf.float32)
    batch_labels = tf.cast(batch_labels, tf.int64)
    return batch_imgs, batch_labels, filename_placeholder, iterator

def cnn_model(x, y):
    with tf.name_scope("inputs"):
        is_training = tf.placeholder(tf.bool, name="is_training")
        x_image = tf.reshape(x, [-1, 32, 32, 3])
    with tf.name_scope("model"):
        conv1 = tf.layers.conv2d(inputs=x_image,
                                 filters=32,
                                 kernel_size=[5,5],
                                 padding='same',
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=[2,2],
                                        strides=2)

        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=[5,5],
                                 padding='same',
                                 activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=[2,2],
                                        strides=2)

        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=64,
                                 kernel_size=[5,5],
                                 padding='same',
                                 activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                        pool_size=[2,2],
                                        strides=2)

        pool3_flatten = tf.layers.flatten(pool3)
        dense = tf.layers.dense(inputs=pool3_flatten,
                                units=128,
                                activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense,
                                    rate=0.1,
                                    training=is_training)
        logits = tf.layers.dense(inputs=dropout,
                                units=10)
    return is_training, logits

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      batch_imgs, batch_labels, filename_placeholder, iterator = create_dataset()
      is_training, logits = cnn_model(batch_imgs, batch_labels)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_labels, logits=logits))
      correct_prediction = tf.equal(tf.argmax(logits, 1), batch_labels)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.001).minimize(
          loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="./distributed_run/train_logs",
                                           hooks=hooks) as mon_sess:
      mon_sess.run(iterator.initializer, feed_dict={filename_placeholder: get_tfrecord_filenames()})
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See <a href="../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        loss_val, accuracy_val, _, global_step_val = mon_sess.run([loss, accuracy, train_op, global_step], feed_dict={is_training: True})
        tf.logging.info("gloal_step: %d, loss_val: %3.5f, accuracy: %3.5f" % (global_step_val, loss_val, accuracy_val))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
