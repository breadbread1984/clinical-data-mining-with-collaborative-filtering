#!/usr/bin/python3

from sys import argv;
import pickle;
import tensorflow as tf;
from kerastuner.tuners import RandomSearch;
from models import NeuMF;

batch_size = 256;

def train(neumf):

  trainset = tf.data.TFRecordDataset(join('datasets', dataset) + '.trainset.tfrecord').repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = tf.data.TFRecordDataset(join('datasets', dataset) + '.testset.tfrecord').repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  tuner = RandomSearch(
    neumf,
    objective = 'val_accuracy',
    max_trials = 100,
    directory = 'my_dir',
    project_name = 'neumf');
  tuner.search(trainset, epochs = 5, validation_data = testset);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  if len(argv) != 2:
    print('Usage: %s <dataset name>' % argv[0]);
    exit(1);
  if argv[1] not in ['ml-1m', 'pinterest']:
    print('dataset name must be in (ml-1m, pinterest)');
    exit(1);
  with open('datasets/' + argv[1] + '.pkl', 'rb') as f:
    spec = pickle.loads(f.read());
    print('num_users: %d num_items: %d' % (spec['num_users'], spec['num_items']));
  neumf = NeuMF(spec['num_users'], spec['num_items'], hp.Float('alpha', 0.1, 0.9, step = 0.1), 8, [64,32,16,8]);
  neumf.compile(optimizer = keras.optimizers.SGD(hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])), 
                loss = tf.keras.losses.BinaryCrossentropy(), 
                metric = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32));
  main(neumf);
