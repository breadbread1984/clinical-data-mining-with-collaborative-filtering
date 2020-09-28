#!/usr/bin/python3

from sys import argv;
import pickle;
from os import mkdir;
from os.path import join, exists;
import tensorflow as tf;
from create_dataset import parse_function;
from models import GMF, MLP;

batch_size = 256;

def pretrain(model, dataset):

  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
  # load dataset
  trainset = iter(tf.data.TFRecordDataset(join('datasets', dataset) + '.trainset.tfrecord').repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
  testset = iter(tf.data.TFRecordDataset(join('datasets', dataset) + '.testset.tfrecord').repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
  # checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # log
  log = tf.summary.create_file_writer('checkpoints');
  # train
  avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  while True:
    users, items, labels = next(trainset);
    with tf.GradientTape() as tape:
      predicts, _ = model([users, items]);
      loss = tf.keras.losses.BinaryCrossentropy()(labels, predicts);
    avg_loss.update_state(loss);
    if tf.equal(optimizer.iterations % 100, 0):
      with log.as_default():
        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
      print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.result()));
      if avg_loss.result() < 0.01: break;
      avg_loss.reset_states();
    grads = tape.gradient(loss, model.trainable_variables);
    optimizer.apply_gradients(zip(grads, model.trainable_variables));
    if tf.equal(optimizer.iterations % 100, 0):
      checkpoint.save(join('checkpoints', 'ckpt'));
      test_loss = tf.keras.metrics.Mean(name = 'test loss', dtype = tf.float32);
      for i in range(10):
        users, items, labels = next(testset);
        predicts, _ = model([users, items]);
        loss = tf.keras.losses.BinaryCrossentropy()(labels, predicts);
        test_loss.update_state(loss);
      with log.as_default():
        tf.summary.scalar('test loss', test_loss.result(), step = optimizer.iterations);
  model.save('model.h5');

if __name__ == "__main__":
  
  if len(argv) != 3:
    print('Usage: %s <model> <dataset name>' % argv[0]);
    exit(1);
  if argv[1] not in ['GMF', 'MLP']:
    print('model must be in (GMF, MLP)');
    exit(1);
  if argv[2] not in ['ml-1m', 'pinterest']:
    print('dataset name must be in (ml-1m, pinterest)');
    exit(1);
  with open('datasets/' + argv[2] + '.pkl', 'rb') as f:
    spec = pickle.loads(f.read());
    print('num_users: %d num_items: %d' % (spec['num_users'], spec['num_items']));
  model = GMF(spec['num_users'], spec['num_items'], 8) if argv[1] == 'GMF' else MLP(spec['num_users'], spec['num_items'], [64,32,16,8]);
  pretrain(model, argv[2]);
