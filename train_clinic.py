#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
import pickle;
import tensorflow as tf;
from models import NeuMF, Regression, Classification;

def load_dataset():

  with open('dict.pkl', 'rb') as f:
    dictionary = pickle.loads(f.read());
  with open('dataset.pkl', 'rb') as f:
    samples = pickle.loads(f.read());
  return samples, dictionary;

def create_models(samples, dictionary):

  neumf = NeuMF(len(samples), len(dictionary), 0.5, 8, [64,32,16,8]);
  attr_nets = dict();
  for key, labels in dictionary.items():
    if labels is None:
      # regression;
      attr_nets[key] = Regression(8);
    else:
      # classification
      # NOTE: class num doesnt include blank labels
      attr_nets[key] = Classification(8, len(labels) - 1);
  return neumf, attr_nets;

def train(neumf, attr_nets, samples, dictionary):

  # optimizer
  optimizers = {key: tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5)) for key in dictionary};
  optimizers['neumf'] = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
  # checkpoints
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(neumf = neumf, attr_nets = attr_nets, optimizers = optimizers);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # log
  log = tf.summary.create_file_writer('checkpoints');
  # metrics
  metrics = {key: tf.keras.metrics.Mean(name = key, dtype = tf.float32) for key in dictionary};
  # loss
  reg_loss = tf.keras.losses.MeanSquaredError();
  cls_loss = tf.keras.losses.SparseCategoricalCrossentropy();
  # train
  while True:
    for jdx, (key, labels) in enumerate(dictionary.items()):
      users = list();
      items = list();
      labelss = list();
      for idx, sample in enumerate(samples):
        if labels is None and sample[key] is None or \
            labels is not None and sample[key] == 0: continue;
        users.append(idx);
        items.append(jdx);
        labelss.append(sample[key] if labels is None else sample[key] - 1);
      users = tf.reshape(users, (-1, 1));
      items = tf.reshape(items, (-1, 1));
      labelss = tf.reshape(labelss, (-1,));
      if labels is None:
        # regression
        with tf.GradientTape(persistent = True) as tape:
          _, features = neumf([users, items]); # features.shape = (non-blank line num, latent_dim)
          preds = attr_nets[key](features); # preds.shape = (non-blank line num, 1)
          loss = reg_loss(labelss, preds); # loss.shape = ()
      else:
        # classification
        with tf.GradientTape(persistent = True) as tape:
          _, features = neumf([users, items]); # features.shape = (non-blank line num, latent_dim)
          preds = attr_nets[key](features); # preds.shape = (non-blank line num, 1)
          loss = cls_loss(labelss, preds); # loss.shape = ()
      # report loss
      metrics[key].update_state(loss);
      with log.as_default():
        tf.summary.scalar(key, metrics[key].result(), step = optimizers[key].iterations);
      print('Step #%d %s Loss: %.6f' % (optimizers[key].iterations, key, metrics[key].result()));
      if tf.equal(optimizers[key].iterations % 10, 0): metrics[key].reset_states();
      # gradient back propagation
      grads = tape.gradient(loss, attr_nets[key].trainable_variables);
      optimizers[key].apply_gradients(zip(grads, attr_nets[key].trainable_variables));
      grads = tape.gradient(loss, neumf.trainable_variables);
      optimizers['neumf'].apply_gradients(zip(grads, neumf.trainable_variables));
      if tf.equal(optimizers[key].iterations % 10 * len(dictionary), 0):
        checkpoint.save(join('checkpoints', 'ckpt'));

if __name__ == "__main__":
  
  assert tf.executing_eagerly();
  samples, dictionary = load_dataset();
  neumf, attr_nets = create_models(samples, dictionary);
  train(neumf, attr_nets, samples, dictionary);
