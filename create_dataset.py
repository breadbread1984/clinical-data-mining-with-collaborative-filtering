#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join, split;
import pickle;
import numpy as np;
import tensorflow as tf;

num_negatives = 4;

def parse_function(serialized_example):

  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'user': tf.io.FixedLenFeature((), dtype = tf.int64),
      'item': tf.io.FixedLenFeature((), dtype = tf.int64),
      'rating': tf.io.FixedLenFeature((), dtype = tf.float32)
    }
  );
  user = feature['user'];
  item = feature['item'];
  label = 0 if feature['rating'] == -1 else 1;
  return user, item, label;

def create_dataset(filename):

  if not exists('datasets'): mkdir('datasets');
  num_users, num_items = 0,0;
  with open(filename + '.train.rating', 'r') as f:
    line = f.readline();
    while line != None and line != "":
      arr = line.split('\t');
      u,i = int(arr[0]), int(arr[1]);
      num_users = max(num_users, u);
      num_items = max(num_items, i);
      line = f.readline();
    num_users += 1;
    num_items += 1;
  spec = {'num_users': num_users, 'num_items': num_items};
  with open(join('datasets', split(filename)[-1]) + '.pkl', 'wb') as f:
    f.write(pickle.dumps(spec));
  with open(filename + '.train.rating', 'r') as f:
    line = f.readline();
    samples = dict();
    while line != None and line != "":
      arr = line.split('\t');
      user, item, rating = int(arr[0]), int(arr[1]), float(arr[2]);
      if user not in samples: samples[user] = dict();
      samples[user][item] = rating;
      line = f.readline();
  writer = tf.io.TFRecordWriter(join('datasets', split(filename)[-1] + ".trainset.tfrecord"));
  for user, items in samples.items():
    # positive samples
    for item, rating in items.items():
      trainsample = tf.train.Example(features = tf.train.Features(
        feature = {
          'user': tf.train.Feature(int64_list = tf.train.Int64List(value = [user])),
          'item': tf.train.Feature(int64_list = tf.train.Int64List(value = [item])),
          'rating': tf.train.Feature(float_list = tf.train.FloatList(value = [rating]))
        }
      ));
      writer.write(trainsample.SerializeToString());
    # negative samples
    for i in range(num_negatives):
      item = np.random.randint(num_items);
      while item in items:
        item = np.random.randint(num_items);
      trainsample = tf.train.Example(features = tf.train.Features(
        feature = {
          'user': tf.train.Feature(int64_list = tf.train.Int64List(value = [user])),
          'item': tf.train.Feature(int64_list = tf.train.Int64List(value = [item])),
          'rating': tf.train.Feature(float_list = tf.train.FloatList(value = [-1]))
        }
      ));
      writer.write(trainsample.SerializeToString());
  writer.close();
  writer = tf.io.TFRecordWriter(join('datasets', split(filename)[-1] + '.testset.tfrecord'));
  with open(filename + '.test.rating', 'r') as f:
    line = f.readline();
    while line != None and line != "":
      arr = line.split('\t');
      user, item, rating = int(arr[0]), int(arr[1]), float(arr[2]);
      trainsample = tf.train.Example(features = tf.train.Features(
        feature = {
          'user': tf.train.Feature(int64_list = tf.train.Int64List(value = [user])),
          'item': tf.train.Feature(int64_list = tf.train.Int64List(value = [item])),
          'rating': tf.train.Feature(float_list = tf.train.FloatList(value = [rating]))
        }
      ));
      writer.write(trainsample.SerializeToString());
      line = f.readline();
  writer.close();
  writer = tf.io.TFRecordWriter(join('datasets', split(filename)[-1] + '.negative.tfrecord'));
  with open(filename + '.test.negative', 'r') as f:
    line = f.readline();
    while line != None and line != "":
      arr = line.split('\t');
      trainsample = tf.train.Example(features = tf.train.Features(
        feature = {
          'negatives': tf.train.Feature(int64_list = tf.train.Int64List(value = np.array(arr[1:]).astype('int')))
        }
      ));
      writer.write(trainsample.SerializeToString());
      line = f.readline();
  writer.close();

if __name__ == "__main__":

  assert tf.executing_eagerly();
  create_dataset('Data/ml-1m');
  create_dataset('Data/pinterest-20');

