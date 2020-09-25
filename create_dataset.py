#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join, split;
import pickle;
import tensorflow as tf;

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
  spec = {'num_users': num_users, 'num_items': num_items};
  with open(split(filename)[-1] + '.pkl', 'wb') as f:
    f.write(pickle.dumps(spec));
  writer = tf.io.TFRecordWriter(join('datasets', split(filename)[-1] + ".trainset.tfrecord"));
  with open(filename + '.train.rating', 'r') as f:
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
          'negatives': tf.train.Feature(int64_list = tf.train.Int64List(value = arr[1:]))
        }
      ));
      writer.write(trainsample.SerializeToString());
      line = f.readline();
  writer.close();

if __name__ == "__main__":

  assert tf.executing_eagerly();
  create_dataset('Data/ml-1m');
  create_dataset('Data/pinterest-20');

