#!/usr/bin/python3

import pickle;
import tensorflow as tf;

def load_dataset():

  with open('dict.pkl', 'rb') as f:
    dictionary = pickle.loads(f.read());
  with open('dataset.pkl', 'rb') as f:
    samples = pickle.loads(f.read());
  return samples, dictionary;

def create_models(samples, dictionary):

  
  
if __name__ == "__main__":
  
  assert tf.executing_eagerly();
  samples, dictionary = load_dataset();
