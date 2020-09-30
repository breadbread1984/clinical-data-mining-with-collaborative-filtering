#!/usr/bin/python3

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
  for key, value in dictionary.items():
    if value is None:
      # regression;
      attr_nets[key] = Regression(8, name = key);
    else:
      # classification
      # NOTE: class num doesnt include blank value
      attr_nets[key] = Classification(8, len(value) - 1, name = key);
  return neumf, attr_nets;

def train(neumf, attr_nets, samples, dictionary):

  

if __name__ == "__main__":
  
  assert tf.executing_eagerly();
  samples, dictionary = load_dataset();
  neumf, attr_nets = create_models();
