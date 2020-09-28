#!/usr/bin/python3

from sys import argv;
import pickle;
import tensorflow as tf;
from models import GMF, MLP;

def save_model(model_name, dataset_name):

  assert model_name in ['GMF', 'MLP'];
  assert dataset_name in ['ml-1m', 'pinterest'];
  with open('datasets/' + dataset_name + '.pkl', 'rb') as f:
    spec = pickle.loads(f.read());
    print('num_users: %d num_items: %d' % (spec['num_users'], spec['num_items']));
  model = GMF(spec['num_users'], spec['num_items'], 8) if model_name == 'GMF' else MLP(spec['num_users'], spec['num_items'], [64,32,16,8]);
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
  checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  model.save('gmf.h5' if model_name == 'GMF' else 'mlp.h5');
  
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
  save_model(argv[1], argv[2]);
