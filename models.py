#!/usr/bin/python3

from os.path import exists;
import tensorflow as tf

def GMF(user_num, item_num, latent_dim = 10):

  users = tf.keras.Input((1,), dtype = tf.int32); # users.shape = (batch, 1)
  items = tf.keras.Input((1,), dtype = tf.int32); # items.shape = (batch, 1)
  users_embed = tf.keras.layers.Embedding(user_num, latent_dim, embeddings_regularizer = tf.keras.regularizers.L2())(users); # users_embed.shape = (batch, 1, latent_dim)
  items_embed = tf.keras.layers.Embedding(item_num, latent_dim, embeddings_regularizer = tf.keras.regularizers.L2())(items); # items_embed.shape = (batch, 1, latent_dim)
  users_embed = tf.keras.layers.Flatten()(users_embed); # users_embed.shape = (batch, latent_dim)
  items_embed = tf.keras.layers.Flatten()(items_embed); # items_embed.shape = (batchm latent_dim)
  results = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([users_embed, items_embed]); # results.shape = (batch, dim)
  logits = results;
  results = tf.keras.layers.Dense(units = 1, activation = tf.math.sigmoid, kernel_regularizer = tf.keras.regularizers.L2(), bias_regularizer = tf.keras.regularizers.L2())(results); # results.shape = (batch, 1)
  return tf.keras.Model(inputs = (users, items), outputs = (results, logits));

def MLP(user_num, item_num, latent_dim = 10, units = [20, 10]):

  users = tf.keras.Input((1,), dtype = tf.int32); # users.shape = (batch, 1)
  items = tf.keras.Input((1,), dtype = tf.int32); # items.shape = (batch, 1)
  users_embed = tf.keras.layers.Embedding(user_num, latent_dim, embeddings_regularizer = tf.keras.regularizers.L2())(users); # users_embed.shape = (batch, 1, latent_dim)
  items_embed = tf.keras.layers.Embedding(item_num, latent_dim, embeddings_regularizer = tf.keras.regularizers.L2())(items); # items_embed.shape = (batch, 1, latent_dim)
  users_embed = tf.keras.layers.Flatten()(users_embed); # users_embed.shape = (batch, latent_dim)
  items_embed = tf.keras.layers.Flatten()(items_embed); # items_embed.shape = (batch, latent_dim)
  results = tf.keras.layers.Concatenate(axis = -1)([users_embed, items_embed]); # results.shape = (batch, 2 * latent_dim)
  for i in range(len(units)):
    results = tf.keras.layers.Dense(units = units[i], kernel_regularizer = tf.keras.regularizers.L2(), bias_regularizer = tf.keras.regularizers.L2())(results);
  logits = results;
  results = tf.keras.layers.Dense(units = 1, activation = tf.math.sigmoid, kernel_regularizer = tf.keras.regularizers.L2(), bias_regularizer = tf.keras.regularizers.L2())(results);
  return tf.keras.Model(inputs = (users, items), outputs = (results, logits));

def NeuMF(user_num, item_num, latent_dim = 10, units = [20, 10]):

  users = tf.keras.Input((1,), dtype = tf.int32); # users.shape = (batch, 1)
  items = tf.keras.Input((1,), dtype = tf.int32); # items.shape = (batch, 1)
  gmf = tf.keras.models.load_model('gmf.h5', compile = False) if exists('gmf.h5') else GMF(user_num, item_num, latent_dim);
  mlp = tf.keras.models.load_model('mlp.h5', compile = False) if exists('mlp.h5') else MLP(user_num, item_num, latent_dim, units);
  _, mf_results = gmf([users, items]);
  _, mlp_results = mlp([users, items]);
  results = tf.keras.layers.Concatenate(axis = -1)([mf_results, mlp_results]);
  logits = results;
  results = tf.keras.layers.Dense(units = 1, activation = tf.math.sigmoid, kernel_regularizer = tf.keras.regularizers.L2(), bias_regularizer = tf.keras.regularizers.L2())(results);
  return tf.keras.Model(inputs = (users, items), outputs = (results, logits));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  gmf = GMF(10,100);
  gmf.save('gmf.h5');
  mlp = MLP(10,100);
  mlp.save('mlp.h5');
  neumf = NeuMF(10,100);
  neumf.save('neumf.h5');
  tf.keras.utils.plot_model(model = gmf, to_file = 'GMF.png', show_shapes = True, dpi = 64);
  tf.keras.utils.plot_model(model = mlp, to_file = 'MLP.png', show_shapes = True, dpi = 64);
  tf.keras.utils.plot_model(model = neumf, to_file = 'NeuMF.png', show_shapes = True, dpi = 64);
