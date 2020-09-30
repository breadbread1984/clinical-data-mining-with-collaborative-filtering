#!/usr/bin/python3

from os.path import exists;
import tensorflow as tf

def GMF(user_num, item_num, latent_dim = 10):

  users = tf.keras.Input((1,), dtype = tf.int32); # users.shape = (batch, 1)
  items = tf.keras.Input((1,), dtype = tf.int32); # items.shape = (batch, 1)
  users_embed = tf.keras.layers.Embedding(user_num, latent_dim, embeddings_regularizer = tf.keras.regularizers.L2(), name = "gmf_user_embed")(users); # users_embed.shape = (batch, 1, latent_dim)
  items_embed = tf.keras.layers.Embedding(item_num, latent_dim, embeddings_regularizer = tf.keras.regularizers.L2(), name = "gmf_item_embed")(items); # items_embed.shape = (batch, 1, latent_dim)
  users_embed = tf.keras.layers.Flatten(name = "gmf_user_flatten")(users_embed); # users_embed.shape = (batch, latent_dim)
  items_embed = tf.keras.layers.Flatten(name = "gmf_item_flatten")(items_embed); # items_embed.shape = (batchm latent_dim)
  results = tf.keras.layers.Lambda(lambda x: x[0] * x[1], name = "gmf_logits")([users_embed, items_embed]); # results.shape = (batch, dim)
  logits = results;
  results = tf.keras.layers.Dense(units = 1, activation = tf.math.sigmoid, kernel_regularizer = tf.keras.regularizers.L2(), bias_regularizer = tf.keras.regularizers.L2(), name = "gmf_dense")(results); # results.shape = (batch, 1)
  return tf.keras.Model(inputs = (users, items), outputs = (results, logits));

def MLP(user_num, item_num, units = [20, 10]):

  users = tf.keras.Input((1,), dtype = tf.int32); # users.shape = (batch, 1)
  items = tf.keras.Input((1,), dtype = tf.int32); # items.shape = (batch, 1)
  users_embed = tf.keras.layers.Embedding(user_num, units[0] // 2, embeddings_regularizer = tf.keras.regularizers.L2(), name = "mlp_user_embed")(users); # users_embed.shape = (batch, 1, latent_dim)
  items_embed = tf.keras.layers.Embedding(item_num, units[0] // 2, embeddings_regularizer = tf.keras.regularizers.L2(), name = "mlp_item_embed")(items); # items_embed.shape = (batch, 1, latent_dim)
  users_embed = tf.keras.layers.Flatten(name = "mlp_user_flatten")(users_embed); # users_embed.shape = (batch, latent_dim)
  items_embed = tf.keras.layers.Flatten(name = "mlp_item_flatten")(items_embed); # items_embed.shape = (batch, latent_dim)
  results = tf.keras.layers.Concatenate(axis = -1, name = "mlp_concat")([users_embed, items_embed]); # results.shape = (batch, 2 * latent_dim)
  for i in range(1, len(units)):
    results = tf.keras.layers.Dense(units = units[i], kernel_regularizer = tf.keras.regularizers.L2(), bias_regularizer = tf.keras.regularizers.L2(), name = "mlp_dense_" + str(i))(results);
  logits = results;
  results = tf.keras.layers.Dense(units = 1, activation = tf.math.sigmoid, kernel_regularizer = tf.keras.regularizers.L2(), bias_regularizer = tf.keras.regularizers.L2(), name = "mlp_dense")(results);
  return tf.keras.Model(inputs = (users, items), outputs = (results, logits));

class CustomModel(tf.keras.Model):

  def compile(self, optimizer, loss, metric):

    super(CustomModel, self).compile();
    self.optimizer = optimizer;
    self.loss_fn = loss;
    self.metric_fn = metric;

  def train_step(self, data):

    users, items, labels = data;
    with tf.GradientTape() as tape:
      predicts, _ = self([users, items]);
      loss = self.loss_fn(labels, predicts);
    grads = tape.gradient(loss, self.trainable_variables);
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables));
    self.metric_fn.update_state(loss);
    return {'loss': loss};

  def test_step(self, data):

    users, items, labels = data;
    with tf.GradientTape() as tape:
      predicts, _ = self([users, items]);
      loss = self.loss_fn(labels, predicts);
    grads = tape.gradient(loss, self.trainable_variables);
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables));
    self.metric_fn.update_state(loss);
    return {'loss': loss};

def NeuMF(user_num, item_num, alpha = 0.5, latent_dim = 10, units = [20, 10]):

  assert 0 < alpha < 1;
  users = tf.keras.Input((1,), dtype = tf.int32); # users.shape = (batch, 1)
  items = tf.keras.Input((1,), dtype = tf.int32); # items.shape = (batch, 1)
  gmf = tf.keras.models.load_model('gmf.h5', compile = False) if exists('gmf.h5') else GMF(user_num, item_num, latent_dim);
  gmf._name = 'gmf';
  mlp = tf.keras.models.load_model('mlp.h5', compile = False) if exists('mlp.h5') else MLP(user_num, item_num, units);
  mlp._name = 'mlp';
  _, mf_results = gmf([users, items]);
  _, mlp_results = mlp([users, items]);
  results = tf.keras.layers.Lambda(lambda x, a: tf.concat([a * x[0], (1-a) * x[1]], axis = -1), arguments = {'a': alpha}, name = "neumf_logits")([mf_results, mlp_results]);
  logits = results;
  results = tf.keras.layers.Dense(units = 1, activation = tf.math.sigmoid, kernel_regularizer = tf.keras.regularizers.L2(), bias_regularizer = tf.keras.regularizers.L2(), name = "neumf_dense")(results);
  return CustomModel(inputs = (users, items), outputs = (results, logits));

def Regression(latent_dim = 10, name = None):

  inputs = tf.keras.Input((latent_dim,)); # inputs.shape = (batch, latent_dim)
  results = tf.keras.layers.Dense(units = 1, kernel_regularizer = tf.keras.regularizers.L2(), bias_regularizer = tf.keras.regularizers.L2(), name = name)(inputs);
  return tf.keras.Model(inputs = inputs, outputs = results);

def Classification(latent_dim = 10, class_num = 10, name = None):

  inputs = tf.keras.Input((latent_dim,)); # inputs.shape = (batch, latent_dim)
  results = tf.keras.layers.Dense(units = class_num, kernel_regularizer = tf.keras.regularizers.L2(), bias_regularizer = tf.keras.regularizers.L2(), name = name)(inputs);
  results = tf.keras.layers.Softmax()(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  gmf = GMF(10,100);
  mlp = MLP(10,100);
  neumf = NeuMF(10, 100);
  neumf.save('neumf.h5');
  tf.keras.utils.plot_model(model = gmf, to_file = 'GMF.png', show_shapes = True, dpi = 64);
  tf.keras.utils.plot_model(model = mlp, to_file = 'MLP.png', show_shapes = True, dpi = 64);
  tf.keras.utils.plot_model(model = neumf, to_file = 'NeuMF.png', show_shapes = True, dpi = 64);
