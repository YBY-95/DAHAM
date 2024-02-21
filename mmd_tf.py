import tensorflow as tf
import utils
from functools import partial
import glob
from dataset import make_sign_dataset
import os as os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def maximum_mean_discrepancy(x, y, kernel=utils.gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.
  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
  """Adds a similarity loss term, the MMD between two representations.
  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.
  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    scope: optional name scope for summary tags.
  Returns:
    a scalar tensor representing the MMD loss value.
  """
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(
      utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

  loss_value = maximum_mean_discrepancy(
      source_samples, target_samples, kernel=gaussian_kernel)

  return loss_value


if __name__ == '__main__':
    cond = ['0', '1', '2', '3']
    data_name = '021'
    dataset_name = 'CWRU'
    for i in cond:
        for j in cond:
            # i 为源域 j为目标域
            if i == j:
                continue
            data_path = r'D:/DATABASE/CWRU_xjs/sample/12k/Drive_End/' + data_name
            sign_s_list = glob.glob(data_path + r'/' + i + r'/**.mat')
            sign_t_list = glob.glob(data_path + r'/' + j + r'/**.mat')

            data_s, data_s_t, _, whole_data_s, _, _ = make_sign_dataset(sign_s_list, batch_size=256, dataset_name=dataset_name, shuffle=None)
            data_t, _, _, whole_data_t, _, _ = make_sign_dataset(sign_t_list, batch_size=256, dataset_name=dataset_name, shuffle=None)

            data_s_iter = iter(data_s)
            data_t_iter = iter(data_t)
            data_s_t_iter = iter(data_s_t)
            signal_s_t = next(data_t_iter)[0]
            signal_s = next(data_s_iter)[0]
            signal_t = next(data_t_iter)[0]
            signal_s_t = tf.cast(tf.reshape(signal_s_t, [signal_s_t.shape[0], signal_s_t.shape[1]]), tf.float32)
            signal_s = tf.cast(tf.reshape(signal_s, [signal_s.shape[0], signal_s.shape[1]]), tf.float32)
            signal_t = tf.cast(tf.reshape(signal_t, [signal_t.shape[0], signal_t.shape[1]]), tf.float32)

            # signal_s = tf.reshape(data_s[0], [len(data_s[0]), -1])
            # signal_t = tf.reshape(data_t[0], [len(data_t[0]), -1])
            print(mmd_loss(signal_s,signal_s_t,1))
            loss = mmd_loss(signal_s, signal_t, 1)
            print('source field:', i, 'target field', j, 'mmd_loss=', loss)
