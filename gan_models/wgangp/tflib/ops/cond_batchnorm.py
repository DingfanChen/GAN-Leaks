import tflib as lib

import numpy as np
import tensorflow as tf

def Batchnorm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True, labels=None, n_labels=None):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    if axes != [0,2,3]:
        raise Exception('unsupported')
    batch_mean, batch_var = tf.nn.moments(inputs, axes, keep_dims=True)
    shape = batch_mean.get_shape().as_list() # shape is [1,n,1,1]
    offset_m = lib.param(name+'.offset', np.zeros([n_labels,shape[1]], dtype='float32'))
    scale_m = lib.param(name+'.scale', np.ones([n_labels,shape[1]], dtype='float32'))
    offset = tf.nn.embedding_lookup(offset_m, labels)
    # offset = tf.Print(offset,['offset',offset])
    scale = tf.nn.embedding_lookup(scale_m, labels)
    # scale = tf.Print(scale,['scale',scale])

    moving_mean = lib.param(name + '.moving_mean', np.zeros(batch_mean.get_shape(), dtype='float32'), trainable=False)
    moving_variance = lib.param(name + '.moving_variance', np.ones(batch_var.get_shape(), dtype='float32'),trainable=False)

    def _batch_norm_training():
        return tf.nn.batch_normalization(inputs, batch_mean, batch_var, offset[:,:,None,None], scale[:,:,None,None], 1e-5)

    def _batch_norm_inference():
        # Version which blends in the current item's statistics
        mean = moving_mean[None, :, None, None]
        var = moving_variance[None, :, None, None]
        '''
        batch_size = tf.cast(tf.shape(inputs)[0], 'float32')
        mean, var = tf.nn.moments(inputs, [2,3], keep_dims=True)
        mean = ((1./batch_size)*mean) + (((batch_size-1.)/batch_size)*moving_mean)[None,:,None,None]
        var = ((1./batch_size)*var) + (((batch_size-1.)/batch_size)*moving_variance)[None,:,None,None]
        '''
        return tf.nn.batch_normalization(inputs, mean, var, offset[:,:,None,None], scale[:,:,None,None],
                                         1e-5), mean, var

    if is_training is None:
        outputs = _batch_norm_training()
    else:
        if is_training:
            outputs = _batch_norm_training()
        else:
            outputs = _batch_norm_inference()

        if update_moving_stats:
            no_updates = lambda: outputs

            def _force_updates():
                """Internal function forces updates moving_vars if is_training."""
                float_stats_iter = tf.cast(stats_iter, tf.float32)
                update_moving_mean = tf.assign(moving_mean,
                                               ((float_stats_iter / (float_stats_iter + 1)) * moving_mean) + (
                                                           (1 / (float_stats_iter + 1)) * batch_mean))
                update_moving_variance = tf.assign(moving_variance,
                                                   ((float_stats_iter / (float_stats_iter + 1)) * moving_variance) + (
                                                               (1 / (float_stats_iter + 1)) * batch_var))
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(outputs)

            if is_training:
                outputs = _force_updates()
            else:
                outputs = no_updates()

    return outputs