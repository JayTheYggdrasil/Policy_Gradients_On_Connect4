import tensorflow as tf
class RL:
    def __init__(self, inShape):

        self.L=[tf.placeholder(tf.float32, shape = inShape)]
        
        self.L.append( self.conv2d( self.L[-1], self.W([4, 4, 1, 10]) ) )
        self.L.append( tf.reshape(self.L[-1], inShape[0]*inShape[1]))
        self.L.append( self.dense(self.L[-1], [inShape[0]*inShape[1],7]) ) #Output size
        
        targ = tf.placeholder(tf.float32)
        self.loss = tf.losses.softmax_cross_entropy(targ, self.L[-1])

        self.optimizer = tf.train.RMSPropOptimizer(0.02)

    def predict(state):
        return self.L[-1].eval(feed_dict = {self.L[0]: state} )
        
            
    def dense(x, shape):
        _W = W( shape )
        _B = B( shape[1] )
        return tf.matmul( x, _W ) + _B
    
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    def W(shape):
        initial=tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def B(shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial)
