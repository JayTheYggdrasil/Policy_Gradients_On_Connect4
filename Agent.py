import tensorflow as tf
class RL:
    def __init__(self, inShape):
        self._x=tf.placeholder(tf.float32, shape = inShape)
        self.L=[tf.reshape( self._x, [-1, 7, 6, 1]) ]
        
        self.L.append( self.conv2d( self.L[-1], self.W([4, 4, 1, 10]) ) )
        self.L.append( tf.reshape(self.L[-1], [1,inShape[1]*10]))
        self.L.append( self.dense(self.L[-1], [inShape[1]*10,7]) ) #Output size
        
        self.targ = tf.placeholder(tf.float32)
        self.loss = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(self.targ, self.L[-1]))

        self.optimizer = tf.train.RMSPropOptimizer(0.02)
        self.train_step=self.optimizer.minimize(self.loss)

    def Update(self, state, action, reward, lr=0.01):
        self.optimizer.learning_rate=lr*reward
        self.train_step.run( feed_dict = { self._x: state, self.targ: action})

    def predict(self, state):
        return self.L[-1].eval(feed_dict = {self._x: [state]} )
        
            
    def dense(self, x, shape):
        _W = self.W( shape )
        _B = self.B( [shape[1]] )
        return tf.matmul( x, _W ) + _B
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    def W(self, shape):
        initial=tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def B(self, shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial)
