import tensorflow as tf
from Snapshot import Snapshot
class RL:
    def __init__(self, inShape):
        self.shape = inShape
        with tf.name_scope('original'):
            self._x=tf.placeholder(tf.float32, shape = inShape)
            self.L=[tf.reshape( self._x, [-1, 7, 6, 1]) ]
            self.Qfix = tf.constant( 1e-35, shape = ( 1, 7 ) )
            self.L.append( self.conv2d( self.L[-1], self.W([4, 4, 1, 20]) ) )
            self.L.append( tf.reshape(self.L[-1], [1,12*20] ) )
            self.L.append( tf.nn.relu( self.L[-1] ) )
            self.L.append( self.dense(self.L[-1], [12*20,7]) )
            self.L.append( tf.nn.softmax( self.L[-1] ) )

            self.reward = tf.placeholder(tf.float32, name='reward')
            self.old = tf.placeholder(tf.float32, name='old_state')
            self.e = tf.placeholder(tf.float32)
            self.r = tf.reduce_mean( ( self.L[-1] / self.old ) * self.reward )
            self.loss = tf.reduce_min( [self.r * self.reward,
                                        tf.clip_by_value( self.r, 1-self.e, 1 + self.e) * self.reward] )
            #self.loss = self.reward * tf.reduce_mean( tf.log( self.L[-1] + self.Qfix ))
            self.optimizer = tf.train.AdamOptimizer(1e-4)
            self.train_step=self.optimizer.minimize(self.loss)
            self.O = Snapshot( self )

    def Update(self, state, reward, e):
        self.train_step.run( feed_dict = { self._x: [state], self.reward: reward, self.old: self.O.predict( state ), self.e: e} )
        
    def Snap(self):
        del self.O
        self.O = Snapshot( self )
        
    def predict(self, state):
        return self.L[-1].eval(feed_dict = {self._x: [state]} )
        
    def dense(self, x, shape):
        _W = self.W( shape )
        _B = self.B( [shape[1]] )
        return tf.matmul( x, _W ) + _B
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def W(self, shape):
        initial=tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def B(self, shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial)
