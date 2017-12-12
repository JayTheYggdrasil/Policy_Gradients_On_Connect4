import tensorflow as tf
class Snapshot:
    def __init__(self, Model):
        self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='original')
        self.cv = 0
        inShape = Model.shape
        self._x=tf.placeholder(tf.float32, shape = inShape)
        self.L=[tf.reshape( self._x, [-1, 7, 6, 1]) ]
        self.Qfix = tf.constant( 1e-35, shape = ( 1, 7 ) )
        
        self.L.append( self.conv2d( self.L[-1], self.W([4, 4, 1, 20]) ) )
        self.L.append( tf.reshape(self.L[-1], [1,12*20] ) )
        self.L.append( tf.nn.relu( self.L[-1] ) )
        self.L.append( self.dense(self.L[-1], [12*20,7]) )
        self.L.append( tf.nn.softmax( self.L[-1] ) )

        self.reward = tf.placeholder(tf.float32)
        self.loss = self.reward * tf.reduce_mean( tf.log(self.L[-1] + self.Qfix))
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train_step=self.optimizer.minimize(self.loss)
        
    def predict(self, state):
        return self.L[-1].eval(feed_dict = {self._x: [state]} )
    
    def Update(self, state, reward):
        self.train_step.run( feed_dict = { self._x: state, self.reward: reward})

    def dense(self, x, shape):
        _W = self.W( shape )
        _B = self.B( [shape[1]] )
        return tf.matmul( x, _W ) + _B
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def W(self, shape):
        initial=tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(self.vars[self.cv])
        self.cv += 1
        return var

    def B(self, shape):
        initial=tf.constant(0.1,shape=shape)
        var = tf.Variable(self.vars[self.cv])
        self.cv += 1
        return var
