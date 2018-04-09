from C4Env2 import Memory, Board
import tensorflow as tf
import numpy as np

decay = 0.99
games = 1000000
logInterval = 10000
History1 = Memory( decay )
History2 = Memory( decay )
Game = Board( )


class Agent:
    def __init__( self ):

        self.x1 = tf.placeholder( tf.float32 )
        self.x2 = tf.placeholder( tf.float32 )
        self.reward = tf.placeholder( tf.float32 )
        self.winLoss = tf.placeholder( tf.float32 )
        x = tf.concat( [ self.x1, self.x2 ], 0 )
        x = tf.reshape( x, [ 1, 6*7*2 ] )
        L = [ x ]
        L.append( tf.nn.relu( self.dense( L[-1], ( 6*7*2, 50 ) ) ) )
        L.append( tf.nn.relu( self.dense( L[-1], ( 50, 50 ) ) ) )
        L.append( tf.nn.relu( self.dense( L[-1], ( 50, 7 ) ) ) )
        #predicted = tf.nn.sigmoid( L[-1][-1] )
        L.append( tf.nn.softmax( L[-1] ) )
        self.out = L[-1]
        e = tf.constant( 1e-30 )
        loss1 = tf.reduce_sum( tf.log( L[-1] + e ) * self.reward )
        #loss2 = tf.reduce_mean( tf.pow( predicted - self.winLoss, 2 ) )
        self.tStep1 = tf.train.RMSPropOptimizer( 1e-4 ).minimize( loss1 )
        #self.tStep2 = tf.train.RMSPropOptimizer( 1e-4 ).minimize( loss2 )

        self.sess = tf.Session( )
        self.sess.run( tf.global_variables_initializer( ) )

    def update1( self, b1, b2, reward ):
        feed_dict = { self.x1: b1, self.x2: b2, self.reward: reward }
        self.sess.run( self.tStep1, feed_dict = feed_dict )

    def update2( self, b1, b2, wr ):
        feed_dict = { self.x1: b1, self.x2: b2, self.winLoss: wr }
        self.sess.run( self.tStep2, feed_dict = feed_dict )

    def getMove( self, b1, b2, view = False ):
        feed_dict = { self.x1: b1, self.x2: b2 }
        p = self.out.eval( session = self.sess, feed_dict = feed_dict )[0]
        actions = list( range( 7 ) )
        if view:
            print( p )
        return np.random.choice( actions, p = p )
    
    def dense( self, s, shape):
        _W = self.W( shape )
        _B = self.B( [shape[1]] )
        return tf.matmul( s, _W ) + _B

    def W( self, shape):
        initial=tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def B(self, shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial)   

agent = Agent( )


def test():
    Game.test( Game.player, agent )

def playGame( p1, p2 ):
    done = False
    Game.reset( )
    History1.reset( )
    s1 = Game.p1
    s2 = Game.p2
    while done != True:
        s1, s2, reward, done = Game.step( agent.getMove( s1, s2 ) )
        History1.add( s1, s2, reward )
    for i in range( History1.length ):
        R, s1, s2 = History1.getPair( i )
        agent.update1( s1, s2, R )
try:
    for i in range( games ):
        if i % logInterval == 0:
            print( i )
        playGame( agent, agent )
except KeyboardInterrupt:
    test()

