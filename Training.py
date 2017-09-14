from Agent import RL
from Connect4Env import connect
import tensorflow as tf

Env = connect(6,7)
Agent1 = RL([None,6*7])
Agent2 = RL([None,6*7])

sess=tf.Session()
sess.run(tf.global_variables_initializer())
with tf.device('/gpu:0'):
    while True:
        Obs=Env.reset()
        R1=0
        R2=0
        Actions1=[]
        Actions2=[]
        while True:
            Actions1.append([Agent1.predict(Obs), Obs])
            Obs, R, done = Env.step(np.argmax(Actions1[-1]))
            R1+=R
            if done:
                break
            Actions2.append(Agent2.predict(Obs))
            Obs, R, done = Env.step(np.argmax(Actions2[-1]))
            R2+=R
            if done:
                break
        for a in Actions1:
            targ = [0] * 7
            targ[np.argmax(a[0])] = 1
            loss = Agent1.loss.eval( feed_dict = {Agent1.L[0]: a[1], targ: targ} )
            Grads = Agent1.optimizer.compute_gradients( loss )
            Grads = tf.multiply(Grads,R1)
            Agent1.optimizer.apply_gradients(Grads)

        for a in Actions2:
            targ = [0] * 7
            targ[np.argmax(a[0])] = 1
            loss = Agent2.loss.eval( feed_dict = {Agent2.L[0]: a[1], targ: targ} )
            Grads = Agent2.optimizer.compute_gradients( loss )
            Grads = tf.multiply(Grads,R1)
            Agent1.optimizer.apply_gradients(Grads)
    
