from Agent import RL
from Connect4Env import connect
import tensorflow as tf
import numpy as np

Env = connect(6,7)
Agent1 = RL([None,6*7])
Agent2 = RL([None,6*7])

def test(player): #Player vs AI: param: what player you want to be, 1 or 2, anything else and you will play against yourself.
        done=False
        Obs=Env.reset()
        while True:
            if player==1:
                Action=Agent1.predict(Obs)
            else:
                c=int(input('col#:'))
                Action=[0]*7
                Action[c-1]=1
            Obs, R, done = Env.step(np.argmax(Action))
            Env.print()
            print('_________________')
            if done:
                print('Player1 wins')
                break
            if player==2:
                Action=Agent2.predict(Obs)
            else:
                c=int(input('col#:'))
                Action=[0]*7
                Action[c-1]=1
            Obs, R, done = Env.step(np.argmax(Action))
            Env.print()
            print('_________________')
            if done:
                print('Player2 wins')
                break
    

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
Games=100000
with tf.device('/gpu:0'):
    while True:
        Obs=Env.reset()
        R1=0
        R2=0
        Actions1=[]
        Actions2=[]
        while True:
            Actions1.append([Agent1.predict(Obs), Obs])
            Obs, R, done = Env.step(np.argmax(Actions1[-1][0]))
            R1+=R
            if done:
                break
            Actions2.append([Agent2.predict(Obs), Obs])
            Obs, R, done = Env.step(np.argmax(Actions2[-1][0]))
            R2+=R
            if done:
                break
        for a in Actions1:
            targ = [0] * 7
            targ[np.argmax(a[0])] = 1
            Agent1.Update([a[1]], targ, R1)
                

        for a in Actions2:
            targ = [0] * 7
            targ[np.argmax(a[0])] = 1
            Agent2.Update([a[1]], targ, R2)
    
