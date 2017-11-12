from Agent import RL
from Connect4Env import connect
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Connect 4 reinforcement learning trainer')
parser.add_argument('--maxgames', type=int, default='500000',
                    help='maximum number of games to play')
parser.add_argument('--loginterval', type=int, default='10000',
                    help='how often to print the game number')
args = parser.parse_args()

Env = connect(6,7)
Agent = RL([None,6*7])

def test(player): #Player vs AI: param: what player you want to be, 1 or 2, anything else and you will play against yourself.
        done=False
        Obs=Env.reset()
        while True:
            if player==1:
                Action=Agent.predict(Obs)
            else:
                c=int(input('col#:'))
                Action=[0]*7
                Action[c-1]=1
            Obs, R, done = Env.step(np.argmax(Action), 1)
            Env.print()
            print('_________________')
            if done:
                print('Player1 wins')
                break
            if player==2:
                Action=Agent.predict(Obs)
            else:
                c=int(input('col#:'))
                Action=[0]*7
                Action[c-1]=1
            Obs, R, done = Env.step(np.argmax(Action), -1)
            Env.print()
            print('_________________')
            if done:
                print('Player2 wins')
                break


sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
Games=1000000
discount=0.6
G=0
with tf.device('/gpu:0'):
        discount=0.5
        for game in range(args.maxgames):
                if game % args.loginterval == 0: 
                    print("Game {:9d} out of {:9d}.".format(game, args.maxgames))
                Obs=Env.reset()
                R1=[]
                R2=[]
                Actions1=[]
                Actions2=[]
                while True:
                    Actions1.append([Agent.predict(Obs), Obs])
                    Obs, R, done = Env.step(np.argmax(Actions1[-1][0]), 1)
                    R1.append(R)
                    if done:
                        R2.append(-R)
                        break
                    Actions2.append([Agent.predict(Obs), Obs])
                    Obs, R, done = Env.step(np.argmax(Actions2[-1][0]), -1)
                    R2.append(R)
                    if done:
                        R1.append(-R)
                        break
                for a in range(1,len(Actions1)+1):
                    targ = [0] * 7
                    targ[np.argmax(Actions1[-a][0])] = 1
                    Agent.Update([Actions1[-a][1]], sum(R1)*discount**a)
                        

                for a in range(1,len(Actions2)+1):
                    targ = [0] * 7
                    targ[np.argmax(Actions2[-a][0])] = 1
                    Agent.Update([Actions2[-a][1]], sum(R2)*discount**a)
    
