from Agent import RL
from Connect4Env import connect
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(description='Connect 4 reinforcement learning trainer')
parser.add_argument('--maxgames', type=str, default='100000',
                    help='maximum number of games to play')
parser.add_argument('--loginterval', type=str, default='100',
                    help='how often to print the game number')
args = parser.parse_args()

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
discount=0.5
for game in range(args.maxgames):
        if game % args.loginterval: print("Game {:9d} out of {:9d}.".format(game, args.maxgames)
        Obs=Env.reset()
        R1=[]
        R2=[]
        Actions1=[]
        Actions2=[]
        while True:
            Actions1.append([Agent1.predict(Obs), Obs])
            Obs, R, done = Env.step(np.argmax(Actions1[-1][0]))
            R1.append(R)
            if done:
                R2.append(-R)
                break
            Actions2.append([Agent2.predict(Obs), Obs])
            Obs, R, done = Env.step(np.argmax(Actions2[-1][0]))
            R2.append(R)
            if done:
                R1.append(-R)
                break
        for a in range(1,len(Actions1)+1):
            targ = [0] * 7
            targ[np.argmax(Actions1[-a][0])] = 1
            Agent1.Update([Actions1[-a][1]], targ, sum(R1)*discount**a)
                

        for a in range(1,len(Actions2)+1):
            targ = [0] * 7
            targ[np.argmax(Actions2[-a][0])] = 1
            Agent1.Update([Actions2[-a][1]], targ, sum(R2)*discount**a)
    
