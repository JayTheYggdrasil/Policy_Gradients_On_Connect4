from Agent import RL
from Connect4Env import connect
from Snapshot import Snapshot
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Connect 4 reinforcement learning trainer')
parser.add_argument('--maxgames', type=int, default='100000',
                    help='maximum number of games to play')
parser.add_argument('--loginterval', type=int, default='10000',
                    help='how often to print the game number')
args = parser.parse_args()

Env = connect(6,7)
Agent = RL([None,6*7])
Snaps = [Agent]

def regularize( x ):
    m = max(x)
    n = min(x)
    a = x.copy()
    for i in a:
        i = ((i - n)/(m - n)) * 2 - 1
    return a

def discountedR( Rt, a ):
    R = 0
    for i in range(len(Rt[a:])):
        R += Rt[a:][i] * (discount**i)
    return R

def discountAr( Arr ):
    A = []
    for Ar in Arr:
        for i in range(len(Ar)):
            A.append( discountedR( Ar, i ) )
    return A

def choice( arr ):
    s = sum( arr )
    n = np.random.uniform( high = s )
    t = 0
    c = -1
    while( t <= n ):
        c+=1
        t += arr[c]
    return c

def test(player, opponent): #Player vs AI: param: what player you want to be, 1 or 2, anything else and you will play against yourself.
    done=False
    Obs=Env.reset()
    while True:
        if player==1:
            Action=opponent.predict(Obs)[0]
            print(Action)
        else:
            c=int(input('col#:'))
            Action=[0]*7
            Action[c-1]=2
        Obs, R, done = Env.step(np.argmax(Action), 1)
        Env.print()
        print('________________________')
        if done:
            print('Player1 wins')
            break
        if player==2:
            Action=opponent.predict(Obs)[0]
            print(Action)
        else:
            c=int(input('col#:'))
            Action=[0]*7
            Action[c-1]=2
        Obs, R, done = Env.step(np.argmax(Action), -1)
        Env.print()
        print('________________________')
        if done:
            print('Player2 wins')
            break


sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
discount=0.9
e = 0.2
snapInterval = 10000
R1=[]
R2=[]
Actions1=[]
Actions2=[]
for game in range(args.maxgames):
    r1=[]
    r2=[]
    if game % args.loginterval == 0: 
        print("Game {:9d} out of {:9d}.".format(game, args.maxgames))
    #if game % snapInterval == 0:
        #Snaps.append(Snapshot(Agent))
        #sess.run(tf.global_variables_initializer())
    Obs=Env.reset()
    Opponent = Snaps[np.random.randint( low = 0, high = len(Snaps) )]
    if( np.random.randint( low = 0, high = 1 ) == 0 ):
        P1 = Agent
        P2 = Opponent
        P = 0
    else:
        P1 = Opponent
        P2 = Agent
        P = 1
    while True:
        actual = choice( P1.predict( Obs )[0] )
        Actions1.append([ Obs, actual ])
        Obs, R, done = Env.step( actual, 1) 
        r1.append(R)
        if done:
            r2.append(-R)
            break
        actual = choice( P2.predict( Obs )[0] )
        Actions2.append([ Obs, actual])
        Obs, R, done = Env.step( actual, -1)
        r2.append(R)
        if done:
            if R == 0:
                R = 1
                r2[-1] = R
            r1.append(-R)
            break
    R1.append(r1)
    R2.append(r2)
    if game % 100 == 0 and game != 0:
        R1 = regularize( discountAr(R1) )
        R2 = regularize( discountAr(R2) )
        A1 = Actions1
        A2 = Actions2
        if P == 0:
            AF = A1
            RF = R1
            PF = P1
        else:
            AF = A2
            RF = R2
            PF = P2
        for a in range(len(A1)):
            P1.Update( A1[a][0], A1[a][1], R1[a], e )
        for a in range(len(A2)):
            P2.Update( A2[a][0], A2[a][1], R2[a], e )
        sess.run(tf.global_variables_initializer())
        R1=[]
        R2=[]
        Actions1=[]
        Actions2=[]
