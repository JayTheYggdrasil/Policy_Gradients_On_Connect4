from random import randint

class Memory:
    def __init__( self, decay ):
        self.decay = decay
        self.reset( )

    def add( self, state1, state2, reward ):
        self.R.append( reward )
        self.States1.append( state1 )
        self.States2.append( state2 )
        self.length += 1
        
    def reset( self ):
        self.length = 0
        self.R = []
        self.States1 = []
        self.States2 = []
        self.processedR = []

    def getPair( self, s, e = None ):
        if len( self.processedR ) == 0:
            self.processedR = self.process( )
        if e == None:
            return self.processedR[s], self.States1[s], self.States2[s]
        else:
            return self.processedR[s:e], self.States1[s:e], self.States2[s:e]

    def process( self ):
        out = []
        temp = []
        
        i = 0
        for v in self.R:
            temp.append( v * pow( self.decay, i ) )
            i += 1
            
        for k in range( len( temp ) ):
            out.append( sum( temp[k:] ) / pow( self.decay, k ) )

        return out
    
class Board:
    def __init__( self ):
        self.reset( )
        self.doPrint = False


    class player:
        def getMove( b1, b2, view = None ):
            return int( input( "P1: Pick a collumn " ) )

    def test( self, p1, p2 ):
        self.reset( )
        x = self.doPrint
        self.doPrint = True
        b1 = self.p1
        b2 = self.p2
        while True:
            b1, b2, R, done = self.step( p1.getMove( b1, b2, view = True ) )
            if done:
                print( "P1 Wins!!!!!!" )
                break
            b1, b2, R, done = self.step( p2.getMove( b1, b2, view = True ) )
            if done:
                print( "P2 Wins!!!!!!" )
                break
        self.doPrint = x

    def step( self, collumn ):
        R = -1
        if self.p1Turn:
            terminal = self.play( self.p1, collumn )
            b1 = self.flatten( self.p1 )
            b2 = self.flatten( self.p2 )
            if terminal:
                R = 10
        else:
            terminal = self.play( self.p2, collumn )
            b1 = self.flatten( self.p2 )
            b2 = self.flatten( self.p1 )
            if terminal:
                R = 10
        self.p1Turn = not self.p1Turn
        return b1, b2, R, terminal

    def flatten( self, board ):
        out = []
        for x in board:
            for y in x:
                out.append( y )
        return out

    def printBoard( self ):
        out = ""
        for y in range(5, -1, -1 ):#loop backwards to flip board rite side up
            for x in range( 7 ):
                out += str( self.p1[x][y] + self.p2[x][y] * 2 ) + " "
            out += "\n"
        print( out )
                
    #Finds the lowest, empty spot in a collumn to play
    #Returns true if it was a winning move, false otherwise
    def play( self, board, collumn ):
        miss = True
        for y in range( 6 ):
            if self.p1[collumn][y] + self.p2[collumn][y] == 0:
                self.playXY( board, collumn, y )
                miss = False
                break
        #Collumn is full
        if miss:
            return self.play( board, randint( 0, 6 ) )

        #if the board is full, it is a draw and p2 wins
        #should be p2's turn so just return True
        total = 0
        for x in range( 7 ):
            total += sum( self.p1[x] ) + sum( self.p2[x] )
        if total == 6*7:
            return True
        
        return self.checkXY( board, collumn, y )

    #might be unessisary
    def playXY( self, board, x, y ):
        board[x][y] = 1
        if self.doPrint:
            self.printBoard( )
    #check in a certian direction for a win
    def checkDirectional( self, board, x, y, dx, dy ): 
        total = 0
        for s in range( 4 ): #checks for (x,y) being at the end of 4
            if x + dx*s >= 0 and x + dx*s < 7: #make sure we are on the board still
                if y + dy*s >= 0 and y + dy*s < 6:
                    total += board[x + dx*s][y + dy*s]
        if total == 4: #4 means we saw 4 1s which means its a win
            return True

        #Same but checks if (x, y) is the middle piece
        total = 0
        for s in range( -1, 3 ): 
            if x + dx*s >= 0 and x + dx*s < 7:
                if y + dy*s >= 0 and y + dy*s < 6:
                    total += board[x + dx*s][y + dy*s]
        if total == 4:
            return True

        return False
    def reset( self ):
        self.p1 = self.makeBoard( 7, 6 )
        self.p2 = self.makeBoard( 7, 6 )
        self.p1Turn = True
    #checks for a win in all directions from x, y point
    def checkXY( self, board, x, y ): 
        for dx in [ -1, 0, 1 ]:
            for dy in [ -1, 0, 1 ]:
                if dx !=0 or dy != 0:
                    if self.checkDirectional( board, x, y, dx, dy ):
                        return True
        return False

    def togglePrint( self ):
        self.doPrint = not( self.doPrint )
        
    def makeBoard( self, width, height ):
        board = []
        for x in range( width ):
            t = [0]
            board.append( t * height )
        return board
#game = Board()
#game.test()
