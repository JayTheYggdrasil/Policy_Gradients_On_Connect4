import numpy as np
class connect:
    def __init__(self, height, width, debug=False):
        self.debug=debug
        self.height=height
        self.width=width
        self.reset(e=False)

    def step(self, col):
        self.M+=1
        if self.M%2==1:
            color=1
        else:
            color=-1
        while not(self.moves[col]):
            col=int(np.random.random()*self.width)
            if not(True in self.moves):
                break
        for i in range(self.height):
            if self.area[col][i]==0:
                if i==(self.height-1):
                    self.moves[col]=False
                self.area[col][i]=color
                break
        if self.checkWin()!=0:
            if color==self.checkWin():
                return self.getObs(), 1, True
            else:
                return self.getObs(), -1, True
        return self.getObs(), 0, False
            
    def getObs(self):
        return np.reshape( self.area, self.height*self.width )
            
    def reset(self, e=True):
        self.area=[]
        self.moves=[]
        self.M=0
        for i in range(self.width):
            self.area.append([])
            self.moves.append(True)
            for j in range(self.height):
                self.area[i].append(0)
        if e: return self.getObs()
        
    def checkWin(self):
        for x in range(len(self.area)):
            for y in range(len(self.area[x])):
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        try:
                            if self.area[x][y]!= 0 and [dx,dy] != [0,0] and x+3*dx >= 0 and y+3*dy>=0 and self.area[x][y]==self.area[x+dx][y+dy] and self.area[x][y]==self.area[x+2*dx][y+2*dy] and self.area[x][y]==self.area[x+3*dx][y+3*dy]:
                                if self.debug==True: print([x,y],[dx,dy])
                                return self.area[x][y]
                        except:
                            nothing=0
        return 0
    
    def print(self):
        p=[]
        for h in range(self.height):
            p.append([])
            for w in range(self.width):
                if self.area[w][h]==-1:
                    p[h].append(2)
                else:
                    p[h].append(self.area[w][h])
        p.reverse()
        for i in p:
            print(i)
