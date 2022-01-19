#flappybird.py Rose Ridder 04-21-2016
''' This is a program to play the game flappy bird.
It includes double pipes and a score board.
The high score file must be downloaded: highScores.txt.
'''
from random import *
from time import sleep
from graphics import *

def main():
    width=600
    height=600
    w=GraphWin("Flappy Bird", width, height)
    w.setBackground('turquoise')
    pp=Pipe(w)
    bb=Bird(w, 20)
    score=0
    dispScore=Text(Point(50,20), "Score: 0")
    dispScore.draw(w)
    dy=0
    start=Text(Point(width/2,height/2),
               "Press any key to begin.\nUp arrow to fly\nSpace to pause")
    start.draw(w)
    w.getKey()
    start.undraw()
    while bb.onScreen()==True:# and bb.touching(pp)==False:
        dy+=.05
        key=w.checkKey()
        if key != None:
            if key=='q':
                w.close()
                return
            if key == 'Up':
                mv=-w.getHeight()/20
                bb.move(mv)
                dy=0
            if key =='space':
                pause=Text(Point(width/2,height/2), "Press any key to continue\n(q to quit)")
                pause.draw(w)
                key=w.getKey()
                if key =='q':
                    w.close()
                    return
                pause.undraw()
        bb.move(dy)
        pp.move(-1)
        try: #in a try except because at the begining, p1 does not exist
            p1.move(-1)
            if bb.touching(p1)==True:
                break
            score+=Score(p1,bb)
        except:
            pass
        if pp.newPipe()==True: #occurs at x=200
            p1=pp #to preserve old pipe until it is off the screen
            pp=Pipe(w)
            dispScore.undraw()
            dispScore.draw(w) #to draw in front of new pipe
        sleep(.01)
        dispScore.setText("Score: %s" %(score))
    end(w, score)

def end(w, score):
    gameOver=Text(Point(w.getWidth()/2, w.getHeight()/2), "Game Over\nScore: %s"
                  %(score))
    gameOver.setSize(36)
    gameOver.setStyle('bold')
    gameOver.draw(w)
    for i in range(10):
        key=w.checkKey()
        if key=='q':
            w.close()
            break
        w.setBackground('red')
        gameOver.setTextColor('white')
        sleep(.2)
        w.setBackground('white')
        gameOver.setTextColor('black')
        sleep(.2)
    w.close()
    highScore(score)
    
def highScore(score):
    scores=[]
    fr=open('highScores.txt', 'r')
    for line in fr:
        l=line.strip()
        w=l.split()
        scr=int(w[0])
        w=[scr, w[1]]
        scores.append(w)
    if score<scores[9][0]:
        return
    scores.pop()
    name=raw_input("Enter your name to be added to the high scores list ")
    entry=[score, name]
    if score>scores[0][0]:
        scores.insert(0,entry)
    elif score<scores[8][0]:
        scores.append(entry)
    else:
        for i in range(len(scores)):
            if scores[i][0]<=score:
                scores.insert(i,entry)
                break
    fw=open('highScores.txt', 'w')
    for i in range(10):
        fw.write('%d %s' %(scores[i][0], scores[i][1]))
        fw.write('\n')
    fw.close()
    print ("Flappy Bird High Scores:")
    for i in scores:
        print ('%12s: %3d' %(i[1], i[0]))


    
def Score(pipeobj, birdobj):
    p=pipeobj.getRightX()
    b=birdobj.birdy[0].getCenter().getX()-birdobj.birdy[0].getRadius()
    if b==p:
        return 1
    else:
        return 0
    
class Pipe(object):
    def __init__(self, win):
        self.win=win
        self.gap=win.getHeight()/12
        
        start=win.getWidth()
        height=self.win.getHeight()
        gapCenter=randrange(self.gap, height-self.gap)
        topPipe=Rectangle(Point(start, gapCenter-self.gap), Point(start+self.gap, 0))
        topPipe.setFill('dark green')
        bottomPipe=Rectangle(Point(start, gapCenter+self.gap), Point(start+self.gap, height))
        bottomPipe.setFill('dark green')
        topPipe.draw(win)
        bottomPipe.draw(win)
        
        self.pipeFull=[topPipe, bottomPipe]

    def getRightX(self):
        return self.pipeFull[0].getP2().getX()

    def getLeftX(self):
        return self.pipeFull[0].getP1().getX()

    def getTopY(self):
        return self.pipeFull[0].getP1().getY()

    def getBottomY(self):
        return self.pipeFull[1].getP1().getY()

    def move(self, dx):
        for piece in self.pipeFull:
            piece.move(dx,0)

    def newPipe(self): #(formerly known as offScren, but with alterations)
        x=self.getRightX()
        if x<=200:
            return True
        else:
            return False

class Bird(object):
    def __init__(self, win, size):
        self.win=win
        self.size=size
        pt=Point(100,100)
        body=Circle(pt, size)
        body.setFill('orange')
        body.draw(win)
        
        eyept=pt.clone()
        eyept.move(size/3, -size/5)
        eye=Circle(eyept, size/4)
        eye.setFill('white')
        eye.draw(win)
        
        pupilpt=eyept.clone()
        pupil=Circle(pupilpt, size/8)
        pupil.setFill('black')
        pupil.draw(win)
        
        beakpt=pt.clone()
        beakpt.move(size*1.2,0)
        beakt=beakpt.clone()
        beakt.move(-size/3, -size/4)
        beakb=beakpt.clone()
        beakb.move(-size/3, size/4)
        beak=Polygon(beakpt, beakb, beakt)
        beak.setFill('yellow')
        beak.draw(win)

        self.birdy=[body, eye, pupil, beak]
        
    def move(self, dy):
        for piece in self.birdy:
            piece.move(0,dy)

    def onScreen(self):
        '''this is written to also check the sides in order to add the
        ability to possibly move the bird forward and backward later
        on'''
        winright=self.win.getWidth()
        winbottom=self.win.getHeight()
        birdypt=self.birdy[0].getCenter()
        birdysize=self.birdy[0].getRadius()
        if birdypt.getY()-birdysize<=0:
            return False
        elif birdypt.getY()+birdysize>=winbottom:
            return False
        elif birdypt.getX()-birdysize<=0:
            return False
        elif birdypt.getX()+birdysize>=winright:
            return False
        else:
            return True

    def touching(self, pipeobj):
        birdypt=self.birdy[0].getCenter()        
        birdysize=self.birdy[0].getRadius()
        birdyright=birdypt.getX()+birdysize
        birdyleft=birdypt.getX()-birdysize
        birdytop=birdypt.getY()-birdysize
        birdybottom=birdypt.getY()+birdysize
        while birdyright>=pipeobj.getLeftX() and birdyleft<=pipeobj.getRightX():
            if birdytop<=pipeobj.getTopY() or birdybottom>=pipeobj.getBottomY():
                return True
            else:
                return False
  



def scoreTest():
    highScore(5)
    print ""
    highScore(8)

def birdTest(w):
    bb=Bird(w, 20)
    while bb.onScreen()==True:
        key=w.checkKey()
        if key != None:
            if key=='q':
                return
            if key == 'Up':
                mv=-5
                for i in range(0,8):
                    bb.move(mv)
                    sleep(.01)
        bb.move(2)
        sleep(.01) 

def pipeTest(w):
    pp=Pipe(w)
    while True:
        key=w.checkKey()
        if key != None:
            if key == 'q':
                return
        if pp.offScreen()==False:
            pp.move(-1)
            sleep(.01)
        elif pp.offScreen()==True:
            pp=Pipe(w)
if __name__=='__main__':
    main()
