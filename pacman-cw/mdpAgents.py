# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util


class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    
    def __init__(self):
        print "Starting up MDPAgent!"
        name = "Pacman"
        
        # Set up parameters
        
        self.gamma = 0.25           # the discount factor
        self.iterations = 200       # the number iterations
        self.eps = 1e-20             # the converge threshold
        
        self.pelletReward = 0.5/self.gamma          # the reward of a pellet
        self.capsuleReward = 0.5/(self.gamma**2)    # the reward of a capsule
        self.ghostReward = -1/(self.gamma**4)       # the reward of a ghost
        
        self.stopReward = -1e16                     # a big negative stop reward to prevent pacman from stopping
        self.ghostStepBound = 1                     # the pacman flee from ghost when the distance to the closer ghost
                                                    # is in this bound
        self.ghostTimerBound = 3                    # the ghost will be ignored if the ediable time
                                                    # is not less than this bound

        
    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        print "Running registerInitialState for MDPAgent!"
        print "I'm at:"
        print api.whereAmI(state)
        
        # Initialize the size of the maze
        corners = api.corners(state)
        self.size  = tuple(x+1 for x in corners[-1])
        
        self.walls = set(api.walls(state))          # Initialize the set of walls
        # Initialize the list of directions
        self.allDirections = [Directions.STOP, Directions.EAST, Directions.WEST, Directions.NORTH, Directions.SOUTH]
        
    # This is what gets run in between multiple games
    def final(self, state):
        print "Looks like the game just ended!"
        
    # This gets the next position after making an action on a given position
    def getNextPos(self, pos, action):
        x,y = int(pos[0]),int(pos[1])
        if action == Directions.STOP:    return (x,y)
        if action == Directions.EAST:    return (x+1,y)
        if action == Directions.WEST:    return (x-1,y)
        if action == Directions.NORTH:   return (x,y+1)
        if action == Directions.SOUTH:   return (x,y-1)
        return None
    
    # This gets the list of legal actions on the given position
    def isLegal(self, pos, action):
        x,y = self.getNextPos(pos, action)
        return (x,y) not in self.walls
        
    # This gets the expecation of making an action on the given position
    def getActionExpectation(self, pos, action,legal):
        # When the action is stop, the expectation is purely the stop reward
        if action == Directions.STOP:   return self.stopReward
        
        # the directions perpendicular to the action
        left = Directions.LEFT[action]
        right = Directions.RIGHT[action]
        
        # the probablities of moving in each possible direction
        frontProb = api.directionProb
        leftProb = (1-frontProb)/2 if left in legal else 0
        rightProb = (1-frontProb)/2 if right in legal else 0
        stopProb = 1-frontProb-leftProb-rightProb
        probs = [frontProb, leftProb, rightProb, stopProb]
        
        # the positions after moving in each possible direction
        nextPoses = [self.getNextPos(pos,direction) for direction in [action, left, right, Directions.STOP]]
    
        # caculate the expectation by summing up the probaility times the corresponding utility
        expectation = 0
        for prob,nextPos in zip(probs,nextPoses):
            expectation += prob*self.U[nextPos]
        
        return expectation
        
    # This gets the policy (best action) on the given position
    def getPolicy(self, pos):
        # get the actions we can try from the given position
        legal = [direction for direction in self.allDirections if self.isLegal(pos, direction)]

        # find the best action and the corresponding expectation
        optimalAction, optimalExpectation = Directions.STOP, self.stopReward
        for action in legal:
            expectation = self.getActionExpectation(pos, action, legal)
            if optimalExpectation < expectation:
                optimalAction, optimalExpectation = action, expectation
        
        return optimalAction,optimalExpectation
    
    # This set up the reward value and the initial utility value for each state
    def getReward(self,state):
    
        n,m = self.size
        
        self.R = {(i,j):0 for i in range(n) for j in range(m)}
        self.U = {(i,j):0 for i in range(n) for j in range(m)}
        
        # count the reward of food and capsules
        for pellet in api.food(state):    self.R[pellet] += self.pelletReward
        for capsule in api.capsules(state):    self.R[capsule] += self.capsuleReward
        
        # the positions of ghosts and how long the time they are ediable
        # the ghosts whose ediable time is long enough will be ignored
        ghostsPosWithTimes = {(int(pos[0]),int(pos[1])):timer for pos,timer in api.ghostStatesWithTimes(state) if timer < self.ghostTimerBound}
        
        # expand the positions of ghosts within the flee distance of the pacman
        for times in range(self.ghostStepBound):
            ghostsPosWithTimes = {self.getNextPos(pos,direction):timer for pos,timer in ghostsPosWithTimes.items() for direction in self.allDirections if self.isLegal(pos,direction)}
        # count the reward of ghosts and their adjacent grids
        for (x,y),timer in ghostsPosWithTimes.items():
            x,y = int(x), int(y)
            self.R[(x,y)] = self.U[(x,y)] = self.ghostReward*(self.ghostReward**timer)
        
        # label the states where the is a ghost as terminal
        self.terminalState = {pos for pos,timer in api.ghostStatesWithTimes(state) if timer < self.ghostTimerBound}

    # For now I just move randomly
    def getAction(self, state):
        # Get the actions we can try, and remove "STOP" if that is one of them.
        
        # set up the reward value and the initial utility value for each state
        self.getReward(state)
        
        n,m = self.size
        discount = self.gamma
        
        # the iterations of bellman equations
        for times in range(self.iterations):
            updated = False
            U_ = {}
            for i in range(n):
                for j in range(m):
                    # the utilities of terminal states will not change
                    if (i,j) in self.walls or  (i,j) in self.terminalState: U_[(i,j)] = self.U[(i,j)]
                    else:
                        action, expectation = self.getPolicy((i,j))
                        # bellman quation
                        U_[(i,j)] = self.R[(i,j)] + discount*expectation
                        
                        # check whether the utility converged
                        dU = abs(U_[(i,j)] - self.U[(i,j)])
                        if dU > self.eps: updated = True
            # the iterations can be terminated in advance, when  utilities converged
            if not updated: break
            self.U = U_
    
        # find the best action of pacman
        action,expectation = self.getPolicy(api.whereAmI(state))
        legal = api.legalActions(state)
#        print action
#        input()
        return api.makeMove(action, legal)
                        
