# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        """Calculating the distance to the nearest piece of food"""
        foodList = newFood.asList()
        disToNearestFood = -5
        for food in foodList:
            if(disToNearestFood == -5):
                disToNearestFood = manhattanDistance(newPos, food)
            else:
                currentFoodDistance = manhattanDistance(newPos, food)
                if(currentFoodDistance < disToNearestFood):
                    disToNearestFood = currentFoodDistance

        """Calculating the distance to the nearest ghost"""
        distanceToNearestGhost = -5
        for ghost_position in successorGameState.getGhostPositions():
            if(distanceToNearestGhost == -5):
                distanceToNearestGhost == manhattanDistance(
                    newPos, ghost_position)
            else:
                currentGhostDistance = manhattanDistance(
                    newPos, ghost_position)
                if(currentGhostDistance < distanceToNearestGhost):
                    distanceToNearestGhost = currentGhostDistance

        """My logic behind this is that a state that is closer to food has better utility so the distance is added
           and if a ghost is close then that is subtracted. """
        return successorGameState.getScore() + (1/float(disToNearestFood)) - (1/float(distanceToNearestGhost))


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # setting up some basic variables for myself
        numAgents = gameState.getNumAgents() - 1
        maxDepth = self.depth
        currentDepth = 0

        # I need to define a method here that will allow me to make recursive calls to it in order to implement the algorithm
        def minimax(gameState, depth, agentIndex):
            if(gameState.isWin() or gameState.isLose() or depth == maxDepth):
                return self.evaluationFunction(gameState)

            if(agentIndex == 0):
                # Setting best value to -infinity
                bestV = -9999999999

                # going through each action and calling minimax on it to get the best state
                for action in gameState.getLegalActions(agentIndex):
                    v = minimax(gameState.generateSuccessor(
                        agentIndex, action), depth, 1)
                    bestV = max(bestV, v)
                return bestV
            else:
                # setting best value as pos infinity
                bestV = 9999999999

                # going through each action and calling minimax on it to get the best state
                for action in gameState.getLegalActions(agentIndex):

                    # Checking to see if all adversarial agents have gone
                    if(agentIndex < numAgents):
                        v = minimax(gameState.generateSuccessor(
                            agentIndex, action), depth, agentIndex + 1)
                        bestV = min(bestV, v)
                    else:
                        if(agentIndex == numAgents):
                            v = minimax(gameState.generateSuccessor(
                                agentIndex, action), depth + 1, 0)
                            bestV = min(bestV, v)
                return bestV

        # Now we need to apply the function to function to pacman :)

        # setting the actionToReturn variable to something random
        actionToReturn = Directions.NORTH

        # setting max util to neg infinity
        maxUtility = -99999999

        # gettting the minimax values from each legal action for pacman
        for action in gameState.getLegalActions(0):
            currentUtility = minimax(
                gameState.generateSuccessor(0, action), currentDepth, 1)
            # checking to see if the current utility should be our max utility
            if (currentUtility > maxUtility):
                maxUtility = currentUtility
                actionToReturn = action

        return actionToReturn


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
