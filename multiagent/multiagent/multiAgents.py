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
        # setting up some basic variables for myself
        numAgents = gameState.getNumAgents() - 1
        maxDepth = self.depth
        currentDepth = 0

        # I need to define a method here that will allow me to make recursive calls to it in order to implement the algorithm
        def minimaxWithPruning(gameState, depth, agentIndex, alpha, beta):
            if(gameState.isWin() or gameState.isLose() or depth == maxDepth):
                return self.evaluationFunction(gameState)

            if(agentIndex == 0):
                # maximizing agent pseudocode
                # initialize V to neg inf
                # for each successor of state:
                #   v = max (v, value(successor, alpha, beta))
                #   if v > beta return v
                #   alpha = max(alpha, v)
                # return v

                # initialize V to neg inf
                bestV = -9999999999

                # for each successor of state:
                for action in gameState.getLegalActions(agentIndex):

                    # getting successor to make the minimaxWithPruning call easier to read
                    successor = gameState.generateSuccessor(
                        agentIndex, action)

                    # v = max (v, value(successor, alpha, beta))
                    v = minimaxWithPruning(successor, depth, 1, alpha, beta)
                    bestV = max(bestV, v)

                    #   if v > beta return v
                    if bestV > beta:
                        return bestV

                    # alpha = max(alpha, v)
                    alpha = max(alpha, bestV)

                # return v
                return bestV
            else:
                # Minimizing agent pseudocode
                # 1. initialize V to neg inf
                # 2. for each successor of state:
                # 3.   v = min (v, value(successor, alpha, beta))
                # 4.   if v > beta return v
                # 5.   beta = min(alpha, v)
                # 6. return v

                # 1. initialize V to neg inf
                bestV = 9999999999

                # 2. for each successor of state:
                for action in gameState.getLegalActions(agentIndex):

                    # getting successor to make the minimaxWithPruning call easier to read
                    successor = gameState.generateSuccessor(
                        agentIndex, action)

                    # Checking to see if all adversarial agents have gone
                    if(agentIndex < numAgents):

                        # 3.   v = min (v, value(successor, alpha, beta))
                        v = minimaxWithPruning(
                            successor, depth, agentIndex + 1, alpha, beta)
                        bestV = min(bestV, v)

                        # 4. if v > beta return v
                        if bestV < alpha:
                            return bestV

                        # 5.   beta = min(alpha, v)
                        beta = min(beta, bestV)
                    else:
                        if(agentIndex == numAgents):

                            # setting new depth to make minimax with pruning call easier to read
                            newDepth = depth + 1

                            # 3.   v = min (v, value(successor, alpha, beta))
                            v = minimaxWithPruning(
                                successor, newDepth, 0, alpha, beta)
                            bestV = min(bestV, v)

                            # 4. if v > beta return v
                            if bestV < alpha:
                                return bestV

                            # 5.   beta = min(alpha, v)
                            beta = min(beta, bestV)

                # 6. return v
                return bestV

        # Now we need to apply the function to function to pacman :)
        # maximizing agent pseudocode
        # 1. initialize V to neg inf
        # 2. for each successor of state:
        # 3.   v = max (v, value(successor, alpha, beta))
        # 4.   if v > beta return v
        # 5.   alpha = max(alpha, v)
        # 6. return v

        # setting the actionToReturn variable to something random
        actionToReturn = Directions.NORTH

        # 1. initialize V to neg inf
        maxUtility = -99999999
        alpha = -999999999
        beta = 999999999

        # gettting the minimax with pruning values from each legal action for pacman(performing the maximize function on pacman)
        # 2. for each successor of state:
        for action in gameState.getLegalActions(0):
            # 3.   v = max (v, value(successor, alpha, beta))
            currentUtility = minimaxWithPruning(
                gameState.generateSuccessor(0, action), currentDepth, 1, alpha, beta)
            # checking to see if the current utility should be our max utility
            if (currentUtility > maxUtility):
                maxUtility = currentUtility
                actionToReturn = action
            # 4.   if v > beta return v
            if (maxUtility > beta):
                return maxUtility
            # 5.   alpha = max(alpha, v)
            alpha = max(maxUtility, alpha)

        # 6. return v
        return actionToReturn


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

        # setting up some basic variables for myself
        numAgents = gameState.getNumAgents() - 1
        maxDepth = self.depth
        currentDepth = 0

        # I need to define a method here that will allow me to make recursive calls to it in order to implement the algorithm
        def expectimax(gameState, depth, agentIndex):
            if(gameState.isWin() or gameState.isLose() or depth == maxDepth):
                return self.evaluationFunction(gameState)

            if(agentIndex == 0):
                # Maximizing agent pseudocode from class
                # 1. init v to -infinity
                # 2. for each successor of state:
                # 3.    v = max(v, value(successor))
                # 4. return v

                # 1. init v to -infinity
                bestV = -9999999999

                # 2. for each successor of state:
                for action in gameState.getLegalActions(agentIndex):
                    # 3.    v = max(v, value(successor))
                    v = expectimax(gameState.generateSuccessor(
                        agentIndex, action), depth, 1)
                    bestV = max(bestV, v)
                # 4. return v
                return bestV
            else:

                # Expectimax agent pseudocode from class
                # 1. init v to 0
                # 2. for each successor of state:
                # 3.    p = probability(successor)
                # 4.    v += p * value(successor)
                # 5. return v

                # 1. init v to 0
                bestV = 0

                # I do this step a little out of order but that is because all of the probs
                # are the same. If each leaf had a different prob then we would calculate that
                # in the for loop instead of out of it.
                # 3.    p = probability(successor)
                totalLegalActions = float(
                    len(gameState.getLegalActions(agentIndex)))
                probability = 1 / totalLegalActions

                # 2. for each successor of state:
                for action in gameState.getLegalActions(agentIndex):

                   # Checking to see if all adversarial agents have gone
                    if(agentIndex < numAgents):

                        # 4.    v += p * value(successor)
                        v = expectimax(gameState.generateSuccessor(
                            agentIndex, action), depth, agentIndex + 1)
                        bestV += probability * v
                    else:
                        if(agentIndex == numAgents):

                            # 4.    v += p * value(successor)
                            v = expectimax(gameState.generateSuccessor(
                                agentIndex, action), depth + 1, 0)
                            bestV += probability * v
                # 5. return v
                return bestV

        # Now we need to apply the function to function to pacman :)
        # Maximizing agent pseudocode from class
        # 1. init v to -infinity
        # 2. for each successor of state:
        # 3.    v = max(v, value(successor))
        # 4. return v

        # setting the actionToReturn variable to something random
        actionToReturn = Directions.NORTH

        # 1. init v to -infinity
        maxUtility = -99999999

        # 2. for each successor of state:
        for action in gameState.getLegalActions(0):
            currentUtility = expectimax(
                gameState.generateSuccessor(0, action), currentDepth, 1)
            # 3.    v = max(v, value(successor))
            if (currentUtility > maxUtility):
                maxUtility = currentUtility
                actionToReturn = action
        # 4. return v
        return actionToReturn


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    I used our above implementation of the eval function for reflex agent to help me make this one.
    Our evaluation function will take a few things into consideration. They are as follows:

    - Distance to closest piece of food
    - Distance to closest active ghost
    - Distance to closest scared ghost
    - number of food pellets left
    - number of big food pieces left
    - Current game score

    We will start with the current game score and add all of the positive things in the game 
        (i.e., being closer to scared ghosts, being close to food pellets, having less food pellets, having less capsules(Our thinking is that the more capsules you eat the further along in the game you are)) 
    and subtract the negative things like being closer to active ghosts.

    To break the function down into parts:
    currentGameState.getScore() +                       # This takes the current score and adds it as our base score to return

    (1/float(disToNearestFood) + 1) +                   # We take the reciprocal of the distance to the nearest food but we add 1 to make sure we don't get a divide by 0 error. This makes the effect that the closer we are to food, the higher the score

    1/(float(foodLeft) * 10) -                          # We take the reciprocal fo the food left because we don't want it to affect the score that much but also it is important to add it in there. Also this way the score goes up as the number of pellets goes down. 
                                                        # the multiplication by 10 helps create a more dramatic change in scoring

    (1/(float(distanceToNearestActiveGhost) + 1)) +     # The closer we are to an active ghost the more our score goes down. We also add 1 here to make sure we do not get a divide by 0 error.

    (1/(float(distanceToNearestScaredGhost) + 1)) +     # The closer we are to a scared ghost the more our score goes up. We also add 1 here to make sure we do not get a divide by 0 error.

    1/(float(capsulesLeft) + 1)                         # Our logic here is that the less capsules there are the closer you are too winning and it will direct pacman towards the ghosts that are scared. Again, we also add 1 here to make sure we do not get a divide by 0 error.




    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    allFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    activeGhosts = []
    scaredGhosts = []
    foodList = allFood.asList()

    # getting the attributes for my eval function
    capsulesLeft = len(currentGameState.getCapsules())
    foodLeft = len(foodList)
    disToNearestFood = -5
    distanceToNearestActiveGhost = -5
    distanceToNearestScaredGhost = -5

    # separating the ghosts into two categories. If they have a scared timer then they are scared and if they don't then they are active
    for ghostState in currentGhostStates:
        if ghostState.scaredTimer > 0:
            scaredGhosts.append(ghostState.getPosition())

        else:
            activeGhosts.append(ghostState.getPosition())

    """Calculating the distance to the nearest piece of food"""
    for food in foodList:
        if(disToNearestFood == -5):
            disToNearestFood = manhattanDistance(pos, food)
        else:
            currentFoodDistance = manhattanDistance(pos, food)
            if(currentFoodDistance < disToNearestFood):
                disToNearestFood = currentFoodDistance

    """Calculating the distance to the nearest active ghost"""
    for ghost_position in activeGhosts:
        if(ghost_position):
            if(distanceToNearestActiveGhost == -5):
                distanceToNearestActiveGhost == manhattanDistance(
                    pos, ghost_position)
            else:
                currentGhostDistance = manhattanDistance(
                    pos, ghost_position)
                if(currentGhostDistance < distanceToNearestActiveGhost):
                    distanceToNearestActiveGhost = currentGhostDistance

    """Calculating the distance to the nearest scared ghost"""
    for ghost_position in scaredGhosts:
        if(ghost_position):
            if(distanceToNearestScaredGhost == -5):
                distanceToNearestScaredGhost == manhattanDistance(
                    pos, ghost_position)
            else:
                currentGhostDistance = manhattanDistance(
                    pos, ghost_position)
                if(currentGhostDistance < distanceToNearestScaredGhost):
                    distanceToNearestScaredGhost = currentGhostDistance

    # Before we calculate the evalResult we need to see if we are in a winning state or a losing state. If we are in a winning we return a number close to pos infinity
    # If we are in a losing state we return a number close to neg infinity
    if currentGameState.isWin():
        return 99999999
    elif currentGameState.isLose():
        return -99999999

    # We know that there will be some food left because we are not in a win state if we make it down here
    # Because of that, we don't need to be worried about a divide by 0 error witht the food left.
    evalResult = currentGameState.getScore() + (1/float(disToNearestFood) + 1) + 1/(float(foodLeft) * 10) - \
        (1/(float(distanceToNearestActiveGhost) + 1)) + \
        (1/(float(distanceToNearestScaredGhost) + 1)) + 1/(float(capsulesLeft) + 1)

    return evalResult


# Abbreviation
better = betterEvaluationFunction
