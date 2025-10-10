import pygame
from tensorflow import GradientTape, clip_by_value, constant, convert_to_tensor, reduce_sum, square, float32
from tensorflow.keras import Input, Model
from tensorflow.math import log
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Literal, Union
import numpy as np

WILDCARD = Literal["*"]

METRIC_NAMES = Literal[
    "Episodes",
    "Win Rate",
    "Highest Score",
    "Time Efficiency",
    "Left Click Avg",
    "Right Click Avg",
    "Total Click Avg",
    "Game Conclusion History"
]

class A2C:
    """
    Advantage Actor-Critic (A2C) Agent for Minesweeper.
    """
    
    def __init__(self, gridSize, maxLives, gameInstance, discountFactor = 0.99, lr: float | Dict[int, float] = 0.001, betaEntropy = 0.01, maxGradNorm = 0.5, advantageClip = 10.0):
        """
        Initializes the Actor-Critic (A2C) agent with the given parameters.

        It's `lr` parameter can be either a `float` to specify a fixed learning rate, or
        a `dict[int, float]` to specify learning rate decay at certain episode milestones.
        If the `lr` is specified as a `dict`, it will always check for the `0` key first,
        and if the episode 0 is non-existent, it will use the default value of `0.001`.

        :param gridSize: Size of the game grid (gridSize x gridSize).
        :type gridSize: int

        :param maxLives: Maximum number of lives the agent can have.
        :type maxLives: int

        :param gameInstance: An instance of the game environment.
        :type gameInstance: MinesweeperGame

        :param discountFactor: Discount factor for future rewards (default is 0.99).
        :type discountFactor: float

        :param lr: Learning rate for the optimizer (default is 0.001). Can be a dictionary to perform learning rate decay.
        :type lr: float | dict[int, float]

        :param betaEntropy: Coefficient for entropy regularization (default is 0.01).
        :type betaEntropy: float

        :param maxGradNorm: Maximum gradient norm for clipping (default is 0.5).
        :type maxGradNorm: float

        :param advantageClip: Clipping value for advantage to stabilize training (default is 10.0).
        :type advantageClip: float
        """
        # Game Related Parameters
        self.GAME = gameInstance
        self.MAX_LIVES: int = maxLives
        self.GRID_SIZE: int = gridSize
        self.ACTION_SPACE: int = gridSize * gridSize * 2
        self.STATE_SHAPE: tuple = (gridSize, gridSize, 6)  # Added channel for interaction history
        self.INPUT_SIZE: int = np.prod(self.STATE_SHAPE)

        # Hyperparameters
        self.DISCOUNT_FACTOR = discountFactor
        self.LR = lr
        self.BETA_ENTROPY = betaEntropy  # Increased from 0.001 to 0.01
        self.MAX_GRAD_NORM = maxGradNorm
        self.ADVANTAGE_CLIP = advantageClip

        # Metrics to track
        self.EPISODES: int = 1               				# Always start at episode 1
        self.ELAPSED_TIME: list = []          				# Time taken for each episode
        self.GAME_CONCLUSIONS: list = []      				# Conclusions from each game
        self.WIN_RATE: float = 0.0             				# Win rate over all episodes
        self.HIGHEST_SCORE: int = 0          				# Highest score achieved
        self.SCORES: list = []                				# Scores from each episode
        self.TIME_EFFICIENCY: float = 0.0      				# Average time efficiency
        self.CLICKS: dict = {
            "left": [],                 					# Total left clicks per episode
            "right": [],                					# Total right clicks per episode
            "total": []                 					# Overall total clicks per episode
        }
        self.ACTION_HISTORY: list = []        				# History of actions taken
        self.ACTION_HISTORY_SIZE: int = 20    				# Increased to track more actions

        # Anti Exploitation Measures - Enhanced
        self.MAX_ACTION_REPEATED: int = 3    				# Reduced from 5 to be more strict
        self.LAST_ACTIONS: dict = {}          				# History of last actions
        self.LAST_ACTION: tuple = None         				# Last action taken
        self.LAST_STATE_CHANGE: float = 0.0    				# Steps since last state change
        self.TILE_INTERACTION_HISTORY: dict = {}  			# Track all interactions per tile
        self.REPETITIVE_PENALTY_MULTIPLIER: float = 1.0  	# Escalating penalty for repetition

        # Initialize episode-specific variables
        if isinstance(lr, float):
            self.currentLr = lr
        else:
            if 0 in lr:
                self.currentLr = lr[0]
            else:
                self.currentLr = 0.001

        # Model and Training Components
        self.model = self.buildModel()
        self.optimizer = self.buildOptimizer()

        return

    def buildModel(self):
        """Builds the Actor-Critic neural network model."""
        # Input layer
        inputLayer = Input(shape = (self.INPUT_SIZE,), name = "State_Input")

        # Shared hidden layers - Increased complexity
        sharedHiddenLayer = Dense(256, activation = "relu")(inputLayer)
        sharedHiddenLayer = Dense(256, activation = "relu")(sharedHiddenLayer)
        sharedHiddenLayer = Dense(128, activation = "relu")(sharedHiddenLayer)
        sharedHiddenLayer = Dense(64, activation = "relu")(sharedHiddenLayer)

        # Actor output layer
        actorOutput = Dense(
            self.ACTION_SPACE,
            activation = "softmax",
            name = "Actor"
        )(sharedHiddenLayer)

        # Critic output layer
        criticOutput = Dense(
            1,
            activation = "linear",
            name = "Critic"
        )(sharedHiddenLayer)

        # Combined output model
        model = Model(inputs = inputLayer, outputs = [actorOutput, criticOutput])
        return model

    def buildOptimizer(self):
        """Builds the optimizer for training the model."""
        return Adam(learning_rate = self.currentLr, clipnorm = self.MAX_GRAD_NORM)

    def _encodeState(self):
        """
        Encodes the game state into a suitable format for the model.

        :returns: Encoded state as a tensor.
        :rtype: tf.Tensor
        """
        stateArr = np.zeros(self.STATE_SHAPE, dtype = np.float32)

        # Channel 0: Encode Board State
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                tile = self.GAME.grid[x][y]

                if tile.is_flagged:
                    # 9 for flagged
                    stateArr[x, y, 0] = 9.0
                elif tile.is_revealed:
                    # 0-8 for revealed; 0 for empty, 1-8 for number of adjacent mines
                    stateArr[x, y, 0] = float(tile.adjacent_mines)
                else:
                    # -1 for unrevealed
                    stateArr[x, y, 0] = -1.0

        # Channel 1: Global Lives
        normalizedLives = self.GAME.lives / self.MAX_LIVES
        stateArr[:, :, 1] = normalizedLives

        # Channel 2: Global Combo
        normalizedCombo = self.GAME.combo / 50.0
        stateArr[:, :, 2] = normalizedCombo

        # Channel 3: Flags available
        stateArr[:, :, 3] = (self.GAME.flag_limit - self.GAME.flags_placed) / self.GAME.flag_limit

        # Channel 4: Progress
        tilesRevealed = sum(1 for x in range(self.GRID_SIZE)
            for y in range(self.GRID_SIZE)
                if self.GAME.grid[x][y].is_revealed and not self.GAME.grid[x][y].is_mine
        )
        stateArr[:, :, 4] = tilesRevealed / (self.GRID_SIZE * self.GRID_SIZE - self.GAME.total_mines)

        # Channel 5: Interaction History - NEW
        # Encodes how recently each tile was interacted with
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                key = (x, y)
                if key in self.TILE_INTERACTION_HISTORY:
                    # Decay value based on how many steps ago the interaction was
                    stepsAgo = len(self.ACTION_HISTORY) - self.TILE_INTERACTION_HISTORY[key]
                    stateArr[x, y, 5] = max(0, 1.0 - (stepsAgo / 10.0))
                else:
                    stateArr[x, y, 5] = 0.0

        # Flatten the state array for model input
        return convert_to_tensor(stateArr.flatten().reshape(1, self.INPUT_SIZE))

    def getValidActionMask(self):
        """
        Creates a mask for valid actions based on the current game state.

        :returns: Boolean mask where `True` = valid action and `False` = invalid action.
        :rtype: np.ndarray
        """
        mask = np.zeros(self.ACTION_SPACE, dtype=bool)

        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                tile = self.GAME.grid[x][y]
                idx = x * self.GRID_SIZE + y

                # Reveal action is valid if tile is not yet revealed and not flagged
                if not tile.is_revealed and not tile.is_flagged:
                    mask[idx] = True

                # Flag action handling
                flagIdx = idx + self.GRID_SIZE * self.GRID_SIZE
                if not tile.is_revealed:
                    # Can flag if not already flagged and have flags remaining
                    if not tile.is_flagged and self.GAME.flags_placed < self.GAME.flag_limit:
                        mask[flagIdx] = True
                    # Can unflag if already flagged
                    elif tile.is_flagged:
                        mask[flagIdx] = True

        # Handle stuck state where no actions are valid
        if not np.any(mask):
            # This typically happens if all remaining unrevealed tiles are flagged.
            # To get unstuck, the only valid move is to un-flag any flagged tile.
            for x in range(self.GRID_SIZE):
                for y in range(self.GRID_SIZE):
                    if self.GAME.grid[x][y].is_flagged:
                        flagIdx = (x * self.GRID_SIZE + y) + self.GRID_SIZE * self.GRID_SIZE
                        mask[flagIdx] = True

        return mask

    def chooseAction(self, state):
        """
        Chooses an action index based on the Actor's policy.

        :param state: The current state of the game, encoded as a tensor.
        :type state: tf.Tensor

        :returns: (x, y, actionType, actionIdx, actionProbs)
        - x, y: Coordinates on the grid
        - actionType: 0 for Reveal, 1 for Flag
        - actionIdx: The index of the chosen action
        - actionProbs: The probabilities of all actions
        """
        # Predict both policy and value from the model
        actionProbs, _ = self.model(state)
        actionProbs = actionProbs.numpy()[0]

        # 💡 FIX: Clip probabilities to [0, 1] range. This prevents np.power()
        # on slightly negative numbers (due to float errors) from returning NaN.
        actionProbs = np.clip(actionProbs, 0.0, 1.0)

        # Temperature scaling for exploration (higher early in training)
        temperature = max(0.8, 1.5 - (self.EPISODES / 3000.0))
        actionProbs = np.power(actionProbs, 1.0 / temperature)

        # Masking for the prediction
        validMask = self.getValidActionMask()
        actionProbs = actionProbs * validMask

        probSum = np.sum(actionProbs)
        validMaskSum = np.sum(validMask)
        
        if probSum > 0:
            actionProbs = actionProbs / probSum
        else:
            # Fallback to uniform distribution over valid actions
            # 💡 SECONDARY FIX: Check if there are ANY valid actions to prevent division by zero.
            if validMaskSum > 0:
                actionProbs = validMask.astype(float) / validMaskSum
            else:
                # Emergency Fallback: If no valid actions remain, select action 0 (Reveal 0, 0)
                # and create a uniform distribution over ALL actions to avoid the ValueError.
                # This should ideally not happen if getValidActionMask() is robust.
                actionProbs = np.ones(self.ACTION_SPACE, dtype=float) / self.ACTION_SPACE
                
        # Add small epsilon for pure exploration (decreases over time)
        epsilon = max(0.01, 0.1 - (self.EPISODES / 2000.0))
        if np.random.random() < epsilon:
            # Pure random action from valid actions
            validIndices = np.where(validMask)[0]
            if len(validIndices) > 0:
                actionIdx = np.random.choice(validIndices)
            else:
                # If there are truly no valid actions, select the first action
                actionIdx = 0  
        else:
            # Sample from the probability distribution
            # This is where the ValueError occurs if actionProbs contains NaN
            actionIdx = np.random.choice(self.ACTION_SPACE, p = actionProbs)

        idx = actionIdx % (self.GRID_SIZE * self.GRID_SIZE)
        x = idx // self.GRID_SIZE
        y = idx % self.GRID_SIZE

        # 0: Reveal; 1: Flag
        actionType = 1 if actionIdx >= (self.GRID_SIZE * self.GRID_SIZE) else 0

        return x, y, actionType, actionIdx, actionProbs

    def train(self, state, actionIdx, reward, nextState, isDone):
        """Trains the model using the given transition."""        
        reward = constant(reward, dtype = float32)

        with GradientTape() as tape:
            
            # Get Predictions from the current state
            actionProbsAll, valueCurrentTensor = self.model(state)
            valueCurrent = valueCurrentTensor[0, 0]

            # Get the value of the next state
            if isDone:
                valueNext = constant(0.0, dtype = float32)
            else:
                _, valueNextTensor = self.model(nextState)
                valueNext = valueNextTensor[0, 0]
            
            # Calculate the TD Target
            valueTarget = reward + self.DISCOUNT_FACTOR * valueNext

            # Calculate the Advantage (Surprise)
            advantage = valueTarget - valueCurrent
            advantage = clip_by_value(advantage, -self.ADVANTAGE_CLIP, self.ADVANTAGE_CLIP)
            
            # --- CRITIC LOSS (Mean Squared Error) ---
            critic_loss = square(valueTarget - valueCurrent)

            # --- ACTOR LOSS (Policy Gradient) ---
            # Get the probability of the action that was actually taken
            actionProb = actionProbsAll[0, actionIdx]
            actionProb = clip_by_value(actionProb, 1e-10, 1.0)
            logProb = log(actionProb)
            
            # The negative sign is to minimize loss to maximize return
            actorLossPolicy = -advantage * logProb
            
            # Entropy Bonus Loss (encourage exploration)
            actionProbsClipped = clip_by_value(actionProbsAll, 1e-10, 1.0)
            entropy = -reduce_sum(actionProbsClipped * log(actionProbsClipped), axis = -1)
            actorLossEntropy = - self.BETA_ENTROPY * entropy
            
            # Total Actor Loss
            actorLoss = actorLossPolicy + actorLossEntropy

            # --- TOTAL LOSS ---
            # We minimize both losses simultaneously
            totalLoss = actorLoss + 0.5 * critic_loss

        # Apply Gradients
        grads = tape.gradient(totalLoss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return totalLoss.numpy(), advantage.numpy()

    def endEpisode(self, newGameInstance):
        """
        Reset episode-specific tracking variables, update tracked metrics,
        and apply learning rate decay if applicable.

        :param newGameInstance: A new instance of the game environment for the next episode.
        :type newGameInstance: MinesweeperGame
        """
        elapsedTime = ((pygame.time.get_ticks() - self.GAME.start_time) / 1000)
        
        # -----------------------
        # Update tracked metrics
        # -----------------------
        self.GAME_CONCLUSIONS.append(1 if self.GAME.game_won else 0)
        self.SCORES.append(self.GAME.score)
        self.EPISODES += 1
        self.ELAPSED_TIME.append(elapsedTime)
        self.WIN_RATE = sum(self.GAME_CONCLUSIONS) / len(self.GAME_CONCLUSIONS)
        self.HIGHEST_SCORE = max(self.HIGHEST_SCORE, self.GAME.score)
        self.TIME_EFFICIENCY = sum(self.CLICKS["total"]) / sum(self.ELAPSED_TIME) if sum(self.ELAPSED_TIME) > 0 else 0

        # ---------------------------------
        # Update episode-specific trackers
        # ---------------------------------
        self.LAST_STATE_CHANGE = 0
        self.LAST_ACTION = None
        self.LAST_ACTIONS = {}
        self.TILE_INTERACTION_HISTORY = {}
        self.ACTION_HISTORY = []
        self.REPETITIVE_PENALTY_MULTIPLIER = 1.0

        # ----------------------
        # Update LR if possible
        # ----------------------
        if isinstance(self.LR, dict):
            new_lr = self.optimizer.learning_rate.numpy() # Start with current LR
            
            # Find the latest episode milestone that has been passed
            for episode_milestone in sorted(self.LR.keys()):
                if self.EPISODES >= episode_milestone:
                    new_lr = self.LR[episode_milestone]
            
            # If the learning rate needs to be changed, update the optimizer
            if new_lr != self.optimizer.learning_rate.numpy():
                print(f"\n[LR Scheduler] Episode {self.EPISODES}: Learning rate changed to {new_lr:.6f}\n")
                self.optimizer.learning_rate.assign(new_lr)

        # ---------------------
        # Update game instance
        # ---------------------
        self.GAME = newGameInstance

        return

    def printMetrics(self, metrics: List[Union[METRIC_NAMES, WILDCARD]] = ["*"], asString: bool = False) -> str | None:
        """
        Prints out the selected metrics.
        Use "*" to print all metrics.

        Metrics Available:
        - Episodes
        - Win Rate
        - Highest Score
        - Time Efficiency
        - Left Click Avg
        - Right Click Avg
        - Total Click Avg
        - Game Conclusion History

        :param metrics: List of metrics to print or ["*"] for all.
        :type metrics: list

        :param asString: If True, returns the metrics as a formatted string instead of printing.
        :type asString: bool

        :returns: Formatted metrics string if `asString` is True, otherwise None.
        :rtype: str | None
        """
        if "*" in metrics or not metrics:
            metrics = ["Episodes", "Win Rate", "Highest Score", "Time Efficiency", "Left Click Avg", "Right Click Avg", "Total Click Avg", "Game Conclusion History"]
        
        metric_strings = []
        for metric in metrics:
            if metric == "Episodes":
                metric_strings.append(f"Episodes: {self.EPISODES - 1}")
            elif metric == "Win Rate":
                metric_strings.append(f"Win Rate: {self.WIN_RATE:.2%}")
            elif metric == "Highest Score":
                metric_strings.append(f"Highest Score: {self.HIGHEST_SCORE:.2f}")
            elif metric == "Time Efficiency":
                metric_strings.append(f"Time Efficiency: {self.TIME_EFFICIENCY:.2f} steps/sec")
            elif metric == "Left Click Avg":
                lca = np.mean(self.CLICKS['left']) if self.CLICKS['left'] else 0
                lcat = np.sum(self.CLICKS['left']) / np.sum(self.ELAPSED_TIME) if np.sum(self.ELAPSED_TIME) > 0 else 0
                metric_strings.append(f"Left Click Avg: {lca:.2f} c/g ({lcat:.2f} c/s)")
            elif metric == "Right Click Avg":
                rca = np.mean(self.CLICKS['right']) if self.CLICKS['right'] else 0
                rcat = np.sum(self.CLICKS['right']) / np.sum(self.ELAPSED_TIME) if np.sum(self.ELAPSED_TIME) > 0 else 0
                metric_strings.append(f"Right Click Avg: {rca:.2f} c/g ({rcat:.2f} c/s)")
            elif metric == "Total Click Avg":
                tca = np.mean(self.CLICKS['total']) if self.CLICKS['total'] else 0
                tcat = np.sum(self.CLICKS['total']) / np.sum(self.ELAPSED_TIME) if np.sum(self.ELAPSED_TIME) > 0 else 0
                metric_strings.append(f"Total Click Avg: {tca:.2f} c/g ({tcat:.2f} c/s)")
            elif metric == "Game Conclusion History":
                metric_strings.append(f"Game Conclusion History: {self.GAME_CONCLUSIONS.count(1)} Wins, {self.GAME_CONCLUSIONS.count(0)} Losses")

        if asString:
            return " | ".join(metric_strings)

        print(" | ".join(metric_strings))
        return