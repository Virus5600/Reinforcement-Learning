from tensorflow import GradientTape, convert_to_tensor, reduce_mean, square, float32, gather_nd
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Literal, Union
import numpy as np
import pygame
import random
from collections import deque

WILDCARD = Literal["*"]

METRIC_NAMES = Literal[
    "Episodes",
    "Current Episode",
    "Win Rate",
    "Highest Score",
    "Time Efficiency",
    "Left Click Avg",
    "Right Click Avg",
    "Total Click Avg",
    "Game Conclusion History"
]

class DQN:
    """
    Deep Q-Network (DQN) Agent for Minesweeper.
    """
    
    def __init__(self, gridSize, maxLives, gameInstance, discountFactor=0.99, lr: float | Dict[int, float] = 0.001, 
                 maxGradNorm=0.5, bufferSize=10000, batchSize=32, targetUpdateFreq=100, 
                 epsilonStart=1.0, epsilonEnd=0.01, epsilonDecay=0.995):
        """
        Initializes the DQN agent with the given parameters.

        :param gridSize: Size of the game grid (gridSize x gridSize).
        :param maxLives: Maximum number of lives the agent can have.
        :param gameInstance: An instance of the game environment.
        :param discountFactor: Discount factor for future rewards (default is 0.99).
        :param lr: Learning rate for the optimizer (default is 0.001).
        :param maxGradNorm: Maximum gradient norm for clipping (default is 0.5).
        :param bufferSize: Size of the replay buffer (default is 10000).
        :param batchSize: Batch size for training (default is 32).
        :param targetUpdateFreq: Frequency of target network updates (default is 100).
        :param epsilonStart: Starting epsilon for epsilon-greedy exploration (default is 1.0).
        :param epsilonEnd: Minimum epsilon value (default is 0.01).
        :param epsilonDecay: Epsilon decay rate (default is 0.995).
        """
        # Game Related Parameters
        self.GAME = gameInstance
        self.MAX_LIVES: int = maxLives
        self.GRID_SIZE: int = gridSize
        self.ACTION_SPACE: int = gridSize * gridSize * 2
        self.STATE_SHAPE: tuple = (gridSize, gridSize, 7)
        self.INPUT_SIZE: int = np.prod(self.STATE_SHAPE)

        # Hyperparameters
        self.DISCOUNT_FACTOR = discountFactor
        self.LR = lr
        self.MAX_GRAD_NORM = maxGradNorm
        self.BUFFER_SIZE = bufferSize
        self.BATCH_SIZE = batchSize
        self.TARGET_UPDATE_FREQ = targetUpdateFreq
        
        # Epsilon-greedy parameters
        self.EPSILON_START = epsilonStart
        self.EPSILON_END = epsilonEnd
        self.EPSILON_DECAY = epsilonDecay
        self.epsilon = epsilonStart

        # Metrics to track
        self.EPISODES: int = 1
        self.ELAPSED_TIME: list = []
        self.GAME_CONCLUSIONS: list = []
        self.WIN_RATE: float = 0.0
        self.HIGHEST_SCORE: int = 0
        self.SCORES: list = []
        self.TIME_EFFICIENCY: float = 0.0
        self.CLICKS: dict = {
            "left": [],
            "right": [],
            "total": []
        }
        self.ACTION_HISTORY: list = []
        self.ACTION_HISTORY_SIZE: int = 20

        # DQN specific components
        self.REPLAY_BUFFER = deque(maxlen=self.BUFFER_SIZE)
        self.TRAIN_STEP = 0

        # Anti Exploitation Measures
        self.LAST_ACTION: tuple = None
        self.TILE_INTERACTION_HISTORY: dict = {}

        # Initialize learning rate
        if isinstance(lr, float):
            self.currentLr = lr
        else:
            if 0 in lr:
                self.currentLr = lr[0]
            else:
                self.currentLr = 0.001

        # Build Q-Network and Target Network
        self.qNetwork = self.buildQNetwork()
        self.targetNetwork = self.buildQNetwork()
        self.updateTargetNetwork()
        
        self.optimizer = self.buildOptimizer()

    def buildQNetwork(self):
        """Builds the Q-Network neural network model."""
        inputLayer = Input(shape=(self.INPUT_SIZE,), name="State_Input")

        # Hidden layers
        hidden = Dense(256, activation="relu")(inputLayer)
        hidden = Dense(256, activation="relu")(hidden)
        hidden = Dense(128, activation="relu")(hidden)
        hidden = Dense(64, activation="relu")(hidden)

        # Output layer - Q-values for each action
        qValues = Dense(self.ACTION_SPACE, activation="linear", name="Q_Values")(hidden)

        model = Model(inputs=inputLayer, outputs=qValues)
        return model

    def buildOptimizer(self):
        """Builds the optimizer for training the model."""
        return Adam(learning_rate=self.currentLr, clipnorm=self.MAX_GRAD_NORM)

    def updateTargetNetwork(self):
        """Updates the target network weights to match the Q-network."""
        self.targetNetwork.set_weights(self.qNetwork.get_weights())

    def _encodeState(self):
        """
        Encodes the game state into a suitable format for the model.

        :returns: Encoded state as a tensor.
        """
        stateArr = np.zeros(self.STATE_SHAPE, dtype=np.float32)

        # Channel 0: Encode Board State
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                tile = self.GAME.grid[x][y]

                if tile.is_flagged:
                    stateArr[x, y, 0] = 9.0
                elif tile.is_revealed:
                    stateArr[x, y, 0] = float(tile.adjacent_mines)
                else:
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

        # Channel 5: Interaction History
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                key = (x, y)
                if key in self.TILE_INTERACTION_HISTORY:
                    stepsAgo = len(self.ACTION_HISTORY) - self.TILE_INTERACTION_HISTORY[key]
                    stateArr[x, y, 5] = max(0, 1.0 - (stepsAgo / 10.0))
                else:
                    stateArr[x, y, 5] = 0.0

        # Channel 6: Adjacent Number Information
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if not self.GAME.grid[x][y].is_revealed:
                    adjacentNumberSum = 0
                    adjacentRevealedCount = 0
                    
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0: continue

                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                                adjTile = self.GAME.grid[nx][ny]

                                if adjTile.is_revealed and not adjTile.is_mine:
                                    adjacentNumberSum += adjTile.adjacent_mines
                                    adjacentRevealedCount += 1
                    
                    if adjacentRevealedCount > 0:
                        stateArr[x, y, 6] = (adjacentNumberSum / adjacentRevealedCount) / 8.0

        return convert_to_tensor(stateArr.flatten().reshape(1, self.INPUT_SIZE))

    def getValidActionMask(self):
        """
        Creates a mask for valid actions based on the current game state.

        :returns: Boolean mask where True = valid action and False = invalid action.
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
                    if not tile.is_flagged and self.GAME.flags_placed < self.GAME.flag_limit:
                        mask[flagIdx] = True
                    elif tile.is_flagged:
                        mask[flagIdx] = True

        # Handle stuck state
        if not np.any(mask):
            for x in range(self.GRID_SIZE):
                for y in range(self.GRID_SIZE):
                    if self.GAME.grid[x][y].is_flagged:
                        flagIdx = (x * self.GRID_SIZE + y) + self.GRID_SIZE * self.GRID_SIZE
                        mask[flagIdx] = True

        return mask

    def chooseAction(self, state):
        """
        Chooses an action using epsilon-greedy policy.

        :param state: The current state of the game, encoded as a tensor.
        :returns: (x, y, actionType, actionIdx, qValues)
        """
        validMask = self.getValidActionMask()
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Random action from valid actions
            validIndices = np.where(validMask)[0]
            if len(validIndices) > 0:
                actionIdx = np.random.choice(validIndices)
            else:
                actionIdx = 0
        else:
            # Greedy action based on Q-values
            qValues = self.qNetwork(state).numpy()[0]
            
            # Mask invalid actions with very negative values
            qValues = np.where(validMask, qValues, -np.inf)
            actionIdx = np.argmax(qValues)

        idx = actionIdx % (self.GRID_SIZE * self.GRID_SIZE)
        x = idx // self.GRID_SIZE
        y = idx % self.GRID_SIZE
        actionType = 1 if actionIdx >= (self.GRID_SIZE * self.GRID_SIZE) else 0

        qValues = self.qNetwork(state).numpy()[0]
        return x, y, actionType, actionIdx, qValues

    def storeTransition(self, state, action, reward, nextState, done):
        """
        Stores a transition in the replay buffer.

        :param state: Current state
        :param action: Action taken
        :param reward: Reward received
        :param nextState: Next state
        :param done: Whether the episode ended
        """
        self.REPLAY_BUFFER.append((state, action, reward, nextState, done))

    def train(self):
        """
        Trains the Q-network using a batch from the replay buffer.
        
        :returns: Loss value if training occurred, None otherwise
        """
        if len(self.REPLAY_BUFFER) < self.BATCH_SIZE:
            return None

        # Sample a batch from replay buffer
        batch = random.sample(self.REPLAY_BUFFER, self.BATCH_SIZE)
        states, actions, rewards, nextStates, dones = zip(*batch)

        states = np.vstack([s.numpy() for s in states])
        nextStates = np.vstack([s.numpy() for s in nextStates])
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Calculate target Q-values using target network (outside tape)
        targetQValues = self.targetNetwork(nextStates).numpy()
        maxTargetQValues = np.max(targetQValues, axis=1)
        targets = rewards + (1 - dones) * self.DISCOUNT_FACTOR * maxTargetQValues
        targets = convert_to_tensor(targets, dtype=float32)

        with GradientTape() as tape:
            # Current Q-values for all actions
            qValues = self.qNetwork(states, training=True)
            
            # Extract Q-values for the actions that were taken
            # Create indices for batch and actions
            batch_indices = np.arange(self.BATCH_SIZE)
            action_indices = np.stack([batch_indices, actions], axis=1)
            
            # Gather Q-values for taken actions
            qValuesForActions = tf.gather_nd(qValues, action_indices)
            
            # Calculate loss (Mean Squared Error)
            loss = reduce_mean(square(targets - qValuesForActions))

        # Apply gradients
        grads = tape.gradient(loss, self.qNetwork.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.qNetwork.trainable_variables))

        # Update target network periodically
        self.TRAIN_STEP += 1
        if self.TRAIN_STEP % self.TARGET_UPDATE_FREQ == 0:
            self.updateTargetNetwork()

        return loss.numpy()

    def decayEpsilon(self):
        """Decays epsilon for epsilon-greedy exploration."""
        self.epsilon = max(self.EPSILON_END, self.epsilon * self.EPSILON_DECAY)

    def endEpisode(self, newGameInstance):
        """
        Reset episode-specific tracking variables and update metrics.

        :param newGameInstance: A new instance of the game environment for the next episode.
        """
        elapsedTime = ((pygame.time.get_ticks() - self.GAME.start_time) / 1000)
        
        # Update tracked metrics
        self.GAME_CONCLUSIONS.append(1 if self.GAME.game_won else 0)
        self.SCORES.append(self.GAME.score)
        self.EPISODES += 1
        self.ELAPSED_TIME.append(elapsedTime)
        self.WIN_RATE = sum(self.GAME_CONCLUSIONS) / len(self.GAME_CONCLUSIONS)
        self.HIGHEST_SCORE = max(self.HIGHEST_SCORE, self.GAME.score)
        self.TIME_EFFICIENCY = sum(self.CLICKS["total"]) / sum(self.ELAPSED_TIME) if sum(self.ELAPSED_TIME) > 0 else 0

        # Decay epsilon
        self.decayEpsilon()

        # Update episode-specific trackers
        self.LAST_ACTION = None
        self.TILE_INTERACTION_HISTORY = {}
        self.ACTION_HISTORY = []

        # Update LR if specified as dict
        if isinstance(self.LR, dict):
            new_lr = self.optimizer.learning_rate.numpy()
            
            for episode_milestone in sorted(self.LR.keys()):
                if self.EPISODES >= episode_milestone:
                    new_lr = self.LR[episode_milestone]
            
            if new_lr != self.optimizer.learning_rate.numpy():
                print(f"\n[LR Scheduler] Episode {self.EPISODES}: Learning rate changed to {new_lr:.6f}\n")
                self.optimizer.learning_rate.assign(new_lr)

        # Update game instance
        self.GAME = newGameInstance

    def printMetrics(self, metrics: List[Union[METRIC_NAMES, WILDCARD]] = ["*"], asString: bool = False) -> str | None:
        """
        Prints out the selected metrics.

        :param metrics: List of metrics to print or ["*"] for all.
        :param asString: If True, returns the metrics as a formatted string.
        :returns: Formatted metrics string if asString is True, otherwise None.
        """
        if "*" in metrics or not metrics:
            metrics = ["Episodes", "Win Rate", "Highest Score", "Time Efficiency", 
                      "Left Click Avg", "Right Click Avg", "Total Click Avg", "Game Conclusion History"]
        
        metric_strings = []
        for metric in metrics:
            if metric == "Episodes":
                metric_strings.append(f"Episodes: {self.EPISODES - 1}")
            elif metric == "Current Episode":
                metric_strings.append(f"Current Episode: {self.EPISODES} | Epsilon: {self.epsilon:.4f}")
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
        return None