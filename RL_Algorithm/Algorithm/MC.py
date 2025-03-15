from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Monte Carlo algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.MONTE_CARLO,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(self):
        """
        Update Q-values using Monte Carlo.

        This method applies the Monte Carlo update rule to improve policy decisions by updating the Q-table.
        """
        G = 0  # Initialize return
        visited = set()  # Track visited state-action pairs

        # Process episode in reverse order (to apply Monte Carlo return formula)
        for t in reversed(range(len(self.obs_hist))):
            obs_dis = self.obs_hist[t]
            action_idx = self.action_hist[t]
            reward = self.reward_hist[t]

            # Compute discounted return
            G = self.discount_factor * G + reward

            # Update Q-value only for first visit to (state, action)
            if (obs_dis, action_idx) not in visited:
                visited.add((obs_dis, action_idx))

                # Increment visit count
                self.n_values[obs_dis][action_idx] += 1

                # Average returns for state-action pair
                alpha = 1.0 / self.n_values[obs_dis][action_idx]
                self.q_values[obs_dis][action_idx] += alpha * (G - self.q_values[obs_dis][action_idx])

        # Clear episode history after update
        self.obs_hist.clear()
        self.action_hist.clear()
        self.reward_hist.clear()