# **Cartpole**

## 1. Setting up Cart pole Agent 

for cartpole I implement the following 

- **Core Functions**

    - **get_discretize_action()**: Returns a discrete action based on the current policy.

    - **mapping_action()**: Converts a discrete action back into a continuous action within the defined action range.

    - **discretize_state()**: Discretizes and scales the state based on observation weights.

    - **decay_epsilon()**: Decreases epsilon over time and returns the updated value.

- **Algorithm**

    - **Monte Carlo class**

    - **SARSA class**

    - **Q-Learning Class**

    - **Double Q-Learning Class**

including 

- A constructor which initializes the same variables as the class it inherits from.

- Superclass Initialization (super().__init__()).

- An update() function that updates the agent’s learnable parameters and advances the training step.

- An update function in train loop 

In reward term I add a little bit more term as follow 

- alive -> Constant running reward , weight = 1.0

- terminating -> Failure penalty , weight = -2.0 

- pole_pos_p -> pole target penalty , weight = -7.0 

- pole_pos -> pole target reward , weight = 1.5

the pole_pos_p and a pole_pos reward term is use as reward and penalty for taget 0.0 ,pole_pos_p term will penalize the agent if the target is not at the target ,and pole_pos term will give increase reward as a exponential if the pole is close to the target (both term is not equal)

## 2. training the Agent 

for training from a lot of traing with Stabilize Cart-Pole Task we can conclude the normal hyperparameter for every algorithm as follow 

- num_of_action = 5
- action_range = [-12.0, 12.0]
- discretize_state_weight = [10,20, 2, 2]

these 3  parameter is come mapping the num_of_action to action_range I will have 5 action contian -12,-6,0,6,12 action that agent can choose from and discretize_state_weight will map the obseve state to weight such as cart posiion that I weight it 10 it mean the cart position are going to have 10 state (from the limit in the code that cart are limit at -3 to 3) the cart pose will the divide in 10 state -3 to 3 in linspace

- learning_rate = 0.3
- n_episodes = 12000
- start_epsilon = 1.0
- epsilon_decay = 0.9996
- final_epsilon = 0.01
- discount = 0.5

### Epsilon calculation 

To calculate the epsilon_decay rate i use the equation

$$
\text{desired epsilon at each epidose} = \text{initial epsilon} \times (\text{epsilon decay rate})^x
$$

such as I want to see that what episode will my epsilon be 0.5 

$$
0.5 = 1.0 \times (0.9997)^x
$$

the answer is around 2310 episode the epsilone value will be at 0.5 and it will converge to min at 0.01 at around 15000 
episode ,The epsilon will be like the picture as follow 

![image](https://github.com/user-attachments/assets/7ad2a431-a3b9-4877-b814-175f04c16f06)



## 3. Evaluate Agent Perfornance

we will evaluate each algorithm first then compare it to each other and to understand each algorithm behavior and to selected the best performance for each algorithm ,we will find tune each hyperparameter as follow 

1. epsilon decay -> to analyze explore and exploit behavior of each algorithm 

2. num action + discretize weight -> -> analyze the effec of resolutions of each algorithm 

3. leaning rate 

4. discount factor 


### 3.1 Q-learning 

#### Normal_q-learing 


![image](https://github.com/user-attachments/assets/2e41e9a0-854d-4c66-a90a-85841e7a4b1d)


( count is timestep (in secound) that agent is alive or not be terminate in average in every 1000 episdode )

and we can plot compare the state with Q-value as follow 

![Normal_q](https://github.com/user-attachments/assets/ff6123d4-58cd-44e6-a6ee-1cda50578e3e)


### 3.2 SARSA

### 3.3 Monte Carlo (MC)

### 3.4 Double Q-learning 

### Conpare every algorithm

## Conclusion 

- 

Which algorithm performs best 
- 

Why does it perform better than the others?
- 

How do the resolutions of the action space and observation space affect the learning process? Why?
- 
