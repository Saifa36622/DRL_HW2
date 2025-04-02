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

- learning_rate = 0.3 <br>
we use 0.3 becuase it from sampling a lot of simulation of a lot of algorithm 0.3 is Reasonable steps	and most Balanced
  
- n_episodes = 12000
- start_epsilon = 1.0
- epsilon_decay = 0.9997
- final_epsilon = 0.01
- discount = 0.5 <br>
  The discount factory is max around 0.9 and min around 0.1 ,So we choose 0.5 to be the normal factor 

### Epsilon calculation 

To calculate the epsilon_decay rate i use the equation

$$
\text{desired epsilon at each epidose} = \text{initial epsilon} \times (\text{epsilon decay rate})^x
$$

such as I want to see that what episode will my epsilon be 0.5 

$$
0.5 = 1.0 \times (0.9997)^x
$$

the x will be around 2310 episode the epsilone value will be at 0.5 and it will converge to min at 0.01 at around 15000 
episode ,The epsilon will be like the picture as follow 

![image](https://github.com/user-attachments/assets/7ad2a431-a3b9-4877-b814-175f04c16f06)



## 3. Evaluate Agent Perfornance

we will evaluate each algorithm first then compare it to each other and to understand each algorithm behavior and to selected the best performance for each algorithm ,we will find tune each hyperparameter as follow 

1. epsilon decay -> to analyze explore and exploit behavior of each algorithm
- compare between 
    - Normal epsilon decay rate : 0.9997
    - Increase epsilon decay rate : 0.9998
    - Decrease epsilon decay rate : 0.9996

2. num action + discretize weight -> analyze the effec of resolutions of each algorithm
   - compare between
        - Normal space :
           - num_of_action = 5
            - action_range = [-12.0, 12.0]
            - discretize_state_weight = [10,20, 2, 2]
        - more space :
           - num_of_action = 7
            - action_range = [-12.0, 12.0]
            - discretize_state_weight = [20,30, 5, 5]
        - less space :
           - num_of_action = 3
            - action_range = [-12.0, 12.0]
            - discretize_state_weight = [5,10, 2, 2]
4. leaning rate -> analyze the effec how much new information overrides old Q-values of each algorithm
   - Normal leaning rate : 0.3
    - Increase leaning rate : 0.7
    - Decrease leaning rate : 0.1

6. discount factor -> analyze how Q-values change over time with different 𝛾 settings of each algorithm
   - compare between 
    - Normal discount factor : 0.3
    - Increase discount factor : 0.99
    - Decrease discount factor : 0.01


### 3.1 Q-learning 


This is result of all training from q-learning algorithm 

p.s. some of the train is in complete due to earth quake but we can still see the trend 

#### Epislon compare 


![image](https://github.com/user-attachments/assets/4c936d0d-a4d8-4c2a-9928-7d53d6ddd7ed)

we plot the Q-value from the state cart pose and pole pose as follow 

![image](https://github.com/user-attachments/assets/bd0aafbf-7d7d-48a8-afd0-790ef445a06f)

![image](https://github.com/user-attachments/assets/44df1089-6432-4228-9197-47e56a0cca1b)

and we track the state visite in heatmap in every timestep 

![image](https://github.com/user-attachments/assets/5e7793ca-6ff1-4a05-82e1-01016653dc23)

![image](https://github.com/user-attachments/assets/20aac7f5-efae-47fd-82b0-f2b007abce76)

![image](https://github.com/user-attachments/assets/50bba180-cb2b-430d-8df5-89de543a02f3)

and if we play agent from the collected q-value it will look like this 

**Q_Normal**

https://github.com/user-attachments/assets/6a2740d4-66e6-4788-a8a5-242c0c1678d3

**Increase epsilon decay rate : 0.9998**


https://github.com/user-attachments/assets/a663f0f1-f0d9-4619-b799-755b820c9152

**Decrease epsilon decay rate : 0.9996**


https://github.com/user-attachments/assets/df61e28f-6d15-483d-95f8-b5103041358c

then we can pick one of the play and plot the observation of the agent movement as follow 

![image](https://github.com/user-attachments/assets/54ec5f75-1ad1-442c-9937-28d378bfe956)

**Conclusion** 


#### Action and observation space compare 

![image](https://github.com/user-attachments/assets/12a18bf0-0c75-4e77-a734-ba0ce5b4bf2c)

we plot the Q-value from the state cart pose and pole pose as follow 

![image](https://github.com/user-attachments/assets/7d23ad24-1574-4287-bb00-6ff479d29a67)

![image](https://github.com/user-attachments/assets/65e6a281-93bb-4c67-be2b-ad65d8f44815)

and we track the state visite in heatmap in every timestep 

![image](https://github.com/user-attachments/assets/5e7793ca-6ff1-4a05-82e1-01016653dc23)

![image](https://github.com/user-attachments/assets/e842e2d7-3024-45d9-931a-6f38a8da1feb)

![image](https://github.com/user-attachments/assets/f9cdec60-3d89-47d1-8a02-630597e1892e)

and if we play agent from the collected q-value it will look like this 


#### Learning rate compare 

![image](https://github.com/user-attachments/assets/d272d748-496a-41be-9353-2fc37a244a3f)

we plot the Q-value from the state cart pose and pole pose as follow 

![image](https://github.com/user-attachments/assets/3d87173e-b798-4bb4-88c8-24a62c36f6c1)

![image](https://github.com/user-attachments/assets/ad41cc29-012f-4512-be24-dff19462d3c4)

and we track the state visite in heatmap in every timestep 

![image](https://github.com/user-attachments/assets/aab08e26-bdb5-434e-ae33-0315ddd7748f)

![image](https://github.com/user-attachments/assets/c1a7fa5c-d9de-4131-bb1d-fca3634f58df)

![image](https://github.com/user-attachments/assets/a76f963f-20f9-4c9b-ab49-ff2bf3c58335)

and if we play agent from the collected q-value it will look like this 

**Increase Learning rate**


https://github.com/user-attachments/assets/1b89e5b8-bd02-49d6-be88-d65b9fc5ef37

**Decrease Learning rate**


https://github.com/user-attachments/assets/d1c3648a-48dc-43c0-ac46-619c04bfde71

then we can pick one of the play and plot the observation of the agent movement as follow 

**Conclusion**


#### Discount factor compare  

![image](https://github.com/user-attachments/assets/025735ad-2df2-45a4-aab2-7fd0c5d98e05)

we plot the Q-value from the state cart pose and pole pose as follow 

![image](https://github.com/user-attachments/assets/236a2dac-058c-4fae-b934-6dc286724485)

![image](https://github.com/user-attachments/assets/72f765df-dcad-4bf0-a8a7-4c6086cfbc4c)

and we track the state visite in heatmap in every timestep 

![image](https://github.com/user-attachments/assets/aab08e26-bdb5-434e-ae33-0315ddd7748f)

![image](https://github.com/user-attachments/assets/a15a176d-a407-4b3d-97a5-754d8841a04c)

![image](https://github.com/user-attachments/assets/001d7e64-ac9c-41dd-aa63-a7399740d2a9)

and if we play agent from the collected q-value it will look like this 

**Increase discount factor : 0.99**


https://github.com/user-attachments/assets/dfb5e020-1e75-4ed1-a844-891610e35218


**Decrease discount factor : 0.01**


https://github.com/user-attachments/assets/84908894-8abf-4f1f-8b91-17dc823c27c5


then we can pick one of the play and plot the observation of the agent movement as follow 


### 3.2 SARSA
This is result of all training from SARSA algorithm 

#### Epislon compare 

![image](https://github.com/user-attachments/assets/dc568a03-81a1-4930-aa6c-b27114717893)

we plot the Q-value from the state cart pose and pole pose as follow 

![image](https://github.com/user-attachments/assets/26ce2411-6f4f-4e9f-a0af-63aa5eb0d3d4)

![image](https://github.com/user-attachments/assets/c809a2b4-13c2-4844-8ef3-fb34d2d4a153)

and we track the state visite in heatmap in every timestep 

![image](https://github.com/user-attachments/assets/f6a80d6d-850c-475e-bad6-84a6a9503747)

![image](https://github.com/user-attachments/assets/5d919da4-dd4c-407a-87f9-81ab9969e87b)

![image](https://github.com/user-attachments/assets/e9120280-9f16-46d9-b5c9-c16debf99588)

and if we play agent from the collected q-value it will look like this 

**Normal SARSA**



https://github.com/user-attachments/assets/70103c73-d616-4ace-bd66-14842dc5ae52

**Increase epsilon**


https://github.com/user-attachments/assets/ab55d63b-5d6e-4031-b342-0f5945a8b91d

**Decrease epsilon**



https://github.com/user-attachments/assets/e563dda5-9a9a-439c-a745-f3e9c5e7c38b



#### Action and observation space compare 

![image](https://github.com/user-attachments/assets/0e154bc4-ee1d-49db-9c61-5df5cee887db)

we plot the Q-value from the state cart pose and pole pose as follow 

![image](https://github.com/user-attachments/assets/cd542153-a714-447f-804c-c81a18dd3e84)

![image](https://github.com/user-attachments/assets/26eade80-ccd4-4de9-bc83-decd99cb7ee9)

and we track the state visite in heatmap in every timestep 

![image](https://github.com/user-attachments/assets/44d81ad1-5322-4f4a-bfa4-596547b49d95)

![image](https://github.com/user-attachments/assets/e6a454f2-5458-43f1-b279-105a815de792)

![image](https://github.com/user-attachments/assets/d2872929-8fd4-4f6e-8b10-76515d05bdbd)

and if we play agent from the collected q-value it will look like this 

#### Learning rate compare 

![image](https://github.com/user-attachments/assets/7f48935e-b322-431c-bde8-48c6ce404103)

#### Discount factor compare  

![image](https://github.com/user-attachments/assets/6c892890-3296-4775-82c4-6cedfb11ec07)


### 3.3 Monte Carlo (MC)

This is result of all training from Monte Carlo (MC) algorithm 


#### Epislon compare 

#### Action and observation space compare 

#### Learning rate compare 

#### Discount factor compare  




### 3.4 Double Q-learning 

This is result of all training from Double Q-learning algorithm

#### Epislon compare 

![image](https://github.com/user-attachments/assets/fba94543-38e0-4af2-ba24-7d0a92bc8536)

we plot the Q-value from the state cart pose and pole pose as follow 

![image](https://github.com/user-attachments/assets/28b3b5ca-9257-497d-8d3a-c1c10523e20c)

![image](https://github.com/user-attachments/assets/fba4d393-5db8-4bdf-91d9-f59620b7cf8d)


and if we play agent from the collected q-value it will look like this 


**Normal_double_Q**


https://github.com/user-attachments/assets/0c52b820-5f5b-4ac3-8aa7-6fa84a28aff3

**Increase epsilon**


**Decrease epsilon**


https://github.com/user-attachments/assets/ef91779e-2749-4064-b5c1-5b66f4189a92


then we can pick one of the play and plot the observation of the agent movement as follow 

#### Action and observation space compare 

![image](https://github.com/user-attachments/assets/6dd339c5-33f0-4f75-b58f-9db95187a286)

#### Learning rate compare 

![image](https://github.com/user-attachments/assets/ba8de33b-9109-445d-9a81-261355efa8ff)


#### Discount factor compare  

![image](https://github.com/user-attachments/assets/a8064467-9db5-478a-9b37-98f42d60cbef)




![image](https://github.com/user-attachments/assets/a9535f32-be1a-4d8a-a2f1-4e726277eaf6)


### Conpare every algorithm

## Conclusion 


**Which algorithm performs best** 

- from the result the algotrithm that perform the best is double q-learn then q-learn then sarsa and MC in the last

**Why does it perform better than the others?**
- Double Q-Learning performs the best in the CartPole problem because it avoids overestimating action values by using two separate q-value , making learning more accurate and stable than Q-Learning, which learns from the best possible next action but it still performs better than SARSA because it learns faster and aims for optimal behavior, while SARSA learns from the actual action taken, making it more cautious and slower to improve. Monte Carlo performs the worst because it only updates after an entire episode ends, making it slow, less efficient, and unstable in environments like CartPole that require fast, step-by-step learning.

**How do the resolutions of the action space and observation space affect the learning process? Why?**
- when the observation and action spaces have high resolution (very detailed), Algorithm has the potential to learn more precise and smarter behavior. But because there are so many possible state to learn from, it needs a lot of exploration to see and understand them all. If it doesn’t explore enough, it will miss important situations, and during the exploitation phase (when it uses what it has learned), it might make poor decisions because it never learned the best actions in those missed states.Therefor if the action space and observation space is change the epsilon decay rate should change to fit the resolution ,also if the resolution of the action space and observation space is too low the agent might not have a precise state to learn enought to exploit more reward and cuase the agent too have poor performance
