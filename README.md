# DiabetesRL
 
DiabetesRL explores reinforcement learning approaches for automated insulin control in Type-1 diabetes patients using the [SimGlucose](https://github.com/jxx123/simglucose) simulator.

The goal is to keep the patient alive and in safe glucose range over a 24 hour simulation by deciding how much insulin to give to the patient at each glucose level. 

The project uses and evaluates PPO-based agents including PPO, PPO with stacked observation space and Recurrent PPO to see how incorporating past observations helps the agent.

Agents are trained on a single Adult patient. 



## observation representation 

observation space for ppo and recurrent ppo model is extended to handle partial observability:

**[normalized CGM,   time-of-day (minutes),   binary meal flag,   CGM slope]**


observation space for the stacked ppo includes past k = 4 observations. 

## reward function 

The reward function encourages glucose to stay in a stable range by penalizing distance from the target blood glucose (110 mg/dL), adding a bonus when glucose moves toward 110, and applying mild penalties for severe
**Hyperglycemia** (> 250 mg/dL) and stronger penalties for **Hypoglycemia** (< 70 mg/dL).


## Results 

* evaluated on the training patient over a single episode (24 hour ismulation)


| Model           | Hours Survived | Time in Range (70–180 mg/dL) | Time Below Range (< 70 mg/dL) | Time Above Range (> 180 mg/dL) |
|-----------------|----------------|------------------------------|-------------------------------|--------------------------------|
| PPO             | 11.55 hr       | 35.93%                       | 64.07%                        | 0.00%                          |
| Stacked PPO     | 22.50 hr       | 42.92%                       | 57.08%                        | 0.00%                          |
| Recurrent PPO   | **24.00 hr**   | **73.75%**                   | **10.00%**                    | **16.25%**                     |


## Experiments

### Survival Evaluation.

* evaluated on the training patient over a 20 episode (20 x 24 hour simulation)

* each step is 3 minutes.


| Model          | Mean Steps | Mean Hours | Std (steps) |
|----------------|-----------:|-----------:|------------:|
| PPO            | 231.1      | 11.55      | 0.4         |
| Stacked PPO    | 464.1      | 23.21      | 14.2        |
| Recurrent PPO  | 480.0      | 24.00      | 0.0         |



### cross patient generalizability experiment.

Trained on a single adult patient, and evaluated on all adults, adolescents, and children, to assess how well the policy generalizes to other patients.

* table shows mean hours survived per age group.

| Patient Group | Stacked PPO (hours) | Recurrent PPO (hours) |
|--------------|---------------------:|-----------------------:|
| Adult (n=10) | 19.15                | 24.00                  |
| Adolescent (n=10) | 8.88            | 21.82                  |
| Child (n=10) | 8.35                 | 11.80                  |



For more details of results and experiments, see  [project.ipynb](project.ipynb).


## Project Structure


```text
DiabetesRL/
├── src/                              
│   ├── make_environment.py            # Environment construction (simple env + stacked env)
│   ├── reward_function.py             # Custom reward definitions for SimGlucose
│   ├── utils.py                       # Evaluation, plotting, and helper utilities
│   └── wrappers.py                    # Gym wrappers (action clipping, feature engineering, stacking observation wrapper)
│
├── trained_policies/                              
│   ├── vanilla_policy_state_dict.pt       # Pretrained vanilla PPO policy weights
│   ├── stacked_policy_state_dict.pt       # Pretrained stacked-observation PPO policy weights
│   └── recurrent_policy_state_dict.pt     # Pretrained Recurrent PPO policy weights
│
│
├── project.ipynb                      # Main evaluation & analysis notebook
├── train.ipynb                        # Notebook for training PPO variants
│
├── requirements.txt                  # Python dependencies to reproduce the environment
└── README.md                        
```


## Setup 

### 1. Clone the Repository
```bash
git clone https://github.com/nedamhs/DiabetesRL.git
cd DiabetesRL
```


### **2. Install Dependencies**

This will install all necessary Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```


## Usage

### TO Load and Evaluate Pretrained Models:

Open `project.ipynb` and run all cells from top to bottom:

```bash
jupyter notebook project.ipynb
```

### To train models:
Open `train.ipynb` to train models:

```bash
jupyter notebook train.ipynb

```





## References

Jinyu Xie. *SimGlucose v0.2.1* (2018). [Online]. Available: https://github.com/jxx123/simglucose. Accessed on: November, 2025.


