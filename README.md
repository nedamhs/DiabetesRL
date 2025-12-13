# DiabetesRL

DiabetesRL explores reinforcement learning approaches for automated insulin
control in Type-1 diabetes using the SimGlucose simulator. The project implements
and evaluates PPO-based agents with custom reward functions and observation
representations, focusing on stable and safe glucose dynamics over full-day
simulations.


## Project Structure


```text
DiabetesRL/
├── src/                              
│   ├── make_environment.py            # Environment construction (simple env + stacked env)
│   ├── reward_function.py             # Custom reward definitions for SimGlucose
│   ├── utils.py                       # Evaluation, plotting, and helper utilities
│   └── wrappers.py                    # Gym wrappers (action clipping, feature engineering, stacking observation wrapper)
│
├── project.ipynb                      # Main evaluation & analysis notebook
├── train.ipynb                        # Notebook for training PPO variants
│
├── vanilla_policy_state_dict.pt       # Pretrained vanilla PPO policy weights
├── stacked_policy_state_dict.pt       # Pretrained stacked-observation PPO policy weights
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


