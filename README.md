# DiabetesRL
RL for diabetes control


## Project Structure

```text
DiabetesRL/
├── src/
│   ├── wrappers.py            # Action scaling, feature engineering, normalization, and observation stacking wrappers
│   ├── make_environment.py    # Environment builders (vanilla and stacked observation environments)
│   ├── reward_function.py     # Custom reward function definitions
│   └── utils.py               # Utility functions for evaluation, visualization, and rollouts
├── project.ipynb              # Main notebook for evaluation and analysis
├── train.ipynb                # Notebook for training PPO agents
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```
