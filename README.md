
### Ada-SQL

This repository is organized as follows:
```
Ada-SQL/
├── evaluate/                     # Code for inference and generation
├── training_codes/               # Training code for reinforcement learning
│   ├── predict_token_budget.py   # predict the reasoning token budget by token budget model  
│   ├── sql_table_column_parse.py   # Used to extract tables and columns from SQL  
│   ├── reward_functions_stage1.py  # Reward function for the RL Stage1
│   ├── reward_functions_stage2.py  # Reward function for the RL Stage2 
│   ├── train_grpo_bird_stage_1.py  # Training code for the RL Stage1
│   └── train_grpo_bird_stage_2.py  # Training code for the RL Stage2
└── README.md
```