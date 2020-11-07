# TD3_Pendulum-v0

This TD3 implementation has been modified from Aloïs Pourchot CEM-RL git repository.
The algorithm can be launched this way :  
```python3 td3_launcher_step_study.py```
The default parameters car be seen in the argParser in this file. 

TD3 has achieved a mean performance of ~-146.4 on 900 episodes of Pendulum-v0 (actor_2.pkl).

The policy obtained has a high variance, maybe it can be reduced by tuning TD3 parameters, like discount value, noise or batch size.
To reproduce the results, run evaluate_actor.py on some actor obtained with TD3.
Two actors have been added in this git : one making ~ -150 (actor_1.pkl), one making ~ -146 (actor_2.pkl), measured with the evaluate_actor.
You can run evaluate_actor this way : 
```python3 evaluate_actor.py --env Pendulum-v0 --file actor_2```

Those are some actors obtained quickly on several short runs of TD3.
Result seems not going better over the time passed a certain limit, it will maybe be studied in the future.
