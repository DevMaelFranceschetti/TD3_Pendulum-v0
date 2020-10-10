# TD3_Pendulum-v0

This TD3 implementation has been modified from Aloïs Pourchot CEM-RL git repository.
The algorithm can be launched this way :  python3 td3_launcher_step_study.py
The default parameters car be seen in the argParser in this file. 

TD3 can achieve a mean performance of -150 on 900 episodes of Pendulum-v0.
The policy obtained has a high variance, maybe it can be reduced by tuning discount value, noise or batch size.
