#Variation Information Maximisation Notes
##Abstract
- For intrinsically motivated reinforcement learning
- For reduce agent's uncertanty of environment
- Intrinsic reward
- No external reward

##Background
- Entropy: uncertanty ( of environment )
- Mutual Information (MI): I(X,Y), represent shared information = H(X) - H(X|Y) = H(Y) - H(Y|X)

##Steps
- Big Episilon(s) = max(w(policy)) I(a,s_plum|s) , How action(a) can control future state(s_plum)
- Tend to go to the state of maximum big episilon(s)
- Cannot efficiently compute H(a|s_plum,s)
- Use q_persi(a|s_plum,s) to estimate p(a|s_plum,s)
- To make q_persi as close as p, use KL Divergence. Minimize D_KL(p||q)
- By KL Divergence, we can get a lower bound of big episilon
- When MI get bigger, q_persi will become closer to p
- Then, we can get the big episilon

##Optimization

##How to apply on training RL?
