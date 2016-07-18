#Curiolsity-driven Exploration Notes
Reference: Curiosity-driven Exploration in Deep Reinforcement Learning via Bayesian Neural Network
##Abstract:
- Key challenge of RL: scalable(need lower complexity) and effective exploration
- For high-dimensional deep RL senarios:
  - simple heuristics: episilon-greedy, etc...
  - Variational Information Maximizing Exploration(VIME)
- **Maximization of information gain** about **Agent belief of environment dynamics**
- Bayesian neural networks(BNN), efficiently handles **continuous** state and action spaces

##Steps:
####Preliminaries
####Curiosity: theoretical foundation
####Variational Bayes: Adapt curiosity to continuous control
- Tractable solution: derive distribution by Bayes' rule
- Integral tends to be intractable, since nn often need acurate environment model
- Variational inference
- Variational lower bound L: similarity between phi and data set D
- Compute aprroximation of information gain
- Bayesian neural network to parameterize agent's dynamics model
####Compression: compression improvement and the variational lower bound
####Implement:
