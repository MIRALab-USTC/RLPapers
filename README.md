# Must-read papers on Reinforcement Learning (RL)


## [Content](#content)

<table>

</tr>
<tr><td colspan="2"><a href="#policy-gradient-and-actor-critic">1. Policy Gradient and Actor-Critic</a></td></tr>


<tr><td colspan="2"><a href="#model-based-rl">2. Model-Based RL</a></td></tr>

<tr><td colspan="2"><a href="#offline-rl">3. Offline RL</a></td></tr>

<tr><td colspan="2"><a href="#exploration">4. Exploration</a></td></tr>

<tr><td colspan="2"><a href="#open-source-reinforcement-learning-platforms-and-algorithm-implementations">5. Open Source Reinforcement Learning Platforms and Algorithm Implementations</a></td></tr>

<tr>
    <td>&emsp;<a href="#platforms">5.1 Platforms</a></td>
    <td>&ensp;<a href="#algorithm-implementations">5.2 Algorithm Implementations</a></td>
</tr>
</table>


## [Policy Gradient and Actor-critic](#content)

1. **Policy Gradient Methods for Reinforcement Learning with Function Approximation**. NIPS 2000. [[Paper](http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)]  
    *Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour*.
    > This paper lists several limitations of value-based methods, including the difficulty of learning stochastic policies and the non-robustness during the training process. To alleviate the above limitations, the authors propose the *first* policy gradient method with function approximation.

2. **Off-Policy Actor-Critic**. ICML 2012. [[Paper](https://icml.cc/2012/papers/268.pdf)]  
    *Thomas Degris, Martha White, Richard S. Sutton.*
    > This paper presents the *first* actor-critic algorithm for off-policy reinforcement learning, called the off-policy actor-critic algorithm (Off-PAC), to improve sample efficiency by reusing previous experience.

3. **Deterministic Policy Gradient Algorithms**. ICML 2014. [[Paper](http://www.jmlr.org/proceedings/papers/v32/silver14.pdf)]   
    *David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller*.
    > This paper proposes an off-policy actor-critic algorithm called the deterministic policy gradient (DPG) algorithm. Experiments show that the DPG algorithm outperforms stochastic policy gradient methods in terms of sample efficiency.

4. **Trust Region Policy Optimization**. ICML 2015. [[Paper](http://proceedings.mlr.press/v37/schulman15.pdf)]  [[Code](https://github.com/openai/baselines)]  
    *John Schulman, Sergey Levine, Philipp Moritz, Michael Jordan, Pieter Abbeel*.
    > The authors propose a policy optimization algorithm, called Trust Region Policy Optimization (TRPO), by making several approximations to an iterative procedure that gives guaranteed monotonic improvement for optimizing policies.

5. **Proximal Policy Optimization Algorithms**. arXiv preprint 2017. [[Paper](https://arxiv.org/pdf/1707.06347.pdf)]  [[Code](https://github.com/openai/baselines)]  
    *John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov*.
    > The authors propose a new family of policy gradient methods, called proximal policy optimization (PPO). They demonstrate that the proposed methods are simpler to implement and have better sample complexity (empirically) than TRPO.

6. **Addressing Function Approximation Error in Actor-Critic Methods**. ICML 2018. [[Paper](https://arxiv.org/pdf/1802.09477.pdf)][[Code](https://github.com/sfujim/TD3https://github.com/sfujim/TD3)]  
    *Scott Fujimoto, Herke van Hoof, David Meger*.
    > This paper identifies value overestimation in actor-critic methods. This paper then proposes Twin Delayed Deep Deterministic policy gradient (TD3) to reduce overestimation bias by introducing three critical tricks: Clipped Double-Q Learning, delayed policy updates, and target policy smoothing.

7. **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**.  ICML 2018. [[Paper](https://arxiv.org/pdf/1801.01290)] [[Code](https://github.com/haarnoja/sac)]  
    *Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine*.
    > This paper proposes soft actor-critic (sac), which is an off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework. 

8. **Soft Actor-Critic Algorithms and Applications**. arXiv preprint 2018. [[Paper](https://arxiv.org/pdf/1812.05905)] [[Code](https://github.com/rail-berkeley/softlearning)]  
    *Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine*.
    > This paper changes the original SAC objective to a constrained formulation, enabling the algorithm to tune the temperature hyperparameter in the original SAC objective automatically.

9. **Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics**. ICML 2020. [[Paper](https://arxiv.org/pdf/2005.04269)]  
    *Arsenii Kuznetsov, Pavel Shvechikov, Alexander Grishin, Dmitry Vetrov*.
    > This paper proposes to alleviate the overestimation bias via *Truncated Quantile Critics (TQC)*, which consists of three ingredients, distributional representation of a critic, truncation of critics prediction, and ensembling of multiple critics.


## [Model-Based RL](#content)

1. **Model-Ensemble Trust-Region Policy Optimization**. ICLR 2018.  [[Paper](https://arxiv.org/pdf/1802.10592)]    
   *Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, Pieter Abbeel*.
    > This paper proposes the algorithm MB-TRPO, which uses an ensemble of models to generate fictitious samples and then optimizes the policy using TRPO on these fictitious samples. MB-TRPO is the first purely model-based approach that can optimize policies over high-dimensional motor-control tasks, such as Humanoid.

1. **Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models**. NeurIPS 2018. [[Paper](http://papers.nips.cc/paper/7725-deep-reinforcement-learning-in-a-handful-of-trials-using-probabilistic-dynamics-models.pdf)]    
   *Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine*
   > This paper proposes probabilistic ensembles with trajectory sampling (PETS) that combines uncertainty-aware deep network dynamics models with sampling-based uncertainty propagation. Experiments demonstrate that PETS achieves comparable results with SAC, while requiring significantly fewer samples.

2. **When to Trust Your Model: Model-Based Policy Optimization**. NeurIPS 2019. [[Paper](https://papers.nips.cc/paper/9416-when-to-trust-your-model-model-based-policy-optimization.pdf)]  [[Code](https://github.com/JannerM/mbpo.git)]  
    *Michael Janner, Justin Fu, Marvin Zhang, Sergey Levine*.
    > This paper proposes the notion of a *branched rollout*, in which a rollout is begined from a real-world state that is sampled by a previous policy.


## [Offline RL](#content)

1. **Off-Policy Deep Reinforcement Learning without Exploration**. ICML 2019. [[Paper](https://arxiv.org/pdf/1812.02900.pdf)]   
    *Scott Fujimoto, David Meger, Doina Precup*.
    > This paper identifies extrapolation error, which comes from the mismatch between the dataset and true state-action visitation of the current policy. This paper then proposes Batch-Constrained deep Q-learning (BCQ) to mitigate extrapolation error via a constraint limiting the distance of selected actions to the offline dataset.



2. **Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction**. NeurIPS 2019. [[Paper](https://arxiv.org/pdf/1906.00949.pdf)]  
    *Aviral Kumar, Justin Fu, George Tucker, Sergey Levine*.
    > This paper identifies bootstrapping error due to bootstrapping from actions that lie outside of the training data distribution. This paper then proposes Bootstrapping Error Accumulation Reduction (BEAR) to mitigate bootstrapping error via a constraint limiting the maximum mean discrepancy between the learned policy and the behavior policy.

3. **An Optimistic Perspective on Offline Reinforcement Learning**. ICML 2020. [[Paper](https://arxiv.org/pdf/2005.05951.pdf)][[code](https://github.com/google-research/batch_rl)]  
    *Rishabh Agarwal, Dale Schuurmans, Mohammad Norouzi*.
    > This paper demonstrates that robust RL off-policy algorithms such as QR-DQN work well on sufficiently large and diverse offline datasets without other offline RL techniques. To enhance robustness and generalization in the offline setting, this paper proposes Random Ensemble Mixture (REM) that enforces optimal Bellman consistency on random convex combinations of multiple Q-value estimates.


## [Exploration](#content)



1. **Model-Based Active Exploration**. ICML 2019. [[Paper](http://proceedings.mlr.press/v97/shyam19a/shyam19a.pdf)] [[Code]( https://github.com/nnaisense/max)]  
    *Pranav Shyam, Wojciech Jaśkowski, Faustino Gomez*.
    > This paper proposes an efficient active exploration algorithm, Model-Based Active eXploration (MAX), which uses an ensemble of forward models to plan to observe novel events.



1. **Curiosity-driven Exploration by Self-supervised Prediction**. ICML 2017. [[Paper](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)]  [[Code](https://github.com/pathak22/noreward-rl)]  
    *Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, Trevor Darrell*.
    > This paper proposes a curiosity-driven exploration method, which formulates curiosity as the error in an agent's ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model. 

1. **Variational Information Maximizing Exploration.** NIPS 2016. [[Paper](https://openreview.net/forum?id=Hyc07bR9&noteId=Hyc07bR9)]  
    *Rein Houthooft, Xi Chen, Yan Duan, John Schulman, Filip De Turck, Pieter Abbeel*.
    > This paper proposes Variational Information Maximizing Exploration (VIME), which is an exploration method based on maximizing information gain about the agent's belief of environment dynamics. 

1. **Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning**. NIPS 2017. [[Paper](http://papers.neurips.cc/paper/6868-exploration-a-study-of-count-based-exploration-for-deep-reinforcement-learning.pdf)]  
    *Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, Xi Chen, Yan Duan, John Schulman, Filip De Turck, Pieter Abbeel*.
    > The authors extend classic counting-based methods to high-dimensional, continuous state spaces, where they discretize the state space with a hash function and apply a bonus based on the state-visitation count. 

1. **Unifying Count-Based Exploration and Intrinsic Motivation**. NIPS 2016. [[Paper](https://papers.nips.cc/paper/6383-unifying-count-based-exploration-and-intrinsic-motivation.pdf)]  
    *Marc G. Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, Remi Munos*.
    > The authors use density models to measure uncertainty, and propose an algorithm for deriving a pseudo-count from a density model. They generalize count-based exploration algorithms to the non-tabular case.

## [Open Source Reinforcement Learning Platforms and Algorithm Implementations](#content)

### [Platforms](#content)

1. [**OpenAI gym**](https://github.com/openai/gym)
    > A toolkit for developing and comparing reinforcement learning algorithms.

1. [**OpenAI universe**](https://github.com/openai/universe) 
    > A software platform for measuring and training an AI's general intelligence across the world's supply of games, websites and other applications.

1. [**OpenAI lab**](https://github.com/kengz/openai_lab)
    > An experimentation system for Reinforcement Learning using OpenAI Gym, Tensorflow, and Keras.

1. [**DeepMind Lab**](https://github.com/deepmind/lab)
    > A 3D learning environment based on id Software's Quake III Arena via ioquake3 and other open source software.

### [Algorithm Implementations](#content)

1. [**rlkit**](https://github.com/vitchyr/rlkit)
    > A reinforcement learning framework with algorithms implemented in PyTorch.

1. [**stable-baselines**](https://github.com/hill-a/stable-baselines)
    > A set of improved implementations of reinforcement learning algorithms based on OpenAI Baselines.
