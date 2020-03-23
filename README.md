# Must-read papers on Reinforcement Learning (RL)


## [Content](#content)

<table>
<tr><td colspan="2"><a href="#deep-model-based-rl">1. Deep Model-Based RL</a></td></tr>
<tr>
    <td>&emsp;<a href="#model-is-learned">1.1 Model is Learned</a></td>
    <td>&ensp;<a href="#model-is-known">1.2 Model is Known</a></td>
</tr>
<tr><td colspan="2"><a href="#policy-gradient-and-actor-critic">2. Policy Gradient and Actor-Critic</a></td></tr>
<tr><td colspan="2"><a href="#open-source-reinforcement-learning-platforms-and-algorithm-implementations">3. Open Source Reinforcement Learning Platforms and Algorithm Implementations</a></td></tr>
<tr>
    <td>&emsp;<a href="#platforms">3.1 Platforms</a></td>
    <td>&ensp;<a href="#algorithm-implementations">3.2 Algorithm Implementations</a></td>
</tr>
</table>

## [Deep Model-Based RL](#content)

### [Model is Learned](#content)

**Learning Continuous Control Policies by Stochastic Value Gradients.**  NIPS 2015. *Nicolas Heess, Gregory Wayne, David Silver, Timothy Lillicrap, Tom Erez, Yuval Tassa.* [[Paper](https://arxiv.org/abs/1510.09142)]

**Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning.** ICRA 2018. *Anusha Nagabandi ; Gregory Kahn ; Ronald S. Fearing ; Sergey Levine.* [[Paper](https://arxiv.org/abs/1708.02596)]

**Model-Ensemble Trust-Region Policy Optimization**. ICLR 2018. *Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, Pieter Abbeel.* [[Paper](https://openreview.net/forum?id=SJJinbWRZ&noteId=SJJinbWRZ)]

**Deep reinforcement learning in a handful of trials using probabilistic dynamics models.** NIPS 2018.  *Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine.* [[Paper](http://papers.nips.cc/paper/7725-deep-reinforcement-learning-in-a-handful-of-trials-using-probabilistic-dynamics-models)]

**Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion.** NIPS 2018. *Jacob Buckman, Danijar Hafner, George Tucker, Eugene Brevdo, Honglak Lee.* [[Paper](https://arxiv.org/abs/1807.01675)]

**Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning.** ICML 2018. *Vladimir Feinberg, Alvin Wan, Ion Stoica, Michael I. Jordan, Joseph E. Gonzalez, Sergey Levine.* [[Paper](https://arxiv.org/abs/1803.00101)]

**Algorithmic Framework for Model-based Deep Reinforcement Learning with Theoretical Guarantees.** ICLR 2019. *Yuping Luo, Huazhe Xu, Yuanzhi Li, Yuandong Tian, Trevor Darrell, Tengyu Ma.* [[Paper](https://arxiv.org/abs/1807.03858)]

**When to trust your model: Model-based policy optimization.** NIPS 2019. *Jacob Buckman, Danijar Hafner, George Tucker, Eugene Brevdo, Honglak Lee.* [[Paper](http://papers.nips.cc/paper/9416-when-to-trust-your-model-model-based-policy-optimization)]

**Benchmarking Model-Based Reinforcement Learning.** *Tingwu Wang, Xuchan Bao, Ignasi Clavera, Jerrick Hoang, Yeming Wen, Eric Langlois, Shunshi Zhang, Guodong Zhang, Pieter Abbeel, Jimmy Ba.* [[Paper](https://arxiv.org/abs/1907.02057)]

**Mastering atari, go, chess and shogi by planning with a learned model.** *Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, David Silver.* [[Paper](https://arxiv.org/abs/1911.08265)]

**Learning Latent Dynamics for Planning from Pixels.** ICML 2019.  *Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson.* [[Paper](http://proceedings.mlr.press/v97/hafner19a.html)]

**SOLAR: Deep Structured Representations for Model-Based Reinforcement Learning.** ICML 2019. *Marvin Zhang, Sharad Vikram, Laura Smith, Pieter Abbeel, Matthew Johnson, Sergey Levine.* [[Paper](http://proceedings.mlr.press/v97/zhang19m.html)]

**DeepMDP: Learning Continuous Latent Space Models for Representation Learning**. ICML 2019. *Carles Gelada, Saurabh Kumar, Jacob Buckman, Ofir Nachum, Marc G. Bellemare.* [[Paper](http://proceedings.mlr.press/v97/gelada19a.html)]

**Model-based reinforcement learning for atari.**  Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos, Blazej Osinski, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, Afroz Mohiuddin, Ryan Sepassi, George Tucker, Henryk Michalewski. [[Paper](https://arxiv.org/abs/1903.00374)]

**Stochastic Latent Actor-Critic: Deep Reinforcement Learning with a Latent Variable Model.** Alex X. Lee, Anusha Nagabandi, Pieter Abbeel, Sergey Levine. [[Paper](https://arxiv.org/abs/1907.00953)]

**Exploring Model-based Planning with Policy Networks.** ICLR 2020. *Tingwu Wang, Jimmy Ba.* [[Paper](https://arxiv.org/abs/1906.08649)]

**Model-Augmented Actor-Critic: Backpropagating through Paths.** ICLR 2020. *Ignasi Clavera, Yao Fu, Pieter Abbeel.* [[Paper](https://openreview.net/forum?id=Skln2A4YDB)]

### [Model is Known](#content)

**Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.** *David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis.* [[Paper](https://arxiv.org/abs/1712.01815)]

**Thinking Fast and Slow with Deep Learning and Tree Search.** *Thomas Anthony, Zheng Tian, David Barber.* [[Paper](https://arxiv.org/abs/1705.08439)]

## [Policy Gradient and Actor-critic](#content)

**Policy Gradient Methods for Reinforcement Learning with Function Approximation.** NIPS 2000. *Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour.* [[Paper](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)]

**Approximately Optimal Approximate Reinforcement Learning.** ICML 2002. *Sham Kakade, John Langford.* [[Paper](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf)]

**Deterministic Policy Gradient Algorithms.** ICML 2014. *David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller.* [[Paper](http://proceedings.mlr.press/v32/silver14.pdf)]

**Continuous Control With Deep Reinforcement Learning.** *Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.* [[Paper](https://arxiv.org/abs/1509.02971)]

**Trust Region Policy Optimization.** JMLR 2015. John Schulman, *Sergey Levine, Philipp Moritz, Michael Jordan, Pieter Abbeel.* [[Paper](https://arxiv.org/abs/1502.05477)]

**Proximal Policy Optimization Algorithms.** *John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.* [[Paper](https://arxiv.org/abs/1707.06347)]

**Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.** ICML 2018. *Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.* [[Paper](https://arxiv.org/abs/1801.01290)]


**Addressing Function Approximation Error in Actor-Critic Methods.** PMLR 2018. *Scott Fujimoto, Herke van Hoof, David Meger.* [[Paper](https://arxiv.org/abs/1802.09477)]

**Maximum a Posteriori Policy Optimisation.** ICLR 2018. *Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, Martin Riedmiller.* [[Paper](https://arxiv.org/abs/1806.06920)]

## [Open Source Reinforcement Learning Platforms and Algorithm Implementations](#content)

### [Platforms](#content)

[OpenAI gym](https://github.com/openai/gym) - A toolkit for developing and comparing reinforcement learning algorithms

[OpenAI universe](https://github.com/openai/universe) - A software platform for measuring and training an AI's general intelligence across the world's supply of games, websites and other applications

[OpenAI lab](https://github.com/kengz/openai_lab) - An experimentation system for Reinforcement Learning using OpenAI Gym, Tensorflow, and Keras.

[DeepMind Lab](https://github.com/deepmind/lab) - A 3D learning environment based on id Software's Quake III Arena via ioquake3 and other open source software.

### [Algorithm Implementations](#content)

[rlkit](https://github.com/vitchyr/rlkit) - A reinforcement learning framework with algorithms implemented in PyTorch.

[stable-baselines](https://github.com/hill-a/stable-baselines) - A set of improved implementations of reinforcement learning algorithms based on OpenAI Baselines
