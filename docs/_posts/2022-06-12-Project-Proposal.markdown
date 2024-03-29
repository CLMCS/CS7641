# TransMimic: Cross Morphologies Motion Skills Transferring 
### Team 6 
#### Guanming Chen, Shan Jiang, Tianyu Li, Chen Lin, Fu Shen


## Introduction
Learning from demonstration is an efficient method of acquiring new skills in robotics, which usually requires the demonstration character and the target character to have similar morphology. For example, human and humanoid[^1][^2] dog and quadrupedal robot[^3]. Yet in the real world, most robots are not animal-like, which makes it hard or impossible to find their corresponding demonstrations. We address the issue of learning from demonstration provided by a character with different morphology, or, can we transfer motions across different morphologies? Specifically, we focus on learning the motion of a mobile manipulation system (Figure 1)[^14] from human demonstrations.

<p align = "center">
<img src="{{site.baseurl}}/img/image1.png" width="50%" height="50%">
</p>
<p align = "center">
Figure 1. A Mobile Manipulation System
</p>

## Problem Definition

Motion transferring/retargeting across morphologies has widely studied in the computer graphics community. Classical motion retargeting methods rely on optimization with respect to hand-crafted kinematics constraints for particular motions[^5][^6]. Designing these constraints often requires a tremendous amount of effort and expertise. To overcome the disadvantage of the engineering method, several data-driven methods have been introduced in recent years[^1][^7]. However, most of them only discuss in a homophobic setting, where the demonstration character and target character shares the same structure. One of the few exceptions is [^8]. Their work manually paired a small set of motions between two characters with different morphologies and used the Gaussian Process Latent Variable Model to generalize to other motions.

## Methods
In this project, we introduce an unsupervised learning method to learn a generator function which transfers human motion to a mobile manipulation system. Comparing to the previous work, for our work:
- The motion is transferred between very different morphologies
- We don’t pair the motion between the source and target morphologies.

The datasets we will use are: 
1. CMU Mocap Dataset for human demonstration[^9]. 
2. Adobe Mocap Dataset for quadrupedal motion[^10]. 

This data-driven method, with the increasing availability of data, will get more attention to motion retargeting, which is usually formulated as: 
1. Given the motion dataset of the source and target morphology.
2. Learn a neural network transfer function in a GAN manner along with other constraints.

For the machine-learning parts, the model consists of two aspects of learning: 
1. A discriminator trained via supervised learning.
2. A neural network generator is trained in an adversarial manner. 

The discriminator is trained to distinguish the real motion and the generated fake motion. The generator is used for generating realistic robot motion to mislead the discriminator.

## Potential Results and Discussion
Our work will transfer a series of human motions to mobile manipulation system. Since our method is machine-learning-based, quantitative metrics such as accuracy, F-1 score, precision and recall score would be considered to evaluate the model. If the model is well-designed, the performance scores would be pleasant. Also, in the final demo, we want to show the motion animations of a mobile manipulation system generated by transferring human motion, which we believe will show the outcome performance if the animation simulation can be observed directly.



## Plan of Activities
### Timeline
All team members have equal contributions.
<p align = "center">
<img src="{{site.baseurl}}/img/tasks.png">
</p>

  
## References
[^1]: Aberman, K., Li, P., Lischinski, D., Sorkine-Hornung, O., Cohen-Or, D., & Chen, B. (2020). Skeleton-aware networks for deep motion retargeting. ACM Transactions on Graphics (TOG), 39(4), 62-1. 
[^2]: Aberman, K., Weng, Y., Lischinski, D., Cohen-Or, D., & Chen, B. (2020). Unpaired motion style transfer from video to animation. ACM Transactions on Graphics (TOG), 39(4), 64-1. 
[^3]: Peng, X. B., Coumans, E., Zhang, T., Lee, T. W., Tan, J., & Levine, S. (2020). Learning agile robotic locomotion skills by imitating animals. arXiv preprint arXiv:2004.00784. 
[^4]: Spot Mini, https://www.wevolver.com/specs/spot.mini 
[^5]: Lee, J., & Shin, S. Y. (1999, July). A hierarchical approach to interactive motion editing for human-like figures. In Proceedings of the 26th annual conference on Computer graphics and interactive techniques (pp. 39-48). 
[^6]: Choi, K. J., & Ko, H. S. (2000). Online motion retargeting. The Journal of Visualization and Computer Animation, 11(5), 223-235. 
[^7]: Villegas, R., Yang, J., Ceylan, D., & Lee, H. (2018). Neural kinematic networks for unsupervised motion retargeting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 8639-8648). 
[^8]: Yamane, K., Ariki, Y., & Hodgins, J. (2010, July). Animating non-humanoid characters with human motion data. In Proceedings of the 2010 ACM SIGGRAPH/Eurographics Symposium on Computer Animation (pp. 169-178). 
[^9]: CMU Graphics Lab Motion Capture Database, http://mocap.cs.cmu.edu/ 
[^10]: Adobe Dog  Motion Capture Dataset https://github.com/sebastianstarke/AI4Animation#ai4animation-deep-learning-for-character-control 
[^11]: https://deepmotionediting.github.io/papers/skeleton-aware-camera-ready.pdf 
[^12]: https://deepmotionediting.github.io/papers/Motion_Style_Transfer-camera-ready.pdf 
[^13]: https://xbpeng.github.io/projects/Robotic_Imitation/2020_Robotic_Imitation.pdf 
[^14]: Figure downloaded from https://www.wevolver.com/specs/spot.mini 


