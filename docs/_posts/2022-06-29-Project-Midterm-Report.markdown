# TransMimic: Cross Morphologies Motion Skills Transferring

### Team 6

#### Guanming Chen, Shan Jiang, Tianyu Li, Chen Lin, Fu Shen

## Introduction

Learning from demonstration is an efficient method of acquiring new motion skills in robotics, which usually requires the demonstration character and the target character to have similar morphology. For example, human and humanoid[^1][^2], dog and quadrupedal robot[^3]. Yet in the real world, most robots are not human-like nor animal-like. This fact makes it difficult or sometimes unattainable to find their corresponding demonstrations. To resolve this issue and obtain future experiments. In this project, we addressed the issue of learning motions from demonstration provided by a character with different morphologies and studied the properties. In simpler words, we answered the question "can we transfer motions across different morphologies?" To narrow down the problem size, we focused on studying the motions of a mobile manipulation system (Figure 1)[^14] from human demonstrations.

<p align = "center">
<img src="{{site.baseurl}}/img/image1.png" width="50%" height="50%">
</p>
<p style="font-size: 85%; text-align: center;">
Figure 1. A Mobile Manipulation System
</p>

## Problem Definition

Motion transfer between morphologies aka motion retargeting has been widely studied within the computer graphics community. Classical motion retargeting methods rely on optimization concerning hand-crafted kinematics constraints for particular motions[^5][^6]. Designing these constraints often requires a tremendous amount of effort and expertise. To overcome the disadvantage of the engineering method, data-driven methods have been introduced in recent years[^1][^7]. However, most of these studies are only discussed in a monotonous setting, where the demonstration character and target character share the same structure. Yamane et al [^8] addressed motion retargeting between human and non-humanoid characters via Gaussian Process Latent Variable Model. However, this method requires manually pairing data.
In this work, we aim to develop an unsupervised data-driven method for training a motion generator that can transfer motions from one morphology to another.

## Methods

Here we introduce adversarial machine learning, the machine learning techniques against an adversarial opponent, which designs a machine-learning algorithm to enable the generator network to compete against the adversary, i.e., the discriminator network, and studies the capabilities and limitations of the adversary. For generative adversarial networks (GANs), we want to solve the unsupervised learning problem by solving the supervised learning problem (discriminator) and optimization problem (generator) jointly. Given training data without labels. Though the problem is unsupervised, the discriminator is trained to distinguish the real motion (from domain) and the generated fake motion by prior and likelihood, which is a binary classifier, for which we do not use labels as part of the training set. The generator is used for generating realistic motion to mislead the discriminator. In other words, we train a generative model via supervised learning with sub-models: training the generator model to generate new examples, and training the discriminator model to classify examples, the samples from training data, and the generator, such that both models are trained together adversarially until the generator model generates plausible examples. As introduced in [^18], when given a set of reference motions with desired motion style, we use an adversarial discriminator. Then the motion prior will act as a measure of similarity between motions by a character and the motions in the dataset.

<p align = "center">
<img src="{{site.baseurl}}/img/method.jpg" width="100%" height="100%">
</p>
<p style="font-size: 85%; text-align: center;">
Figure 2. Structure Overview
</p>

The core method of this project is Adversarial learning, an unsupervised technique that has demonstrated its effectiveness in many domains. There are two major components in adversarial learning, generator(G) and discriminator(D), both of them are represented as a neural network in our setting. The generator is used to generate realistic robot motion given a human motion as input. The discriminator is used to distinguish between the real motion (from the dataset) and the generated fake motion.

The discriminator is trained in a supervised manner given the label of the motion(real/fake).

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Loss^{D} = D(\bar{x}^q) + [1-D(\hat{x}^q)]" title="eq4" />

Here, <img src="https://latex.codecogs.com/svg.latex?\Large&space;\bar{x}^q" title="eq4" /> is the real robot data from the dataset while <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{x}^q" title="eq4" />is the generated motion from the generator.

The generator is optimized in order to mislead the discriminator. Its loss is designed as:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Loss^{G} = -D(G(\bar{x^p}))" title="eq4" />

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\bar{x^p}" title="eq4" />is the input human motion and <img src="https://latex.codecogs.com/svg.latex?\Large&space;G(\bar{x^p})" title="eq4" />is the generated robot motion. The logic behind this is that, as long as the generator generates realistic motion, the discriminator will unable to distinguish the generated motion, thus, <img src="https://latex.codecogs.com/svg.latex?\Large&space;D(G(\bar{x^p}))"  title="eq4" /> will become higher, and loss will go lower.

Throughout the learning, we train the discriminator and the generator iteratively until they reach an equilibrium point. To stabilize their training, we borrowed a few techniques from literatures[^15] such as feature matching, mini-batch discrimination, historical averaging, one-sided label smoothing, and virtual batch normalization. We may also consider using different neural network structures to improve the learning results.

## Dataset and Data Processing

### CMU Mocap Dataset

We use the CMU motion capture dataset included in the AMASS dataset to serve as our resource of human motion. This dataset contains more than nine hours of motion capture, spanning over 100 subjects. The motions in the dataset are highly diverse and unstructured since the data is captured from different subjects and marker setups. In our initial work, we only select the motions which only contain basic locomotion skills, such as walking and running. The AMASS dataset uses the statistical body shape prior called SMPL+H, where motions and body shape are modeled separately. As a result, the CMU mocap data from different subjects can be reassigned to the same body shape. Here, we use a male model whose body shape parameters are that of the average male. The root joint is offset based on the minimum height for the contact points in each motion.

Specifically, A SMPL-H body model consisted of bones that are connected with joints. There are 52 joints and 22 of them belong to the main body and the other 30 joints are for hands. Each joint owns 3 rotational degrees.

<p align = "center">
<img src="{{site.baseurl}}/img/smpl.png" width="50%" height="50%">
</p>
<p style="font-size: 85%; text-align: center;">
Figure 3. Body part segmentation for the SMPL model [^21]
</p>

### Dog Mocap Dataset

Our motion capture data consists of 30 minutes of unstructured dog motion capture data that is composed of various locomotion modes including walk, pace, trot, and canter, as well as other types of motions such as sitting, standing, idling, lying, and jumping. The skeleton model consists of 27 bones which leads to a total of 81 degrees of freedom for the whole body.

When using the motion from recorded animal motion, the subject’s morphology tends to be different from the target robot’s. In order to tackle this discrepancy, the source motion is manually retargeted to the robot’s morphology based on its kinematics property. First, a set of source key points are specified on the subject’s body, paired with the corresponding target key points on the robot body. These key points contain the hips and feet position. At each timestep, the source motion specifies the 3D position of each key point. The corresponding target keypoint is then determined via inverse kinematics.[^16]

## Experiments and Initial Results

### Sanity Check Experiments

Although the effectiveness of adversarial learning methods has been demonstrated in many domains, such as computer vision and NLP, there is a lack of work that applies these methods in the motion generation field. Thus, we conducted a series of simple experiments to answer whether we can generate realistic robot motion via adversarial learning as well as studying the properties of the learning method.

In the first experiment, we test whether is possible to learn a robot walking motion via our method. The experiment setting n in the following figure:

<p align = "center">
  <img src="{{site.baseurl}}/img/exp_1.png" width="50%" height="50%">
</p>
<p style="font-size: 85%; text-align: center;">
  Figure 4. Learning Framework
</p>

In this experiment, the generator takes the robot's current observation <img src="https://latex.codecogs.com/svg.latex?\Large&space;\bar{x}_t" title="eq1" /> and current latent variable <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{z}_t" title="eq1" /> as input to predict the next robot state<img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{x}_{t+1}" title="eq2" />. The discriminator uses the state transition <img src="https://latex.codecogs.com/svg.latex?\Large&space;\{\bar{x}_t, \hat{x}_{t+1}\}" title="eq3" /> to justify whether the transition is realistic or not. The definition of these variables are listed here:

- <img src="https://latex.codecogs.com/svg.latex?\Large&space;\bar{x}^q_t = \{q^{root}_{t-N:t+N}, q^{ee}_{t}\}" title="eq4" /><br>
  Root trajectory and end effector positions in the 2-D horizontal plane relative to the robot frame at time t. The time horizon is 2N = 10 in our setting.<br><br>
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{x}^q_{t+1} =  \{r^{root}_t, q^{root}_{t+1-N:t+1+N}, q^{ee}_{t+1}\}" title="eq5" /><br>
  Root trajectory and end effector positions in the 2-D horizontal plane relative to the robot frame at time t+1.<br><br>
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{z}_t" title="eq6" />
  Latent variable, here we use the robot’s desired velocity, calculated via robot data.

In this experiment, for each input, we already have the ground truth output which can be generated from the dataset. Thus, before testing our method, we trained a generator using supervised learning to ensure the hyperparameter setting, as well as the input output space design, can generate realistic motion. We also test the adversarial loss and supervised discriminator loss to set a standard for our method. The motion generated by the supervised learning trained generator and the training curve is resented in the following figure:

<p align = "center">
  <img src="{{site.baseurl}}/img/exp_result_0.png">
</p>
<p style="font-size: 85%; text-align: left;">
  Figure 5. Supervised learning experiment learning curve. The adversarial loss and the discriminator loss both converage to a certain level.
</p>

The figure shows that as the learning goes on, the adversarial loss and the supervised learning loss converge to a certain point. This result indicates that the generator is able to generate motions that could not be distinguished by the trained discriminator.

Our work will transfer a series of human motions to mobile manipulation system. Since our method is machine-learning-based, quantitative metrics such as accuracy, F-1 score, precision, and recall score would be considered to evaluate the model. If the model is well-designed, the performance scores would be pleasant. Also, in the final demo, we want to show the motion animations of a mobile manipulation system generated by transferring human motion, which we believe will show the outcome performance if the animation simulation can be observed directly.

Then, we implement our simplest experiment: using adversarial learning to generate a single realistic motion. The result motion and learning curve are shown in Figure 4. The result motion is almost identical comparing to the real motion. The training plot also indicates that the adversarial loss and discriminator supervised loss converge to a balanced level as expected.

<p align = "center">
  <img src="{{site.baseurl}}/img/exp_result_1.png">
</p>
<p  style = "font-size: 90%; text-align: left;">
  Figure 6(a). The learning curve for adversarial learning(blue) compared to the supervised learning curve(orange)
</p>
<p align = "center">
  <img src="{{site.baseurl}}/img/exp_gif_1.gif" width="40%" height="40%"> 
  <img src="{{site.baseurl}}/img/exp_tgt_1.gif" width="40%" height="40%">
</p>
<p  style = "font-size: 85%; text-align: left;">
Figure 6(b). Result of the first experiment. The learned motion(Left) is almost identical to the source motion(Right).
</p>

Besides simple walking motion, we also tested our method on generating more complex motions such as slow turning. The result shows that the generator can still acquire that motions via our method.

<p align = "center">
  <img src="{{site.baseurl}}/img/exp_gif_2.gif"  width="40%" height="40%">
  <img src="{{site.baseurl}}/img/exp_tgt_2.gif" width="40%" height="40%">
</p>
<p style = "font-size: 85%; text-align: left;">
  Figure 7. Result learning a more complex motion. The learned motion(Left) is still 'somewhat' copying the source motion(Right).
</p>

Finally, we tested if a single human walking trajectory can be mapped to a robot walking trajectory. We slightly modified our method by replacing the engineering latent variable <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{z}_t" title="eq4" />with a human motion trajectory<img src="https://latex.codecogs.com/svg.latex?\Large&space;\bar{x}^p_t" title="eq4" />. As shown in the resulting video, our learning framework is able to map a human trajectory to a robot trajectory without pairing data.

<p align = "center">
  <img src="{{site.baseurl}}/img/exp_3.png" width="50%" height="50%">
</p>
<p style="font-size: 85%; text-align: center;">Figure 8. Human to robot motion transfer learning Framework</p>
<p align = "center">
  <img src="{{site.baseurl}}/img/human2robot_init.gif" width="40%" height="40%">
</p>
<p style="font-size: 85%; text-align: center;">
  Figure 9. Human to robot motion transfer learning initial result.
</p>

## Potential Results and Discussion

Promising as our initial result, we still have several points which we could address:

- Our initial experiment is relatively simple, in our final result, we want to include more human and robot motions.
- The input and output space need to be redesigned for high-quality motion generation as well as better generalization of our learning framework.

## Future Work

Since both the generator model and the discriminator model are trained simultaneously it makes GANs hard to train. Due to this, we need to find an equilibrium point between the two models, otherwise improvements to one model will cause the expense of the other model [^17]. Also, every time we update the parameters of one model, the solution to the optimization problem we just solved will change. The stable training of GANs remains an open problem, Radford et al. [^19] crafted a deep convolutional GAN (DCGAN) architecture, which resulted in stable training across a range of datasets and gets training higher resolution and deeper generative models. Tim Salimans et al. [^17] mentioned five techniques to improve convergence when training GANs: feature matching, mini-batch discrimination, historical averaging, one-sided label smoothing, and virtual batch normalization. Also, since in practice, the discriminator is usually deeper and has more filters per layer than the generator by NIPS, we may not schedule more training in the generator or discriminator based on the relative changes in the loss. There are some other tips, for example, flipping the labels and loss function when training the generator, adding random noise to the tables in the discriminator, using DCGAN architecture, etc. We may apply some tips to our model after experiments.

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
[^10]: Adobe Dog Motion Capture Dataset https://github.com/sebastianstarke/AI4Animation#ai4animation-deep-learning-for-character-control
[^11]: https://deepmotionediting.github.io/papers/skeleton-aware-camera-ready.pdf
[^12]: https://deepmotionediting.github.io/papers/Motion_Style_Transfer-camera-ready.pdf
[^13]: https://xbpeng.github.io/projects/Robotic_Imitation/2020_Robotic_Imitation.pdf
[^14]: Figure downloaded from https://www.wevolver.com/specs/spot.mini
[^15]: Li, R., Li, X., Fu, C. W., Cohen-Or, D., & Heng, P. A. (2019). Pu-gan: a point cloud upsampling adversarial network. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 7203-7212).
[^16]: Zhang, H., Starke, S., Komura, T., & Saito, J. (2018). Mode-adaptive neural networks for quadruped motion control. ACM Transactions on Graphics (TOG), 37(4), 1-11.
[^17]: Salimans, Tim and Goodfellow, Ian and Zaremba, Wojciech and Cheung, Vicki and Radford, Alec and Chen, Xi. (2016). Improved Techniques for Training GANs. arXiv: 1606.03498.
[^18]: Peng, Xue Bin and Ma, Ze and Abbeel, Pieter and Levine, Sergey and Kanazawa, Angjoo. (2021). AMP Adversarial Motion Priors for Stylized Physics-Based Character Control. ACM Transactions on Graphics (TOG), 40(4), (pp. 1-20).
[^19]: Radford, Alec and Metz, Luke and Chintala, Soumith. (2016). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. arXiv: 1511.06434.
[^20]: Zhang, H., Starke, S., Komura, T., & Saito, J. (2018). Mode-adaptive neural networks for quadruped motion control. ACM Transactions on Graphics (TOG), 37(4), 1-11.
[^21]: Ranjan, A., Hoffmann, D. T., Tzionas, D., Tang, S., Romero, J., & Black, M. J. (2020). Learning multi-human optical flow. International Journal of Computer Vision, 128(4), 873-890.
