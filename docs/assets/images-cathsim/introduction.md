---
permalink: /introduction/
title: "Introduction"
toc: true
---

Welcome to CathSim Project! We developed a simulator for rapid development and
prototyping of algorithms in the context of endovascular navigation.

<style>
  .image-container {
    display: flex;
    justify-content: space-around;
    gap: 10px;
    padding: 20px;
  }
</style>

## Comparison with other simulators

![Simulators](/assets/images-cathsim/table_simulators.png)

<!-- ![Aortas](/assets/images-cathsim/overview.jpg) -->

## Architecture

<div class="image-container">
  <img src="{{ site.baseurl }}/assets/images-cathsim/overview.jpg" width="80%" alt="Low Tort Aorta">
</div>

The CathSim simulator includes four components:  

- **Follower Robot** – Based on the design by Abdelaziz et al., the robot controls catheter movement.
- **Aorta Phantom** – A physical representation of the aorta for realistic simulation.
- **Guidewire Model** – A flexible model directing the catheter along paths in the simulator.
- **Blood Simulation and AR/VR** – Blood modeled as a Newtonian fluid within an augmented or virtual reality environment.

MuJoCo serves as the foundation, chosen for its computational efficiency and machine learning compatibility, which expedites development for endovascular intervention. The modular, real-time, and extensible design allows for easy upgrades.

## Simulation Model

CathSim assumes all components are rigid bodies, a standard in simulators to balance computation speed and accuracy. The components follow continuous-time motion equations:

$$
M\dot{v} +c = \tau +J^Tf
$$

where \( M \) is inertia, \( \dot{v} \) acceleration, \( c \) bias force, and \( \tau \) applied force. The Recursive-Newton-Euler algorithm computes \( c \), while the Composite Rigid-Body algorithm calculates \( M \), with forward kinematics determining these quantities and inverse dynamics applying Newton’s method for \( \tau \).

## Aortic Models

<div class="image-container">
  <img  width="300px" src="{{ site.baseurl }}/assets/images-cathsim/rebuttal/low_tort.png" alt="Low Tort Aorta">
  <img  width="300px" src="{{ site.baseurl }}/assets/images-cathsim/rebuttal/aneurysm.png" alt="Aneurysm Aorta">
  <img  width="300px" src="{{ site.baseurl }}/assets/images-cathsim/rebuttal/type2.png" alt="Type 2 Aorta">
</div>

Detailed 3D meshes from silicone-based phantoms are used to simulate different aortic structures. Convex hulls generated through V-HACD simplify collision modeling. These models include:

- Type-I Aortic Arch
- Type-II Aortic Arch
- Type-I Aortic Arch with Aneurysm
- Low-Tortuosity Aorta (based on a CT scan)

These models diversify the simulator’s anatomical dataset, enhancing its utility in research and education.

<div class="image-container">
  <img width="300px" src="{{ site.baseurl }}/assets/images-cathsim/aorta_front.png" alt="Low Tort Aorta">
  <img  width="300px" src="{{ site.baseurl }}/assets/images-cathsim/aorta_side.png" alt="Aneurysm Aorta">
</div>

## Guidewire

<div class="image-container">
  <img src="{{ site.baseurl }}/assets/images-cathsim/catheter.png" width="80%" alt="Low Tort Aorta">
</div>

The guidewire is a flexible, segmented structure composed of a main body and a softer tip, connected through revolute or spherical joints. This structure allows real-time simulations with accurate, low-cost shape predictions and a capsule-based collision model to mimic real catheter behavior.

## Blood Simulation

For simplicity, blood is modeled as an incompressible Newtonian fluid with rigid vessel walls. This approach reduces computational demands while realistically opposing guidewire movements.

## Robotic Follower

<div class="image-container">
  <img  width="300px" src="{{ site.baseurl }}/assets/images-cathsim/cathbot-follower.drawio.jpg" alt="Robotic Follower Design">
  <img  width="300px" src="{{ site.baseurl }}/assets/images-cathsim/cathbot_sim.png" alt="Simulated Robotic Follower">
</div>

The robotic follower, based on the CathBot design, maintains a linear relationship between leader and follower positions. Our simulation includes four modular platforms attached to a main rail, with prismatic joints for translational movement and revolute joints for rotation.

### Actuation

CathBot’s actuation relies on frictional forces between the guidewire and clamps. In our model, friction prevents slippage, simplifying control over joints without simulating sliding friction. This approach enhances computational efficiency by reducing contact points.

## Validation

<div class="image-container">
  <img width="500px" src="{{ site.baseurl }}/assets/images-cathsim/force_distributions.png" alt="Force Distributions">
</div>

### Statistical Analysis

The empirical force distribution observed in experiments aligns closely with a Gaussian distribution based on real experimental data by Kundrat et al. Using the Mann-Whitney test, we compared these distributions, yielding a statistic of \(76076\) and a p-value of \(p \approx 0.445\). This suggests that any differences are likely due to chance, confirming that our simulator’s force distribution resembles that of the real system, validating CathSim’s capability to mimic real-world robotic interactions.

### User Study

A study with 10 participants assessed CathSim’s realism and effectiveness. Participants, unfamiliar with endovascular procedures, first watched a fluoroscopic video of real navigation, then interacted with CathSim to perform tasks like cannulating key arteries. Feedback was collected on a 5-point Likert scale across seven criteria:

1. **Anatomical Accuracy** – Realism in replicating vessel anatomy.
2. **Navigational Realism** – Authenticity in visual navigation experience.
3. **User Satisfaction** – Overall satisfaction with simulator performance.
4. **Friction Accuracy** – Accuracy in simulating guidewire resistance.
5. **Interaction Realism** – Realism in guidewire-vessel interaction visuals.
6. **Motion Accuracy** – Alignment of guidewire movement with real-life expectations.
7. **Visual Realism** – Visual authenticity of the guidewire in the simulation.

#### User Study Results

| Question             | Average | STD  |
|----------------------|---------|------|
| Anatomical Accuracy  | 4.57    | 0.53 |
| Navigation Realism   | 3.86    | 0.69 |
| User Satisfaction    | 4.43    | 0.53 |
| Friction Accuracy    | 4.00    | 0.82 |
| Interaction Realism  | 3.75    | 0.96 |
| Motion Accuracy      | 4.25    | 0.50 |
| Visual Realism       | 3.67    | 1.15 |

Overall, users provided positive feedback, although visual realism was identified as an area for potential improvement.
