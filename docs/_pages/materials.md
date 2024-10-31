---
permalink: /materials/
title: "Materials"
---



<style>
  .image-container {
    display: flex;
    justify-content: space-around;
    gap: 10px;
    padding: 20px;
  }
</style>

<!-- ![Aortas](/assets/images-cathsim/overview.jpg) -->

<div class="image-container">
  <img src="/assets/images-cathsim/overview.jpg" width="80%" alt="Low Tort Aorta">
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
  <img  width="300px" src="/assets/images-cathsim/rebuttal/low_tort.png" alt="Low Tort Aorta">
  <img  width="300px" src="/assets/images-cathsim/rebuttal/aneurysm.png" alt="Aneurysm Aorta">
  <img  width="300px" src="/assets/images-cathsim/rebuttal/type2.png" alt="Type 2 Aorta">
</div>

Detailed 3D meshes from silicone-based phantoms are used to simulate different aortic structures. Convex hulls generated through V-HACD simplify collision modeling. These models include:

- Type-I Aortic Arch
- Type-II Aortic Arch
- Type-I Aortic Arch with Aneurysm
- Low-Tortuosity Aorta (based on a CT scan)

These models diversify the simulator’s anatomical dataset, enhancing its utility in research and education.

<div class="image-container">
  <img width="300px" src="/assets/images-cathsim/aorta_front.png" alt="Low Tort Aorta">
  <img  width="300px" src="/assets/images-cathsim/aorta_side.png" alt="Aneurysm Aorta">
</div>

## Guidewire

<div class="image-container">
  <img src="/assets/images-cathsim/catheter.png" width="80%" alt="Low Tort Aorta">
</div>

The guidewire is a flexible, segmented structure composed of a main body and a softer tip, connected through revolute or spherical joints. This structure allows real-time simulations with accurate, low-cost shape predictions and a capsule-based collision model to mimic real catheter behavior.

## Blood Simulation

For simplicity, blood is modeled as an incompressible Newtonian fluid with rigid vessel walls. This approach reduces computational demands while realistically opposing guidewire movements.

## Robotic Follower

<div class="image-container">
  <img  width="300px" src="/assets/images-cathsim/cathbot-follower.drawio.jpg" alt="Robotic Follower Design">
  <img  width="300px" src="/assets/images-cathsim/cathbot_sim.png" alt="Simulated Robotic Follower">
</div>

The robotic follower, based on the CathBot design, maintains a linear relationship between leader and follower positions. Our simulation includes four modular platforms attached to a main rail, with prismatic joints for translational movement and revolute joints for rotation.

### Actuation

CathBot’s actuation relies on frictional forces between the guidewire and clamps. In our model, friction prevents slippage, simplifying control over joints without simulating sliding friction. This approach enhances computational efficiency by reducing contact points.
