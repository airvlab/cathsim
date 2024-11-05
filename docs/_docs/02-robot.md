---
title: "Endovascular Robot"
permalink: /docs/robot/
excerpt: "How the theme is organized and what all of the files are for."
last_modified_at: 2024-03-18T15:19:22-04:00
---

We develop endovascular robotic systems specifically designed to assist in robot-assisted catheterization tasks. The robotic platforms integrate seamlessly with the CathSim simulator, providing a comprehensive toolset for researchers and clinicians to explore, refine, and practice advanced endovascular techniques.

## CathEase Robot

<div class="image-container">
  <img src="{{ site.baseurl }}/assets/images/robot_cathease.jpg" width="100%" alt="robot_cathease">
</div>

CathEase is a simplified endovascular robot that focuses solely on translating and rotating the guidewire, making its design much easier to replicate compared to more complex multi-DoF systems. It uses a Nema 17 stepper motor for translation, an additional motor for rotation, and is controlled by an Arduino Uno Rev3 with a CNC shield and two A4899 drivers, powered by a 12V DC source. Teleoperation input is provided through a Google Stadia joystick, ensuring an accessible and cost-effective setup for basic endovascular procedures.


## CathBot Robot

<div class="image-container">
  <img src="{{ site.baseurl }}/assets/images/robot_cathbot.jpg" alt="robot_cathbot">
</div>

CathBot is a versatile master-slave robotic system designed for use in Magnetic Resonance (MR) environments. Unlike earlier platforms, CathBot employs pneumatic actuation and additive manufacturing, enabling safe operation within MR settings. The master robot serves as an intuitive human-machine interface (HMI), closely mimicking natural human motion—such as grasping, inserting, retracting, and rotating the instrument—while providing users with haptic feedback from the navigation system. This motion is mapped directly to a 4-degree-of-freedom (DOF) MR-safe slave robot, offering precise control and enhanced user experience during procedures.

<video width="640" height="480" controls>
  <source src="cathbot.mp4" type="video/mp4">
</video>