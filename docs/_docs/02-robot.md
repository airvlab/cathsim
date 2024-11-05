---
title: "Endovascular Robot"
permalink: /docs/robot/
excerpt: "Endovascular Robot."
---

We develop endovascular robotic systems specifically designed to assist in robot-assisted catheterization tasks. The robotic platforms integrate seamlessly with the CathSim simulator, providing a comprehensive toolset for researchers and clinicians to explore endovascular techniques.

## CathEase Robot

<div class="image-container">
  <img src="{{ site.baseurl }}/assets/images/robot_cathease.jpg" width="100%" alt="robot_cathease">
</div>

CathEase is an accessible and cost-effective endovascular robot that focuses on translating and rotating the guidewire. It uses a Nema 17 stepper motor for translation, an additional motor for rotation, and is controlled by an Arduino Uno Rev3 with a CNC shield and two A4899 drivers, powered by a 12V DC source. Teleoperation input is provided through a Google Stadia joystick.


## CathBot Robot

<div class="image-container">
  <img src="{{ site.baseurl }}/assets/images/robot_cathbot.jpg" alt="robot_cathbot">
</div>

CathBot is a versatile master-slave robotic system designed for use in MRI environments. CathBot employs pneumatic actuation, enabling safe operation within MR settings. The master robot closely mimicks natural human motion such as grasping, retracting, and rotating the instrument while providing users with haptic feedback. This motion is mapped directly to a 4DoF MRI-safe slave robot, offering precise control and enhanced user experience during procedures.

Watch our robot in action in the video below:
<video width="640" height="420" controls>
  <source src="{{ site.baseurl }}/assets/videos/cathbot.mp4" alt="cathbot video" type="video/mp4">
</video>