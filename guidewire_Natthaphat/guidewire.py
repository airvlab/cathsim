import xml.etree.ElementTree as ET
from xml.dom import minidom

# general parameters
SCALE = 1
ELEVATE = 0.003 #m
density = 6450 # nitinol density in SI unit (kg/cu.m)

# body parameters
TOTAL_BODY_LENGTH = 0.5 # m
N_BODIES = 30 # pieces
BODY_DIAMETER = 0.0005 * SCALE # m
BODY_RADIUS = BODY_DIAMETER / 2 # m
BODY_LENGTH = TOTAL_BODY_LENGTH / N_BODIES * SCALE# m

# options
options = {
        "density": "1055",
        "viscosity": "0.004",
        "o_margin": "0.009",
        "integrator": "implicitfast",  # euler, implicit, rk4
        "cone": "pyramidal",
        "jacobian": "sparse",
        "solver": "Newton",  # cg, Newton, pgs
        }

# create the root element
root = ET.Element("mujoco", model="Guidewire")

# extension
ET.SubElement(root, "extension").append(ET.Element("plugin", plugin="mujoco.elasticity.cable"))

# scene and aortic arch integration
ET.SubElement(root, "include", file="scene.xml")
ET.SubElement(root, "include", file="phantom2.xml")

# compiler and memory size
ET.SubElement(root, "compiler", autolimits="false")
ET.SubElement(root, "size", memory="1G")

# options
ET.SubElement(root, "option", attrib=options)

# world body
worldbody = ET.SubElement(root, "worldbody")

# core and joints
core = ET.SubElement(worldbody, "body", name="core", pos=f"0 -0.49 {ELEVATE + BODY_RADIUS}", euler="0 0 90")
ET.SubElement(core, "joint", name="slider", type="slide", axis="1 0 0", stiffness="0", damping="0")
ET.SubElement(core, "joint", name="rotator", type="hinge", axis="1 0 0", stiffness="0", damping="5", armature="0.1")

# cable composite
composite = ET.SubElement(core, "composite", type="cable", curve="s", count=f"{N_BODIES + 1} 1 1", size=f"{TOTAL_BODY_LENGTH}", offset=f"{BODY_RADIUS} 0 0", initial="none")
ET.SubElement(composite, "geom", type="capsule", size=f"{BODY_RADIUS}", rgba="1 1 1 1", condim="1", density=f"{density}", margin="0.008")
ET.SubElement(composite, "joint", kind="main", damping="0.1", stiffness="0.0001", margin="0.008")
composite_plugin = ET.SubElement(composite, "plugin", plugin="mujoco.elasticity.cable")
ET.SubElement(composite_plugin, "config", key="twist", value="4e7")
ET.SubElement(composite_plugin, "config", key="bend", value="7e12")
ET.SubElement(composite_plugin, "config", key="vmax", value="0.001")

# actuators
actuator = ET.SubElement(root, "actuator")
ET.SubElement(actuator, "velocity", name="slider_actuator", joint="slider")
ET.SubElement(actuator, "general", name="rotator_actuator", joint="rotator", biastype="none", dynprm="1 0 0", gainprm="40 0 0", biasprm="0 40 0")

# convert to string
xml_str = ET.tostring(root, encoding="unicode")

# parse the string and pretty-print
dom = minidom.parseString(xml_str)
pretty_xml_str = dom.toprettyxml()

# create XML file
with open("guidewire.xml", "w") as file:
    file.write(pretty_xml_str)