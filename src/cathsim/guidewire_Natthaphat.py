import xml.etree.ElementTree as ET
from xml.dom import minidom

# general parameters
SCALE = 1
ELEVATE = 0.003 # m
INSERT = 0.01 # m
DENSITY = 6450 # nitinol density in SI unit (kg/cu.m)

# body parameters
TOTAL_BODY_LENGTH = 0.5 # m
N_BODIES = 30 # pieces
BODY_DIAMETER = 0.0005 * SCALE # m
BODY_RADIUS = BODY_DIAMETER / 2 # m
BODY_LENGTH = TOTAL_BODY_LENGTH / N_BODIES * SCALE # m

# options for environment (human blood)
options = {
        "density": "1055", # kg/cu.m
        "viscosity": "0.0035", # Pa
        "o_margin": "0.02",
        "integrator": "implicitfast", # euler, implicit, rk4
        "collision": "all", # "all", predefined, dynamic
        "cone": "pyramidal", # "pyramidal", elliptic
        "jacobian": "sparse", # dense, sparse, "auto"
        "solver": "Newton",  # PGS, CG, "Newton"
        }

def create_guidewire():
    """
    Create an XML representation for a MuJoCo guidewire model using the elasticity plugin.
    
    This function constructs an XML tree that describes a guidewire model for MuJoCo. The model integrates 
    the elasticity plugin for the guidewire to simulate its behavior in a realistic manner. The XML model 
    includes several elements, including the aortic arch, compiler settings, world body properties (such as lighting), 
    the core structure of the guidewire with its joints, the cable composite of the guidewire, and actuators.
    
    Returns:
        str: A prettified XML string representation of the guidewire model.
    """
    try:
        # create the root element
        root = ET.Element("mujoco", model="Guidewire")

        # extension
        ET.SubElement(root, "extension").append(ET.Element("plugin", plugin="mujoco.elasticity.cable"))

        # aortic arch integration
        ET.SubElement(root, "include", file="assets/phantom2.xml")

        # compiler and memory size
        ET.SubElement(root, "compiler", autolimits="false", meshdir="assets/meshes")
        ET.SubElement(root, "size", memory="1G")

        # options
        ET.SubElement(root, "option", attrib=options)

        # world body
        worldbody = ET.SubElement(root, "worldbody")

        # light
        ET.SubElement(worldbody, "light", diffuse=".4 .4 .4", specular="0.1 0.1 0.1", pos="0 0 2", dir="0 0 -1", castshadow="false")
        ET.SubElement(worldbody, "light", directional="true", diffuse=".8 .8 .8", specular="0.2 0.2 0.2", pos="0 0 4", dir="0 0 -1")

        # core and joints
        core = ET.SubElement(worldbody, "body", name="core", pos=f"0 {INSERT - TOTAL_BODY_LENGTH} {ELEVATE + BODY_RADIUS}", euler="0 0 90")
        ET.SubElement(core, "joint", name="slider", type="slide", axis="1 0 0", stiffness="0", damping="0")
        ET.SubElement(core, "joint", name="rotator", type="hinge", axis="1 0 0", stiffness="0", damping="5", armature="0.1")

        # cable composite
        composite = ET.SubElement(core, "composite", type="cable", curve="s", count=f"{N_BODIES + 1} 1 1", size=f"{TOTAL_BODY_LENGTH}", offset=f"{BODY_RADIUS} 0 0", initial="none")
        ET.SubElement(composite, "geom", type="capsule", size=f"{BODY_RADIUS}", rgba="1 1 1 1", condim="1", density=f"{DENSITY}", margin="0.02")
        ET.SubElement(composite, "joint", kind="main", damping="0.1", stiffness="0.01", margin="0.02")
        composite_plugin = ET.SubElement(composite, "plugin", plugin="mujoco.elasticity.cable")
        ET.SubElement(composite_plugin, "config", key="twist", value="1e6")
        ET.SubElement(composite_plugin, "config", key="bend", value="1e9")
        ET.SubElement(composite_plugin, "config", key="vmax", value="0.0005")

        # actuators
        actuator = ET.SubElement(root, "actuator")
        ET.SubElement(actuator, "velocity", name="slider_actuator", joint="slider")
        ET.SubElement(actuator, "general", name="rotator_actuator", joint="rotator", biastype="none", dynprm="1 0 0", gainprm="40 0 0", biasprm="0 40 0")

        # convert to string
        xml_str = ET.tostring(root, encoding="unicode")
        dom = minidom.parseString(xml_str)
        pretty_xml_str = dom.toprettyxml()
        return pretty_xml_str
    
    except ET.ParseError as e:
        print(f"Error during XML generation: {e}")
        return None
    
def create_xml_file(pretty_xml_str, file_name):
    """
    Save the provided XML string to a file.
    
    This function writes the given prettified XML string to an XML file with the specified file name.
    
    Parameters:
        pretty_xml_str (str): The prettified XML string to be written to a file.
        file_name (str): The name of the file (without extension) to which the XML string will be written.
        
    Returns:
        None
    """
    try:
        with open(f"{file_name}.xml", "w") as file:
            file.write(pretty_xml_str)
    except OSError as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    # generate the guidewire model XML, print it, and save to a file named "guidewire_model.xml"
    try:
        guidewire_model = create_guidewire()
        if guidewire_model:
            print(guidewire_model)
            create_xml_file(guidewire_model, "guidewire_model")
        else:
            print("ERROR: Failed to create the guidewire model.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")