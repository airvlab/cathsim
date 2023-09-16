import xml.etree.ElementTree as ET
from xml.dom import minidom

class GuidewireModel:
    """
    Represents a MuJoCo guidewire model and provides methods for creating and saving the XML representation.
    """
    
    # Class level constants
    SCALE = 1
    ELEVATE = 0.003  # Height elevation in meters
    INSERT = 0.01  # Insertion length in meters
    DENSITY = 6450  # Density of the material (nitinol) in SI unit (kg/cu.m)
    TOTAL_BODY_LENGTH = 0.5  # Total length of the guidewire body in meters
    N_BODIES = 30  # Number of individual pieces or sections
    BODY_DIAMETER = 0.0005 * SCALE  # Diameter of each body section in meters
    BODY_RADIUS = BODY_DIAMETER / 2  # Radius, which is half of the diameter
    BODY_LENGTH = TOTAL_BODY_LENGTH / N_BODIES * SCALE  # Length of each body section

    # Environment options representing properties of human blood
    ENV_OPTIONS = {
        "density": "1055",  # Density of human blood in kg/cu.m
        "viscosity": "0.0035",  # Viscosity of blood in Pa
        "o_margin": "0.02",
        "integrator": "implicitfast",  # Numerical integration method
        "collision": "all",  # Collision detection type
        "cone": "pyramidal",  # Shape of the contact friction cone
        "jacobian": "sparse",  # Type of Jacobian matrix used for constraints
        "solver": "Newton"  # Numerical solver type
    }

    @staticmethod
    def create_guidewire():
        """
        Create an XML representation for a MuJoCo guidewire model using the elasticity plugin.
        
        Constructs an XML tree that describes the guidewire model integrating the elasticity plugin to simulate 
        its behavior realistically. The model includes the aortic arch, compiler settings, world body properties, 
        the guidewire's core structure, and actuators.
        
        Returns:
            str: A prettified XML string representation of the guidewire model.
        """
        try:
            # Root element of the XML structure
            root = ET.Element("mujoco", model="Guidewire")

            # Extension for the MuJoCo plugin
            ET.SubElement(root, "extension").append(ET.Element("plugin", plugin="mujoco.elasticity.cable"))

            # Incorporate the aortic arch into the model
            ET.SubElement(root, "include", file="assets/phantom2.xml")

            # Compiler and memory specifications
            ET.SubElement(root, "compiler", autolimits="false", meshdir="assets/meshes")
            ET.SubElement(root, "size", memory="1G")

            # Define the environment options (e.g., blood properties)
            ET.SubElement(root, "option", attrib=GuidewireModel.ENV_OPTIONS)

            # World body: contains all the physical entities in the simulation
            worldbody = ET.SubElement(root, "worldbody")

            # Define lighting for visualization
            ET.SubElement(worldbody, "light", diffuse=".4 .4 .4", specular="0.1 0.1 0.1", pos="0 0 2", dir="0 0 -1", castshadow="false")
            ET.SubElement(worldbody, "light", directional="true", diffuse=".8 .8 .8", specular="0.2 0.2 0.2", pos="0 0 4", dir="0 0 -1")

            # Core structure and joints of the guidewire
            core = ET.SubElement(worldbody, "body", name="core", pos=f"0 {GuidewireModel.INSERT - GuidewireModel.TOTAL_BODY_LENGTH} {GuidewireModel.ELEVATE + GuidewireModel.BODY_RADIUS}", euler="0 0 90")
            ET.SubElement(core, "joint", name="slider", type="slide", axis="1 0 0", stiffness="0", damping="0")
            ET.SubElement(core, "joint", name="rotator", type="hinge", axis="1 0 0", stiffness="0", damping="5", armature="0.1")

            # Define the cable composite representing the body of the guidewire
            composite = ET.SubElement(core, "composite", type="cable", curve="s", count=f"{GuidewireModel.N_BODIES + 1} 1 1", size=f"{GuidewireModel.TOTAL_BODY_LENGTH}", offset=f"{GuidewireModel.BODY_RADIUS} 0 0", initial="none")
            ET.SubElement(composite, "geom", type="capsule", size=f"{GuidewireModel.BODY_RADIUS}", rgba="1 1 1 1", condim="1", density=f"{GuidewireModel.DENSITY}", margin="0.02")
            ET.SubElement(composite, "joint", kind="main", damping="0.1", stiffness="0.01", margin="0.02")
            composite_plugin = ET.SubElement(composite, "plugin", plugin="mujoco.elasticity.cable")
            ET.SubElement(composite_plugin, "config", key="twist", value="1e6")
            ET.SubElement(composite_plugin, "config", key="bend", value="1e9")
            ET.SubElement(composite_plugin, "config", key="vmax", value="0.0005")

            # Actuators for controlling the guidewire motion
            actuator = ET.SubElement(root, "actuator")
            ET.SubElement(actuator, "velocity", name="slider_actuator", joint="slider")
            ET.SubElement(actuator, "general", name="rotator_actuator", joint="rotator", biastype="none", dynprm="1 0 0", gainprm="40 0 0", biasprm="0 40 0")

            # Convert the XML tree to a string
            xml_str = ET.tostring(root, encoding="unicode")
            return GuidewireModel.prettify_xml(xml_str)
            
        except ET.ParseError as e:
            print(f"Error during XML generation: {e}")
            return None
        
    @staticmethod
    def prettify_xml(xml_str):
        """
        Convert an XML string into a prettified version for better readability.

        Parameters:
            xml_str (str): The XML string to be prettified.

        Returns:
            str: A prettified XML string representation.
        """
        try:
            dom = minidom.parseString(xml_str)
            return dom.toprettyxml()
        except Exception as e:
            print(f"Error during XML prettification: {e}")
            return None

    @staticmethod
    def create_xml_file(pretty_xml_str, file_name):
        """
        Save the provided XML string to a file.

        Parameters:
            pretty_xml_str (str): The prettified XML string.
            file_name (str): Name of the file to save the XML string to (without extension).

        Returns:
            None
        """
        try:
            with open(f"{file_name}.xml", "w") as file:
                file.write(pretty_xml_str)
        except OSError as e:
            print(f"Error writing to file: {e}")

if __name__ == "__main__":
    try:
        # Generate the XML representation for the guidewire.
        guidewire_model = GuidewireModel.create_guidewire()
        
        # If successfully generated, print and save it.
        if guidewire_model:
            print(guidewire_model)
            GuidewireModel.create_xml_file(guidewire_model, "guidewire_model")
        else:
            print("ERROR: Failed to create the guidewire model.")
    except Exception as e:
        print(f"ERROR: {e}")