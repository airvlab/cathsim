from pathlib import Path

from dm_control import mjcf
from dm_control import composer

from cathsim.dm.utils import normalize_rgba, get_env_config

phantom_config = get_env_config("phantom")
phantom_default = phantom_config["default"]


class Phantom(composer.Entity):
    def _build(
        self, phantom_xml: str = "phantom3.xml", assets_dir: Path = None, **kwargs
    ):
        """
        Build Phantom3. xml file and set default values.

        Args:
            phantom_xml: Name of the XML file to use
            assets_dir: Directory where assets are saved
        """

        self.rgba = normalize_rgba(phantom_default["geom"]["rgba"])
        self.scale = [phantom_config["scale"] for i in range(3)]

        path = Path(__file__).parent
        model_dir = path / "phantom_assets"
        phantom_xml_path = (model_dir / phantom_xml).as_posix()
        self._mjcf_root = mjcf.from_file(
            phantom_xml_path, False, model_dir.as_posix(), **kwargs
        )
        self._set_defaults()

        self.set_scale(scale=self.scale)
        self.set_rgba(rgba=self.rgba)
        self.phantom_visual = (
            model_dir / f'meshes/{phantom_xml.split(".")[0]}/visual.stl'
        )
        self.simplified = (
            model_dir / f'meshes/{phantom_xml.split(".")[0]}/simplified.stl'
        )

    def _set_defaults(self):
        """Sets the default values for the Phantom3."""
        self._mjcf_root.default.geom.set_attributes(
            **phantom_default["geom"],
        )
        self._mjcf_root.default.site.set_attributes(
            **phantom_default["site"],
        )

    def set_rgba(self, rgba: list):
        """Sets the RGBA values for the Phantom3.

        Used to change the color of the Phantom3. This can be used for domain randomization.

        Args:
            rgba (list): List of RGBA values (normalized to 1.0)
        """
        self.rgba = rgba
        self._mjcf_root.find("geom", "visual").rgba = self.rgba
        collision_rgba = rgba.copy()
        collision_rgba[-1] = 0
        self._mjcf_root.default.geom.set_attributes(rgba=collision_rgba)

    def set_hulls_alpha(self, alpha: float):
        """Sets the alpha value for the hulls.

        Usefull for debugging and visualization.

        Args:
            alpha (float): Alpha value to set
        """
        self.rgba[-1] = alpha
        self._mjcf_root.default.geom.set_attributes(rgba=self.rgba)

    def set_scale(self, scale: tuple):
        """Changes the scale of the phantom.


        Args:
            scale (tuple): The scale to set
        """
        self._mjcf_root.default.mesh.set_attributes(scale=scale)
        self._mjcf_root.find("mesh", "visual").scale = [x * 1.005 for x in scale]

    def get_scale(self) -> list:
        return self.scale

    def get_rgba(self) -> list:
        return self.rgba

    @property
    def sites(self) -> dict:
        """
        Gets the sites from the mesh. Useful for declaring navigation targets or areas of interest.
        """
        sites = self._mjcf_root.find_all("site")
        return {site.name: site.pos for site in sites}

    @property
    def mjcf_model(self):
        return self._mjcf_root


if __name__ == "__main__":
    phantom = Phantom("phantom3.xml")
    print(phantom.sites())
