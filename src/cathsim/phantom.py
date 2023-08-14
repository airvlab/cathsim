from pathlib import Path

from dm_control import mjcf
from dm_control import composer

from cathsim.utils import normalize_rgba, get_env_config

env_config = get_env_config()
phantom_config = env_config["phantom"]
phantom_config["rgba"] = normalize_rgba(phantom_config["rgba"])


class Phantom(composer.Entity):
    def _build(
        self, phantom_xml: str = "phantom3.xml", assets_dir: Path = None, **kwargs
    ):
        """

        :param phantom_xml: str:  (Default value = "phantom3.xml")
        :param assets_dir: Path:  (Default value = None)

        """
        self.rgba = phantom_config["rgba"]
        self.scale = [phantom_config["scale"] for i in range(3)]

        path = Path(__file__).parent
        model_dir = path / "assets"
        phantom_xml_path = (model_dir / phantom_xml).as_posix()
        self._mjcf_root = mjcf.from_file(
            phantom_xml_path, False, model_dir.as_posix(), **kwargs
        )
        self._mjcf_root.default.geom.set_attributes(
            margin=0.004,
            group=0,
            condim=phantom_config["condim"],
        )
        self._mjcf_root.default.site.set_attributes(
            rgba=[0, 0, 0, 0],
        )
        self._mjcf_root.default.site.set_attributes(
            type="sphere",
            size=[0.002],
            rgba=[0.8, 0.8, 0.8, 0],
        )

        self.set_scale(scale=self.scale)
        self.set_rgba(rgba=self.rgba)
        self.phantom_visual = (
            model_dir / f'meshes/{phantom_xml.split(".")[0]}/visual.stl'
        )
        self.simplified = (
            model_dir / f'meshes/{phantom_xml.split(".")[0]}/simplified.stl'
        )

    def set_rgba(self, rgba: list) -> None:
        """Set the RGBA of the phantom.

        :param rgba: list: [r, g, b, a]

        """
        self.rgba = rgba
        self._mjcf_root.find("geom", "visual").rgba = self.rgba
        collision_rgba = rgba.copy()
        collision_rgba[-1] = 0
        self._mjcf_root.default.geom.set_attributes(rgba=collision_rgba)

    def set_hulls_alpha(self, alpha: float) -> None:
        """Set the transparency of the convex hulls

        :param alpha: float:

        """
        self.rgba[-1] = alpha
        self._mjcf_root.default.geom.set_attributes(rgba=self.rgba)

    def set_scale(self, scale: tuple) -> None:
        """Set the scale of the mesh.

        :param scale: tuple: [x_scale, y_scale, z_scale]

        """
        self._mjcf_root.default.mesh.set_attributes(scale=scale)
        self._mjcf_root.find("mesh", "visual").scale = [x * 1.005 for x in scale]

    def get_scale(self) -> list:
        return self.scale

    def get_rgba(self) -> list:
        return self.rgba

    @property
    def sites(self) -> dict:
        """Gets the sites from the mesh. Useful for declaring navigation targets or areas of interest."""
        sites = self._mjcf_root.find_all("site")
        return {site.name: site.pos for site in sites}

    @property
    def mjcf_model(self):
        return self._mjcf_root


if __name__ == "__main__":
    phantom = Phantom("phantom3.xml")
    print(phantom.sites())
