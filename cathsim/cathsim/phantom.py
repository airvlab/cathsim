from pathlib import Path

from dm_control import mjcf
from dm_control import composer

from cathsim.cathsim.common import env_config, normalize_rgba

phantom_config = env_config['phantom']
phantom_config['rgba'] = normalize_rgba(phantom_config['rgba'])


class Phantom(composer.Entity):
    def _build(self, phantom_xml: str = 'phantom3.xml', assets_dir: Path = None, **kwargs):
        self.rgba = phantom_config['rgba']
        self.scale = [phantom_config['scale'] for i in range(3)]

        path = Path(__file__).parent
        model_dir = path / 'assets'
        phantom_xml_path = (model_dir / phantom_xml).as_posix()
        self._mjcf_root = mjcf.from_file(phantom_xml_path, False,
                                         model_dir.as_posix(), **kwargs)
        self._mjcf_root.default.geom.set_attributes(
            margin=0.004,
            group=0,
            condim=phantom_config['condim'],
        )
        self._mjcf_root.default.site.set_attributes(
            rgba=[0, 0, 0, 0],
        )
        self._mjcf_root.default.site.set_attributes(
            type='sphere',
            size=[0.002],
            rgba=[0.8, 0.8, 0.8, 0],
        )

        self.set_scale(scale=self.scale)
        self.set_rgba(rgba=self.rgba)

    def set_rgba(self, rgba: list) -> None:
        self.rgba = rgba
        self._mjcf_root.find('geom', 'visual').rgba = self.rgba
        collision_rgba = rgba.copy()
        collision_rgba[-1] = 0
        self._mjcf_root.default.geom.set_attributes(rgba=collision_rgba)

    def set_hulls_alpha(self, alpha) -> None:
        self.rgba[-1] = alpha
        self._mjcf_root.default.geom.set_attributes(rgba=self.rgba)

    def set_scale(self, scale: list) -> None:
        self._mjcf_root.default.mesh.set_attributes(scale=scale)
        self._mjcf_root.find('mesh', 'visual').scale = [x * 1.005 for x in scale]

    def get_scale(self) -> list:
        return self.scale

    def get_rgba(self) -> list:
        return self.rgba

    @property
    def sites(self):
        sites = self._mjcf_root.find_all('site')
        return {site.name: site.pos for site in sites}

    @ property
    def mjcf_model(self):
        return self._mjcf_root

class PhantomFluid(composer.Entity):
    def _build(self, phantom_xml: str = 'phantom3_fluid_cathsim.xml', assets_dir: Path = None, **kwargs):
        self.rgba = phantom_config['rgba']
        self.scale = [phantom_config['scale'] for i in range(3)]

        path = Path(__file__).parent
        model_dir = path / 'assets'
        phantom_xml_path = (model_dir / phantom_xml).as_posix()
        self._mjcf_root = mjcf.from_file(phantom_xml_path, False,
                                         model_dir.as_posix(), **kwargs)
        self._mjcf_root.default.geom.set_attributes(
            margin=0.004,
            group=0,
            condim=phantom_config['condim'],
        )
        self._mjcf_root.default.site.set_attributes(
            rgba=[0, 0, 0, 0],
        )
        self._mjcf_root.default.site.set_attributes(
            type='sphere',
            size=[0.002],
            rgba=[0.8, 0.8, 0.8, 0],
        )

        self.set_scale(scale=self.scale)
        self.set_rgba(rgba=self.rgba)

    def set_rgba(self, rgba: list) -> None:
        self.rgba = rgba
        self._mjcf_root.find('geom', 'visual').rgba = self.rgba
        collision_rgba = rgba.copy()
        collision_rgba[-1] = 0
        self._mjcf_root.default.geom.set_attributes(rgba=collision_rgba)

    def set_hulls_alpha(self, alpha) -> None:
        self.rgba[-1] = alpha
        self._mjcf_root.default.geom.set_attributes(rgba=self.rgba)

    def set_scale(self, scale: list) -> None:
        self._mjcf_root.default.mesh.set_attributes(scale=scale)
        # self._mjcf_root.find('mesh', 'visual').scale = [x * 4 for x in scale]

    def get_scale(self) -> list:
        return self.scale

    def get_rgba(self) -> list:
        return self.rgba

    @property
    def sites(self):
        sites = self._mjcf_root.find_all('site')
        return {site.name: site.pos for site in sites}

    @ property
    def mjcf_model(self):
        return self._mjcf_root


if __name__ == "__main__":
    phantom = Phantom("phantom3.xml")
    print(phantom.sites())
