from dm_control import composer
from abc import ABC, abstractmethod

from functools import cached_property


class BaseModel(ABC, composer.Entity):
    @abstractmethod
    def _build(self):
        """Building method for the guidewire.

        Method to be implemented by subclasses. This method should build the
        guidewire model.

        Raises:
            NotImplementedError: Error raised if the method is not implemented

        Examples:
        >>> class MyGuidewire(BaseGuidewire):
        >>>     def _build(self):
        >>>         self._mjcf_root = mjcf.RootElement(model="guidewire")
        >>>         self._mjcf_root.worldbody.add("geom", type="sphere", size=[0.1])
        >>>         self._mjcf_root.actuator.add("velocity", name="my_actuator")
        >>>         self._mjcf_root.joint.add("slide", name="my_joint")
        >>>         self._mjcf_root.joint.add("hinge", name="my_hinge")

        Notes:
        It is a good idea to separate the steps of building the model into multiple
        private methods. For example:

        >>> class MyGuidewire(BaseGuidewire):
        >>>     def _build(self):
        >>>         self._mjcf_root = mjcf.RootElement(model="guidewire")
        >>>         self._set_defaults()
        >>>         self._set_bodies_and_joints()
        >>>         self._set_actuators()
        """
        raise NotImplementedError("Subclasses should implement this!")

    @property
    @abstractmethod
    def mjcf_model(self):
        raise NotImplementedError("Subclasses should implement this!")

    @cached_property
    def actuators(self):
        """Get the actuators of the guidewire."""
        return tuple(self.mjcf_model.find_all("actuator"))

    @cached_property
    def joints(self):
        return tuple(self.mjcf_model.find_all("joint"))

    @cached_property
    def bodies(self):
        return tuple(self.mjcf_model.find_all("body"))


class BaseGuidewire(BaseModel, ABC):
    """Base class for guidewires.

    This class is an abstract class that defines the interface for guidewires.
    It is not meant to be instantiated directly. Instead, it should be subclassed
    and the subclass should implement the _build method and the mjcf_model property.
    """


class BasePhantom(BaseModel, ABC):
    @property
    def sites(self) -> dict:
        """
        Gets the sites from the mesh. Useful for declaring navigation targets or areas of interest.
        """
        sites = self.mjcf_model.find_all("site")
        return {site.name: site.pos for site in sites}
