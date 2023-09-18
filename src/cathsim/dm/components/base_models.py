from dm_control import composer
from abc import ABC, abstractmethod


class BaseGuidewire(composer.Entity, ABC):

    @abstractmethod
    def _build():
        """Building method for the guidewire.

        Method to be implemented by subclasses. This method should build the
        guidewire model.

        Raises:
            NotImplementedError: Error raised if the method is not implemented

        Examples:
        >>> class MyGuidewire(BaseGuidewire):
        >>>     def _build(self):
        >>>     self._mjcf_root = mjcf.RootElement(model="guidewire")
        >>>     self._mjcf_root.worldbody.add("geom", type="sphere", size=[0.1])
        >>>     self._mjcf_root.actuator.add("velocity", name="my_actuator")
        >>>     self._mjcf_root.joint.add("slide", name="my_joint")
        >>>     self._mjcf_root.joint.add("hinge", name="my_hinge")

        Notes:
        It is a good idea to separate the steps of building th emodel into multiple
        private methods. For example:

        >>> class MyGuidewire(BaseGuidewire):
        >>>     def _build(self):
        >>>         self._mjcf_root = mjcf.RootElement(model="guidewire")
        >>>         self._set_defaults(self)
        >>>         self._set_bodies_and_joints(self)
        >>>         self._set_actuators(self)

        """
        raise NotImplementedError("Subclasses should implement this!")

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def actuators(self):
        """Get the actuators of the guidewire."""
        return tuple(self._mjcf_root.find_all("actuator"))

    @property
    def joints(self):
        return tuple(self._mjcf_root.find_all("joint"))
