import subprocess

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
        install.run(self)


setup(
    name="cathsim",
    version="0.1.dev1",
    description="Your package description",
    packages=find_packages(),
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
    # Other setup parameters
)
