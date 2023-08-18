from setuptools import setup, find_packages

extra_dev = [
    "pytest",
]

setup(
    name="cathsim",
    version="0.1.dev1",
    url="cathsim.github.io",
    author="Tudor Jianu",
    author_email="tudorjnu@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "rl": ["config/*.yaml"],
    },
    setup_requires=[
        "setuptools==58.0.0",
    ],
    install_requires=[
        "gym==0.21.*",
        "dm-control",
        "pyyaml",
        "opencv-python",
        "matplotlib",
        "stable-baselines3==1.8.0",
        "torch",
        "imitation",
        "tqdm",
        "rich",
        "mergedeep",
        "progressbar2",
        "progress",
        "trimesh",
        "rtree",
        "toolz",
    ],
    extras_require={
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            "sphinxcontrib.spelling",
            "sphinx-autodoc-typehints",
            "sphinx_copybutton",
        ],
        "dev": extra_dev,
    },
    entry_points={
        "console_scripts": [
            "run_env=utils.console:cmd_run_env",
            "visualize_agent=utils.console:cmd_visualize_agent",
            "train=utils.console:cmd_train",
        ],
    },
)
