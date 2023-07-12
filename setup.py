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
    packages=find_packages(
        where="src",
        include=[
            "cathsim",
            "rl",
            "human",
            "evaluation",
        ],
    ),
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
    ],
    extras_require={
        "dev": extra_dev,
    },
    entry_points={
        "console_scripts": [
            "run_env=cathsim.cathsim.env:run_env",
            "record_traj=human.utils:cmd_record_traj",
            "visualize_agent=rl.utils:cmd_visualize_agent",
        ],
    },
)
