from setuptools import setup, find_packages

extra_dev = [
    'pytest',
]

extra_utils = [
    'rich',
    'progressbar2',
    'progress',
    'tqdm',
    'opencv-python',
    'matplotlib',
    'seaborn',
]

extra_rl = [
    'imitation',
    'stable-baselines3==1.8.0',
    'torch',
]

extra = extra_dev + extra_rl


setup(
    name='cathsim',
    version='0.1.dev1',
    url='cathsim.github.io',
    author='Tudor Jianu',
    author_email='tudorjnu@gmail.com',
    packages=find_packages(
        where='src',
        include=[
            'cathsim',
            'rl',
            'human',
            'evaluation',
        ]
    ),
    install_requires=[
        'gym==0.21.*',
        'dm-control',
        'pyyaml',
        'mergedeep',
        *extra_rl,
        *extra_utils,
    ],
    extras_require={
        'dev': extra_dev,
        'rl': extra_rl,
        'all': extra,
    },
    entry_points={
        'console_scripts': [
            'run_env=cathsim.cathsim.env:run_env',
            'record_traj=human.utils:cmd_record_traj',
            'visualize_agent=rl.utils:cmd_visualize_agent',
        ],
    },
)
