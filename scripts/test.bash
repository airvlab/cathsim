#/bin/bash

# use a loop to train all the models
echo "-----Test-----"
python rl/train.py --config test --target bca  --phantom phantom3 --n-runs 1 --n-timesteps 2
echo "Done"
