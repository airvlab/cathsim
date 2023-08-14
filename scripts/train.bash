#/bin/bash

# use a loop to train all the models
echo "-----Training-----"
configs=(full internal_pixels internal pixels pixels_mask)
targets=(lcca)
phantom=phantom3
n_runs=6
n_timesteps=600000
for config in ${configs[@]}; do
	for target in ${targets[@]}; do
		echo "Training $config on $phantom - $target - run $run"
		train --config $config --target $target --phantom $phantom --n-runs $n_runs --n-timesteps $n_timesteps
	done
done
