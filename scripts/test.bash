#/bin/bash

# use a loop to train all the models
echo "-----Test-----"
configs=(test_full) # internal_pixels internal pixels pixels_mask)
targets=(bca lcca)
phantoms=(phantom3)
trial=test
n_runs=2
n_timesteps=2000
for config in ${configs[@]}; do
	for target in ${targets[@]}; do
		for phantom in ${phantoms[@]}; do
			echo "Training $config on $phantom - $target"
			train --config $config --target $target --phantom $phantom --n-runs $n_runs --n-timesteps $n_timesteps --trial-name trial --base-path ./data/test-main -e
		done
	done
done
