# cryptic_prediction

## Usage: train_runner.sh and predict_runner.sh

Training and prediction can be run using train_runner.sh and predict_runner.sh
train_runner.sh:
input: config file
predict_runner.sh:
inputs: ckpt_path config_file

Sample config files:
train_config.yaml
predict_config.yaml

## Usage: pure python

To train the model using the specified configuration, run the training script with the following command:

```bash
python trainer.py fit --config config.yaml
```

To test or predict using an existing checkpoint file:
```bash
python trainer.py test --ckpt_path ckpt_path.ckpt --config config.yaml
```

```bash
python trainer.py predict --ckpt_path ckpt_path.ckpt --config config.yaml
```

To predict on a whole genome fasta using the specified configuration, run the training script with the following command:

```bash
python trainer.py predict --ckpt_path ckpt_path.ckpt--config config.yaml
```

Replace `ckpt_path.yaml` and `config.yaml` with the path to your checkpoint and configuration file if it's named differently.

## Note

Ensure that you have the required dependencies installed. You can install them using the following command:

```bash
pip install requirements.txt
```

Feel free to customize the configuration file and adjust the parameters to suit your specific use case.
