#!/bin/bash
# Train model
# Usage: ./train_runner.sh train_config.yaml

CONFIG_FILE="$1"

COMMAND="python trainer.py fit --config $CONFIG_FILE"

echo "$COMMAND"
eval "$COMMAND"


