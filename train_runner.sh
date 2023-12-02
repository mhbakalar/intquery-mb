#!/bin/bash
# Train model

CONFIG_FILE="$1"

COMMAND="python trainer.py fit --config $CONFIG_FILE"

echo "$COMMAND"
eval "$COMMAND"


