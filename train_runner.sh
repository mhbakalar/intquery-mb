#!/bin/bash
# Train model

CONFIG_FILE="$1"

CHR_NAME="chr$CHR_NAME"
COMMAND="python trainer.py fit --config $CONFIG_FILE"

echo "Executing command for $CHR_NAME(+):"
echo "$COMMAND"
eval "$COMMAND"


