#!/bin/bash
# Predict positive strand: predict_runner.sh +
# Predict negative strand: predict_runner.sh -

# Check if the number of arguments is correct
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <STRAND>"
    exit 1
fi

# Get the value of the STRAND argument
STRAND="$1"

CKPT_PATH="lightning_logs/version_0/checkpoints/epoch\=19-step\=10060.ckpt"
CONFIG_FILE="predict_config.yaml"

for CHR_NAME in {1..4}; do
    CHR_NAME="chr$CHR_NAME"
    COMMAND="python trainer.py predict --ckpt_path $CKPT_PATH --config $CONFIG_FILE --data.chr_name $CHR_NAME --trainer.callbacks.init_args.chr_name $CHR_NAME --data.init_args.strand $STRAND --trainer.callbacks.init_args.strand $STRAND"

    echo "Executing command for $CHR_NAME($STRAND):"
    echo "$COMMAND"

    # Uncomment the line below to execute the command
    eval "$COMMAND"

    echo "--------------------------------------"
done

