#!/bin/bash

CKPT_PATH="lightning_logs/version_0/checkpoints/epoch\=19-step\=10060.ckpt"
CONFIG_FILE="predict_config.yaml"

for CHR_NAME in {18..22}; do
    CHR_NAME="chr$CHR_NAME"
    COMMAND="python trainer.py predict --ckpt_path $CKPT_PATH --config $CONFIG_FILE --data.chr_name $CHR_NAME --trainer.callbacks.init_args.chr_name $CHR_NAME"

    echo "Executing command for $CHR_NAME:"
    echo "$COMMAND"

    # Uncomment the line below to execute the command
    eval "$COMMAND"

    echo "--------------------------------------"
done

