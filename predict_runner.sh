#!/bin/bash
# Usage: ./predict_runner.sh lightning_logs/version_0/checkpoints/epoch\=19-step\=10060.ckpt predict_config.yaml
# Full commmand executed:
# python trainer.py predict --ckpt_path lightning_logs/version_0/checkpoints/epoch\=19-step\=10060.ckpt --config predict_config.yaml --data.chr_name chr21

CKPT_PATH="$1"
PREDICT_CONFIG="$2"

POS="+"
NEG="-"

for CHR in {22..22}; do
    CHR_NAME="chr$CHR"
    COMMAND_POS="python trainer.py predict --ckpt_path $CKPT_PATH --config $PREDICT_CONFIG --data.chr_name $CHR_NAME --data.strand $POS"
    COMMAND_NEG="python trainer.py predict --ckpt_path $CKPT_PATH --config $PREDICT_CONFIG --data.chr_name $CHR_NAME --data.strand $NEG"

    echo "Executing command for $CHR_NAME($POS):"
    echo "$COMMAND_POS"
    eval "$COMMAND_POS"

    echo "Executing command for $CHR_NAME($NEG):"
    echo "$COMMAND_NEG"
    eval "$COMMAND_NEG"

    echo "--------------------------------------"
done

