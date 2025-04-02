#!/bin/bash

# Define the source and destination directories
SOURCE_DIR=~/Documents/NN_Paper/ventil_lstm/Experiment_Meassurements   # Change this if needed
DEST_DIR=~/Documents/NN_Paper/Code_4_paper  # Change this if needed

# List of files to copy (add more as needed)
FILES_TO_COPY=(
    "meas_remote_training_lstm _not_pretrained.py"
    "meas_NN_classes.py"
    "meas_load_data.py"
    "meas_get_data.py"
    "meas_dataloader_fs.py"
    "meas_test_func_fs.py"
)

# Loop through each file and copy it
for FILE in "${FILES_TO_COPY[@]}"; do
    # Ensure the file exists before copying
    if [ -f "$SOURCE_DIR/$FILE" ]; then
        # Create subdirectories if needed
        mkdir -p "$DEST_DIR/$(dirname "$FILE")"
        
        # Copy the file
        cp "$SOURCE_DIR/$FILE" "$DEST_DIR/$FILE"
        echo "Copied: $FILE"
    else
        echo "Warning: $FILE not found in $SOURCE_DIR"
    fi
done

echo "File copy process completed!"
