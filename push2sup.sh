#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if the user provided a NetID
if [ -z "$1" ]; then
    echo "No NetID provided. Defaulting to 'dps2000'.\n"
    NETID="dps2000"
else
    # Assign the input NetID to a variable
    NETID=$1
fi

# Define the name of the temporary zip file
ZIP_NAME="code_archive.zip.tar.gz"

# create a json file called commits.json and write the date and the net id to it
echo "{\"date\": \"$(date)\", \"netid\": \"$NETID\"}" > commits.json

# Find all .py and .json files, preserving their folder structure, and zip them into an archive
echo "Creating zip archive from 'py', and 'json' files..."
# find . -type f \( -name "*.py" -o -name "*.json" -o -name "*.txt" \) | tar -czf "$ZIP_NAME" -T -
find . \( -path "./wandb" -o -path "./apis" \) -prune -o -type f \( -name "*.py" -o -name "*.json" \) -print | tar -czf "$ZIP_NAME" -T -


# Check if the zip command was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to create the zip archive."
    exit 1
fi

# Define the remote path
REMOTE_PATH="/home/$NETID/groups/grp_fold/$ZIP_NAME"

# Transfer the zip file to the remote host
echo "Transferring zip file to remote server..."
scp "$ZIP_NAME" "$NETID@ssh.rc.byu.edu:$REMOTE_PATH"

# Check if the scp command was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to transfer the zip file to the remote server."
    exit 1
fi

# Clean up by removing the local zip file
echo "Cleaning up the local zip file..."
rm "$ZIP_NAME"
rm commits.json

# Check if the remove command was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to delete the local zip file."
    exit 1
fi

echo "Zipped folder has been copied successfully to $REMOTE_PATH."
echo "Please run the corresponding 'pull' script on the supercomputer to finish the process."
