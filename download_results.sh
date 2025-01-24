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

# Check if the user provided the desired name
if [ -z "$2" ]; then
    echo "No desired name provided. Defaulting to 'results'.\n"
    ZIPNAME="results"
else
    # Assign the input name to a variable
    ZIPNAME=$2
fi

REMOTE_PATH="/home/$NETID/groups/grp_fold/auto_scripts/${ZIPNAME}_archive.zip.tar.gz"

scp "$NETID@ssh.rc.byu.edu:$REMOTE_PATH" data/

# delete old remote file
read -p "Do you want to delete the remote file? (y/n) " -n 1 -r
echo    # Move to a new line after user input for better formatting.
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting the remote file..."
    ssh "$NETID@ssh.rc.byu.edu" rm "$REMOTE_PATH"
    if [[ $? -eq 0 ]]; then
        echo "File successfully deleted."
    else
        echo "Failed to delete the file. Check the path and permissions."
    fi
else
    echo "Okay, file will not be deleted."
fi

# unzip the file
echo "Unzipping the file..."
tar -xzf data/${ZIPNAME}_archive.zip.tar.gz -C data/

rm data/${ZIPNAME}_archive.zip.tar.gz

echo "Zip file has been successfully downloaded to the 'data' directory and unzipped."


