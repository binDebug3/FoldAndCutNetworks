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

REMOTE_PATH="/home/$NETID/groups/grp_fold/auto_scripts/results_archive.zip.tar.gz"

scp "$NETID@ssh.rc.byu.edu:$REMOTE_PATH" data/

# unzip the file
tar -xzf data/results_archive.zip.tar.gz -C data/
rm data/results_archive.zip.tar.gz

echo "Zip file has been successfully downloaded to the 'data' directory and unzipped."


