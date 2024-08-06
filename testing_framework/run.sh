#!/bin/bash

# Run ilp.py in the background
python3 ilp.py &

# Add a 2-second delay before starting openshift_client.py to generate the initial user configuration
sleep 2

# Run openshift_client.py in the background
python3 openshift_client.py &

# Wait for all background jobs to finish
wait

echo "Testing done."
