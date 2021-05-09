#!/usr/bin/env bash

set -e  # Stop on failure
set -x  # Show lines during run

echo "Simulating a pipeline until failure or completion"

pylint ./commons/ --rcfile ./pylintrc
pylint ./training/ --rcfile ./pylintrc

# add pytest here later

echo "Pipeline simulation ran successfully"
