#!/bin/bash

# Get the current directory of the script
current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set the path to the folder containing Python scripts (one folder up)
script_folder="$current_dir/../WES2024/Generate"

# Log file path
log_file="$current_dir/../log_generate.txt"

# Check if the folder exists
if [ -d "$script_folder" ]; then
    echo "Running Python scripts in $script_folder"

    # Create or truncate the log file
    > "$log_file"

    # Function to run a Python script and display progress
    run_python_script() {
        python_script="$1"
        script_name=$(basename "$python_script" .py)
        echo -n "Running $script_name... "
        { poetry run ipython "$python_script" >> "$log_file" 2>&1 && echo -e "\e[32m[OK]\e[0m"; } || echo -e "\e[31m[FAILED]\e[0m"
    }

    # Iterate through all Python files in the folder and execute them
    for script_file in "$script_folder"/*.py; do
        if [ -f "$script_file" ]; then
            run_python_script "$script_file"
        fi
    done

    echo "Finished running Python scripts. Check the log file for details: $log_file"
else
    echo "Error: The specified folder does not exist."
fi