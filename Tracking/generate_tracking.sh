#!/bin/bash

# Define input and output directories
input_dir="./detections"
output_dir="./tracking"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Count the total number of .csv files to process
total_files=$(ls "$input_dir"/*.csv | wc -l)
current_file=0

# Check if the --show_live_plot argument is passed to the bash script
show_live_plot=""
if [[ "$*" == *"--show_live_plot"* ]]; then
  show_live_plot="--show_live_plot"
fi

# Function to display a progress bar
show_progress() {
  local progress=$1
  local total=$2
  local width=50  # Width of the progress bar
  local percent=$((progress * 100 / total))
  local filled=$((progress * width / total))
  local empty=$((width - filled))

  printf "\r["
  printf "%0.s#" $(seq 1 $filled)
  printf "%0.s " $(seq 1 $empty)
  printf "] %d%% (%d/%d)" "$percent" "$progress" "$total"
}

# Loop over all .csv files in the input directory
for input_file in "$input_dir"/*.csv; do
  # Extract the base name (e.g., 'Detection_Classification_Localization_Submission_Run_48' from 'Detection_Classification_Localization_Submission_Run_48.csv')
  base_name=$(basename "$input_file" .csv)

  # Define the output file path with .csv extension
  output_file="$output_dir/$base_name.csv"

  # Run the Python script with the input and output paths and merged tracks
  python3 kf_cv_tracker_sample_run.py --input "$input_file" --output "$output_file" $show_live_plot --merge_tracks
  
  # Run the Python script with the input and output paths, without merged tracks
  #python3 kf_cv_tracker_sample_run.py --input "$input_file" --output "$output_file" $show_live_plot

  # Increment the current file count
  current_file=$((current_file + 1))

  # Show progress bar
  show_progress "$current_file" "$total_files"
done

# Print a new line after the progress bar completes
echo

