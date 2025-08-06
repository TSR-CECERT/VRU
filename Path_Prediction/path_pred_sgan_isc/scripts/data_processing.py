# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 11:04:34 2024

@author: EdisonLee
"""

import pandas as pd
import os

# Define the base input directory
base_input_dir = r'E:\CE-CERT\ISC\Validation Data'  # The location of the 'train', 'test', 'val' folders

# Define the base output directory
base_output_dir = r'E:\CE-CERT\ISC\sgan-master\sgan-master\datasets\ISC'
output_dirs = ['train', 'test', 'val']  # Names of the new folders to be created

# Create the new output directories
for folder in output_dirs:
    os.makedirs(os.path.join(base_output_dir, folder), exist_ok=True)

# Function to format and align the text output
def format_row(row):
    # Adjust the width of each column to align the output
    return "{:<20} {:<30} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20}".format(*row)

# Process each folder (train, test, val)
for folder in output_dirs:
    input_folder = os.path.join(base_input_dir, folder)
    output_folder = os.path.join(base_output_dir, folder)

    # List all CSV files in the input directory
    csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]

    # Process each CSV file
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(input_folder, csv_file))
        
        # Replace NaN with 0
        df = df.fillna(0)

        # Prepare an empty list to store the transformed rows
        transformed_rows = []

        # Iterate over each row in the CSV
        for index, row in df.iterrows():
            timestamp = row['Time']  # Extract timestamp
            
            # Extract unique agent classes from the column headers
            agent_classes = set('_'.join(col.split('_')[:-1]) for col in df.columns if '_' in col)

            # Iterate over each agent class and collect its data
            for agent_class in agent_classes:
                # Prepare a row for each agent class
                transformed_row = [
                    timestamp,
                    agent_class,
                    row.get(f'{agent_class}_xctr', 0),
                    row.get(f'{agent_class}_yctr', 0),
                    row.get(f'{agent_class}_zctr', 0),
                    row.get(f'{agent_class}_xlen', 0),
                    row.get(f'{agent_class}_ylen', 0),
                    row.get(f'{agent_class}_zlen', 0)
                ]
                # Replace any 'nan' with 0 in the row
                transformed_row = [0 if x == 'nan' else x for x in transformed_row]

                # Append the row to the list of transformed rows
                transformed_rows.append(transformed_row)

        # Format each row for aligned text output
        formatted_rows = [format_row(row) for row in transformed_rows]

        # Save the formatted data to a new text file
        output_file = os.path.join(output_folder, csv_file.replace('.csv', '.txt'))
        with open(output_file, 'w') as f:
            f.write('\n'.join(formatted_rows))

print("Transformation complete. Files saved in the new 'train', 'test', and 'val' folders.")
