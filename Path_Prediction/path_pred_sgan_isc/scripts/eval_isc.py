import os
import pandas as pd
import numpy as np

# Paths for predicted and ground truth
predicted_folder = r"E:\CE-CERT\Intersection Safety Challenge\Path Prediction\ISC\sgan-master\sgan-master\datasets\ISC\predicted_val"
ground_truth_folder = r"E:\CE-CERT\Intersection Safety Challenge\Path Prediction\ISC\sgan-master\sgan-master\datasets\ISC\processed_val"

# Function to determine class based on subclass
def get_class_from_subclass(subclass):
    if "Vehicle" in subclass:
        return "Vehicle"
    elif "VRU" in subclass:
        return "VRU"
    else:
        return "Unknown"

# Function to calculate Weighted Average Displacement Error (ADE)
def calculate_weighted_ade(ground_truth_df, predicted_df):
    ade_sum = 0
    count = 0

    for timestamp in predicted_df["Timestamps"].unique():
        if timestamp not in ground_truth_df["Timestamps"].unique():
            continue  # Skip timestamps that are not in the ground truth data
        
        gt_data = ground_truth_df[ground_truth_df["Timestamps"] == timestamp]
        pred_data = predicted_df[predicted_df["Timestamps"] == timestamp]
        
        matched_count = 0
        for _, pred_row in pred_data.iterrows():
            subclass = pred_row["subclass"]
            gt_row = gt_data[gt_data["subclass"] == subclass]
            if not gt_row.empty:
                matched_count += 1
                # Using L2 distance as per Equation 7
                displacement_error = np.sqrt((gt_row["x_center"].values[0] - pred_row["x_center"]) ** 2 +
                                             (gt_row["y_center"].values[0] - pred_row["y_center"]) ** 2 +
                                             (gt_row["z_center"].values[0] - pred_row["z_center"]) ** 2)
                # Use the confidence score directly as weight (w_i,c)
                weight = pred_row["confidence"] if "confidence" in pred_row else 1.0
                
                ade_sum += weight * displacement_error
                count += 1
        print(f"Timestamp {timestamp}: Matched predictions: {matched_count}")
    return ade_sum / count if count > 0 else 0

# Function to calculate unweighted ADE (for comparison)
def calculate_unweighted_ade(ground_truth_df, predicted_df):
    ade_sum = 0
    count = 0

    for timestamp in predicted_df["Timestamps"].unique():
        if timestamp not in ground_truth_df["Timestamps"].unique():
            continue  # Skip timestamps that are not in the ground truth data
        
        gt_data = ground_truth_df[ground_truth_df["Timestamps"] == timestamp]
        pred_data = predicted_df[predicted_df["Timestamps"] == timestamp]
        
        for _, pred_row in pred_data.iterrows():
            subclass = pred_row["subclass"]
            gt_row = gt_data[gt_data["subclass"] == subclass]
            if not gt_row.empty:
                # Using L2 distance as per Equation 7
                displacement_error = np.sqrt((gt_row["x_center"].values[0] - pred_row["x_center"]) ** 2 +
                                             (gt_row["y_center"].values[0] - pred_row["y_center"]) ** 2 +
                                             (gt_row["z_center"].values[0] - pred_row["z_center"]) ** 2)
                ade_sum += displacement_error
                count += 1
    return ade_sum / count if count > 0 else 0

# Function to calculate ADE based on class
def calculate_class_ade(ground_truth_df, predicted_df, target_class):
    filtered_gt = ground_truth_df[ground_truth_df["class"] == target_class]
    filtered_pred = predicted_df[predicted_df["class"] == target_class]
    return calculate_weighted_ade(filtered_gt, filtered_pred)

# Variables to store cumulative metrics for overall calculations
overall_weighted_ade_sum = 0
overall_vehicle_ade_sum = 0
overall_vru_ade_sum = 0
unweighted_ade_sum = 0
total_runs = 0

overall_count = 0
vehicle_count = 0
vru_count = 0

# Iterate through predicted files and corresponding ground truth files
for predicted_filename in os.listdir(predicted_folder):
    if predicted_filename.startswith("Path_Prediction_Submission_Run_") and predicted_filename.endswith(".csv"):
        run_number = predicted_filename.split("_")[-1].replace(".csv", "")
        ground_truth_filename = f"{run_number}_val.csv"
        
        predicted_path = os.path.join(predicted_folder, predicted_filename)
        ground_truth_path = os.path.join(ground_truth_folder, ground_truth_filename)
        
        if os.path.exists(ground_truth_path):
            predicted_df = pd.read_csv(predicted_path)
            ground_truth_df = pd.read_csv(ground_truth_path)
            
            # Filter predicted data to only include timestamps present in the ground truth
            valid_timestamps = ground_truth_df["Timestamps"].unique()
            predicted_df = predicted_df[predicted_df["Timestamps"].isin(valid_timestamps)]
            
            # Check for accidental overlap between predicted and ground truth
            if predicted_df.equals(ground_truth_df):
                print(f"Warning: Predicted and ground truth data are identical for run {run_number}. Skipping this run.")
                continue
            
            # Add a new column to predicted dataframe for class
            predicted_df["class"] = predicted_df["subclass"].apply(get_class_from_subclass)
            ground_truth_df["class"] = ground_truth_df["subclass"].apply(get_class_from_subclass)
            
            # Calculate Weighted ADE as per Equation 7
            weighted_ade = calculate_weighted_ade(ground_truth_df, predicted_df)
            overall_weighted_ade_sum += weighted_ade
            overall_count += 1

            # Calculate Unweighted ADE (for comparison)
            unweighted_ade = calculate_unweighted_ade(ground_truth_df, predicted_df)
            unweighted_ade_sum += unweighted_ade
            
            # Calculate metrics for each class (Vehicle and VRU)
            vehicle_weighted_ade = calculate_class_ade(ground_truth_df, predicted_df, "Vehicle")
            vru_weighted_ade = calculate_class_ade(ground_truth_df, predicted_df, "VRU")
            
            if vehicle_weighted_ade > 0:
                overall_vehicle_ade_sum += vehicle_weighted_ade
                vehicle_count += 1
            if vru_weighted_ade > 0:
                overall_vru_ade_sum += vru_weighted_ade
                vru_count += 1
            
            total_runs += 1

# Calculate overall metrics
overall_weighted_ade = overall_weighted_ade_sum / overall_count if overall_count > 0 else 0
overall_vehicle_ade = overall_vehicle_ade_sum / vehicle_count if vehicle_count > 0 else 0
overall_vru_ade = overall_vru_ade_sum / vru_count if vru_count > 0 else 0
overall_unweighted_ade = unweighted_ade_sum / overall_count if overall_count > 0 else 0

overall_score = 0.5 * overall_vehicle_ade + 0.5 * overall_vru_ade

print(f"Overall Weighted ADE: {overall_weighted_ade}")
print(f"Overall Unweighted ADE: {overall_unweighted_ade}")
print(f"Overall Vehicle Weighted ADE: {overall_vehicle_ade}")
print(f"Overall VRU Weighted ADE: {overall_vru_ade}")
print(f"Overall Score: {overall_score}")

print("All metrics have been calculated.")