import os
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import yaml
import argparse
import matplotlib.pyplot as plt
import csv

def merge_tracks(tracking_results, time_threshold=3, spatial_threshold=5.0):
    """
    Merges fragmented tracks based on temporal and spatial consistency.
    
    Parameters:
    - tracking_results: Dictionary of tracks with timestamps as keys.
    - time_threshold: Maximum allowed gap in time between track fragments.
    - spatial_threshold: Maximum allowed spatial distance between track fragments.
    """
    
    # Store already merged track IDs to avoid redundant merging
    merged_track_ids = {}

    # Sort timestamps to process them in sequence
    sorted_timestamps = sorted(tracking_results.keys())

    # Iterate over each timestamp
    for i, timestamp in enumerate(sorted_timestamps):
        tracks = tracking_results[timestamp]

        for track in tracks:
            if track['state'] == 'confirmed':
                track_id = track['id']
                vehicle_type = track['vehicle_type']
                track_position = np.array(track['position'])

                # Check next timestamps within the time threshold
                for j in range(i+1, len(sorted_timestamps)):
                    other_timestamp = sorted_timestamps[j]
                    if (other_timestamp - timestamp) > time_threshold:
                        break  # Exit if the time difference exceeds the threshold

                    other_tracks = tracking_results[other_timestamp]

                    # Compare with other tracks in the future timestamp
                    for other_track in other_tracks:
                        if other_track['state'] == 'confirmed':
                            other_track_id = other_track['id']
                            other_vehicle_type = other_track['vehicle_type']
                            other_position = np.array(other_track['position'])

                            # Check if tracks belong to the same vehicle class
                            if vehicle_type == other_vehicle_type:
                                # Calculate spatial distance between tracks
                                distance = np.linalg.norm(track_position - other_position)

                                # If tracks are close in space, and belong to the same vehicle type, merge
                                if distance <= spatial_threshold:
                                    # If other track has already been merged, assign the previous merged ID
                                    if other_track_id in merged_track_ids:
                                        merged_track_id = merged_track_ids[other_track_id]
                                    else:
                                        merged_track_id = track_id  # Assign the current track's ID

                                    # Assign the same ID to the other track
                                    other_track['id'] = merged_track_id
                                    merged_track_ids[other_track_id] = merged_track_id  # Track merged IDs

                                    #print(f"Merged track {track_id} with {other_track_id} at time {other_timestamp}")

    return tracking_results


def load_config(config_file='config_kf_cv.yaml'):
    """Loads the configuration from a YAML file."""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return None

def is_dimension_consistent(vehicle_type, x_length, y_length, z_length, predefined_sizes, tolerance=0.2):
    """Check if the dimensions of a detection are consistent with its class type."""
    if vehicle_type not in predefined_sizes:
        return False

    predefined = predefined_sizes[vehicle_type]
    
    consistent = (
        abs(x_length - predefined['x_length']) <= tolerance * predefined['x_length'] and
        abs(y_length - predefined['y_length']) <= tolerance * predefined['y_length'] and
        abs(z_length - predefined['z_length']) <= tolerance * predefined['z_length']
    )
    
    return consistent

def remove_close_duplicates(detections, predefined_sizes, distance_threshold=1.0, tolerance=0.2):
    """Removes duplicate detections and checks for consistency among class types and dimensions."""
    if len(detections) == 0:
        return []

    positions = np.array([[det[0], det[1], det[2]] for det in detections])
    scores = np.array([det[7] for det in detections])

    keep = []

    for i in range(len(detections)):
        detection = detections[i]
        x, y, z, x_length, y_length, z_length, vehicle_type, score, z_rotation = detection

        # Commented out to allow all detections without dimension filtering
        #if not is_dimension_consistent(vehicle_type, x_length, y_length, z_length, predefined_sizes, tolerance):
        #    continue

        is_duplicate = False
        for j in range(i + 1, len(detections)):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < distance_threshold:
                if scores[i] < scores[j]:
                    is_duplicate = True
                    break
        if not is_duplicate:
            keep.append(i)

    return [detections[i] for i in keep]

def load_detections(detection_file, predefined_sizes):
    """Loads detections from a CSV file and removes duplicates."""
    detections = {}
    row_count = 0
    try:
        with open(detection_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_count += 1
                timestamp = float(row['Timestamps'])
                vehicle_type = row['subclass']
                x = float(row['x_center'])
                y = float(row['y_center'])
                z = float(row['z_center'])
                x_length = float(row['x_length'])
                y_length = float(row['y_length'])
                z_length = float(row['z_length'])
                z_rotation = float(row['z_rotation'])

                if timestamp not in detections:
                    detections[timestamp] = []

                detections[timestamp].append([x, y, z, x_length, y_length, z_length, vehicle_type, 1.0, z_rotation])

        # Debug: Print the number of timestamps and total detections
        print("Number of timestamps:", len(detections))
        print("Total number of detections:", sum(len(dets) for dets in detections.values()))
        
        for ts in detections:
            detections[ts] = remove_close_duplicates(detections[ts], predefined_sizes)

        # Debug: Print the total number of detections after duplicate removal
        print("Total number of detections after removing duplicates:", sum(len(dets) for dets in detections.values()))

    except Exception as e:
        print(f"Error loading detections: {e}")

    return detections

def save_tracking_results(results, output_file):
    """Saves the tracking results to a CSV file."""
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            #writer.writerow(["Timestamps", "subclass", "x_center", "x_length", "y_center", "y_length", "z_center", "z_length", "z_rotation"])
            writer.writerow(["Timestamps", "tracking_id", "subclass", "x_center", "x_length", "y_center", "y_length", "z_center", "z_length", "z_rotation"])
            confirmed_count = 0

            # # Debug: Print total number of timestamps in results
            # print("Total timestamps in results:", len(results))

            for timestamp, tracks in results.items():
                for track in tracks:

                    # # Debug: Print track state and details
                    # print(f"Timestamp: {timestamp}, Track ID: {track['id']}, State: {track['state']}")


                    if track['state'] == 'confirmed':
                        track_id = track['id']
                        vehicle_type = track['vehicle_type']
                        x, y, z = track['position']
                        x_length, y_length, z_length = track['dimensions']
                        heading = track['heading']

                        writer.writerow([
                            f"{timestamp:.5f}", track_id, vehicle_type, 
                            f"{x:.9f}", f"{x_length:.9f}", f"{y:.9f}", 
                            f"{y_length:.9f}", f"{z:.9f}", f"{z_length:.9f}", 
                            f"{heading:.7f}"
                        ]) 
                        confirmed_count += 1

            # # Debug: Print the total count of confirmed tracks saved
            # print("Total confirmed tracks saved:", confirmed_count)

            # # Check if no confirmed tracks were saved
            # if confirmed_count == 0:
            #     print("Warning: No confirmed tracks found for saving.")

    except Exception as e:
        print(f"Error saving tracking results: {e}")


def visualize_tracking(timestamp, trackers, ax, history, plot_folder, show_live_plot):
    """Visualizes the tracking results."""
    ax.clear()
    ax.set_title(f"Timestamp: {timestamp:.6f}")
    ax.set_xlim(-30, 62)
    ax.set_ylim(-50, 55)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')

    for tracker in trackers:
        if tracker.state == 'confirmed':
            x, y = tracker.kf.x[:2]
            track_id = tracker.id
            vehicle_type = tracker.vehicle_type
            heading = tracker.heading

            ax.plot(x, y, 'o', markersize=5, label=f'ID: {track_id}, Type: {vehicle_type}, Heading: {heading:.2f}°')
            ax.text(x, y, f'{track_id}, {vehicle_type}, {heading:.2f}°', fontsize=8, color='black', bbox=dict(facecolor='white', alpha=0.7))

            if track_id in history:
                hx, hy = zip(*history[track_id])
                ax.plot(hx, hy, 'o--', color='orange', alpha=0.7)

            if track_id not in history:
                history[track_id] = []
            history[track_id].append((x, y))

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc='upper left')

    if show_live_plot:
        plt.pause(0.1)

    output_path = os.path.join(plot_folder, f"timestamp_{timestamp:.6f}.png")
    plt.savefig(output_path)

class CVKalmanTracker:
    def __init__(self, config):
        self.dt = config['dt']
        self.kf = self.create_kf_cv(config['initial_state'], config['initial_covariance'])
        self.id = None
        self.vehicle_type = None
        self.dimensions = [0, 0, 0]
        self.heading = 0
        self.age = 0
        self.hits = 0
        self.missed = 0
        self.association_history = []
        self.state = 'tentative'
        self.confirmation_frames_needed = config['confirmation_frames_needed']
        self.confirmation_window = config['confirmation_window']
        self.deletion_missed_threshold = config['deletion_missed_threshold']
        self.deletion_window = config['deletion_window']

    def create_kf_cv(self, initial_state, initial_covariance):
        """Creates a Kalman Filter for a Constant Velocity (CV) model."""
        kf = KalmanFilter(dim_x=6, dim_z=3)
        dt = self.dt
        kf.F = np.array([[1, 0, 0, dt, 0, 0],
                         [0, 1, 0, 0, dt, 0],
                         [0, 0, 1, 0, 0, dt],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])
        kf.R = np.eye(3) * 0.1
        kf.Q = np.eye(6) * 0.01
        kf.x = initial_state.copy()
        kf.P = np.eye(6) * initial_covariance
        return kf

    def predict(self, dt):
        """Predict the next state with a given time step."""
        self.kf.F[:3, 3:] = np.eye(3) * dt
        self.kf.predict()
        self.age += 1
        vx, vy = self.kf.x[3], self.kf.x[4]
        if vx == 0 and vy == 0:
            self.heading = self.heading if self.heading is not None else 0.0
        else:
            angle = np.degrees(np.arctan2(-vx, vy))
            if angle < 0:
                angle += 360
            self.heading = angle

    def update(self, z, dimensions):
        """Update the state based on a new measurement."""
        self.kf.update(z)
        self.dimensions = dimensions
        self.hits += 1
        self.missed = 0
        self.association_history.append(1)
        
        if len(self.association_history) > self.confirmation_window:
            self.association_history.pop(0)
        
        if self.state == 'tentative' and self.association_history.count(1) >= self.confirmation_frames_needed:
            self.state = 'confirmed'
        elif self.state == 'confirmed':
            if self.association_history[-self.deletion_window:].count(0) >= self.deletion_missed_threshold:
                self.state = 'deleted'

class MultiObjectTracker:
    def __init__(self, config):
        self.trackers = []
        self.next_id = 0
        self.cost_threshold = config['cost_threshold']
        self.config = config

    def update(self, measurements, dt):
        """Updates all trackers with new measurements and manages tracker lifecycle."""
        for tracker in self.trackers:
            tracker.predict(dt)

        cost_matrix = self.compute_cost_matrix(measurements)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_detections = set()
        for t, d in zip(row_ind, col_ind):
            if cost_matrix[t, d] <= self.cost_threshold:
                detection = measurements[d]
                tracker = self.trackers[t]
                tracker.update(detection[:3], detection[3:6])
                tracker.vehicle_type = detection[6]
                assigned_tracks.add(t)
                assigned_detections.add(d)

        for t, tracker in enumerate(self.trackers):
            if t not in assigned_tracks:
                tracker.missed += 1
                tracker.association_history.append(0)
                if len(tracker.association_history) > tracker.confirmation_window:
                    tracker.association_history.pop(0)
                if tracker.state != 'deleted' and tracker.association_history[-tracker.deletion_window:].count(0) >= tracker.deletion_missed_threshold:
                    tracker.state = 'deleted'

        self.trackers = [tracker for tracker in self.trackers if tracker.state != 'deleted']

        for d, detection in enumerate(measurements):
            if d not in assigned_detections:
                initial_state = np.array([detection[0], detection[1], detection[2], 0, 0, 0])
                initial_covariance = np.eye(6) * 100
                new_tracker = CVKalmanTracker(self.config)
                new_tracker.id = self.next_id
                new_tracker.vehicle_type = detection[6]
                new_tracker.kf.x = initial_state.copy()
                new_tracker.dimensions = detection[3:6]
                self.next_id += 1
                self.trackers.append(new_tracker)

    def compute_cost_matrix(self, measurements):
        """Compute cost matrix for data association, incorporating subclass matching."""
        cost_matrix = np.zeros((len(self.trackers), len(measurements)))
        for t, tracker in enumerate(self.trackers):
            for d, detection in enumerate(measurements):
                predicted_state = tracker.kf.x[:3]
                measurement_state = detection[:3]
                distance_cost = np.linalg.norm(predicted_state - measurement_state)

                # Subclass matching cost: Higher cost if subclass does not match
                if tracker.vehicle_type != detection[6]:
                    subclass_cost = 10.0  # Assign a high cost if subclass doesn't match
                else:
                    subclass_cost = 0.0

                cost_matrix[t, d] = distance_cost + subclass_cost
        return cost_matrix

def main():
    predefined_sizes = {
        'VRU_Adult_Using_Bicycle': {'x_length': 0.89, 'y_length': 1.53, 'z_length': 1.38},
        'VRU_Adult_Using_Non-Motorized_Device/Prop_Other': {'x_length': 1.11, 'y_length': 1.15, 'z_length': 1.68},
        'Passenger_Vehicle': {'x_length': 2.6, 'y_length': 5.08, 'z_length': 1.85},
        'VRU_Adult_Using_Wheelchair': {'x_length': 1.08, 'y_length': 1.2, 'z_length': 1.28},
        'VRU_Adult_Using_Scooter_or_Skateboard': {'x_length': 1.31, 'y_length': 1.71, 'z_length': 1.4},
        'VRU_Adult': {'x_length': 0.83, 'y_length': 0.75, 'z_length': 1.59}
        ,'VRU_Child': {'x_length': 0.68, 'y_length': 0.65, 'z_length': 1.08},
        'VRU_Other': {'x_length': 0.83, 'y_length': 0.75, 'z_length': 1.59},
        'Vehicle_Other': {'x_length': 2.6, 'y_length': 5.08, 'z_length': 1.85}
    }

    parser = argparse.ArgumentParser(description='Run CV-based Multi-Object Tracking on detection CSV data.')
    parser.add_argument('--input', type=str, required=True, help='Path to CSV detection file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the tracking results CSV.')
    parser.add_argument('--show_live_plot', action='store_true', help='Flag to display live plot during tracking.')
    parser.add_argument('--merge_tracks', action='store_true', help='Flag to enable track merging.')
    args = parser.parse_args()

    config = load_config()

    detections = load_detections(args.input, predefined_sizes)

    # Create a plot folder for unmerged and merged tracks
    unmerged_plot_folder = os.path.join(os.path.dirname(args.output), f'plots_unmerged_{os.path.splitext(os.path.basename(args.output))[0]}')
    merged_plot_folder = os.path.join(os.path.dirname(args.output), f'plots_merged_{os.path.splitext(os.path.basename(args.output))[0]}')
    
   # os.makedirs(unmerged_plot_folder, exist_ok=True)
   # os.makedirs(merged_plot_folder, exist_ok=True)

    mot_tracker = MultiObjectTracker(config)

    tracking_results = {}
    history = {}

    fig, ax = plt.subplots()

    sorted_timestamps = sorted(detections.keys())

    previous_timestamp = None
    for timestamp in sorted_timestamps:
        if previous_timestamp is None:
            dt = config['dt']
        else:
            dt = timestamp - previous_timestamp

        mot_tracker.update(detections[timestamp], dt)
        tracking_results[timestamp] = []
        for tracker in mot_tracker.trackers:
            result = {
                'id': tracker.id,
                'vehicle_type': tracker.vehicle_type,
                'state': tracker.state,
                'position': tracker.kf.x[:3],
                'dimensions': tracker.dimensions,
                'heading': tracker.heading
            }
            tracking_results[timestamp].append(result)

        # Visualize unmerged tracking results
        #visualize_tracking(timestamp, mot_tracker.trackers, ax, history, unmerged_plot_folder, args.show_live_plot)
        previous_timestamp = timestamp

    # Check if track merging is enabled by user
    if args.merge_tracks:
        print("Merging fragmented tracks...")
        tracking_results = merge_tracks(tracking_results)

        # Reset the figure and history for merged visualization
        fig, ax = plt.subplots()
        history = {}

        # Visualize merged tracking results
        #for timestamp in sorted_timestamps:
            # Visualize merged tracking results after merging
            #visualize_tracking(timestamp, mot_tracker.trackers, ax, history, merged_plot_folder, args.show_live_plot)

    # Save tracking results after potential merging
    save_tracking_results(tracking_results, args.output)

    if args.show_live_plot:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
