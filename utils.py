import numpy as np
import os
def generate_room_parameters(params):
    """
    Generate random room parameters for a given number of sources.

    Args:
        num_spk (int): Number of sources in the room.

    Returns:
        dict: A dictionary containing the following keys:
            - T60 (float): Random T60 between 0.3 and 1.2.
            - Lx (float): Random room length between 4 and 18 meters.
            - Ly (float): Random room width between 4 and 18 meters.
            - Lz (float): Random room height between 6 and 8 meters.
            - mic_x (float): Random microphone x position between 0.5 and Lx - 0.5 meters.
            - mic_y (float): Random microphone y position between 0.5 and Ly - 0.5 meters.
            - mic_z (float): Random microphone z position between 2 and Lz - 0.5 meters.
            - sources (list): A list of dictionaries containing the x, y, and z positions of each source.
            - roomClass (int): Room classification based on dimensions and volume:
                - 0: Small room (V < 35)
                - 1: Mid-size room (35 <= V and XYratio < 4)
                - 2: Long room (XYratio >= 4)
            - V (float): Room volume (Lx * Ly * Lz).
    """
    # Generate random T60 between 0.3 and 1.2
    T60 = np.random.uniform(params.T60_min, params.T60_max)
    # Generate random room dimensions
    Lx = np.random.uniform(params.room_min_x, params.room_max_x)
    Ly = np.random.uniform(params.room_min_y, params.room_max_y)
    Lz = np.random.uniform(params.room_min_height, params.room_max_height)
    
    ## check that the mic and sources are in the room
    while Lx < 2*params.mic_min_distance_from_wall or Ly < 2*params.mic_min_distance_from_wall:
        Lx = np.random.uniform(params.room_min_x, params.room_max_x)
        Ly = np.random.uniform(params.room_min_y, params.room_max_y)
    while Lz < 2*params.mic_min_height:
        Lz = np.random.uniform(params.room_min_height, params.room_max_height)
        
    # Generate random microphone position and ensure it's not too close to the wall
    mic_x = np.random.uniform(params.mic_min_distance_from_wall, Lx - params.mic_min_distance_from_wall)
    mic_y = np.random.uniform(params.mic_min_distance_from_wall, Ly - params.mic_min_distance_from_wall)
    mic_z = np.random.uniform(params.mic_min_height, Lz - params.mic_min_height)

    while np.sqrt((Lx - mic_x) ** 2 + (Ly - mic_y) ** 2) < params.mic_min_distance_from_wall:
        mic_x = np.random.uniform(params.mic_min_distance_from_wall, Lx - params.mic_min_distance_from_wall)
        mic_y = np.random.uniform(params.mic_min_distance_from_wall, Ly - params.mic_min_distance_from_wall)

    # Generate random source positions and ensure they are not too close to the mic and not too close to the wall 
    sources = []
    for spk in range(params.num_spk):
        source_x = np.random.uniform(params.speaker_min_distance_from_wall, Lx - params.speaker_min_distance_from_wall)
        source_y = np.random.uniform(params.speaker_min_distance_from_wall, Ly - params.speaker_min_distance_from_wall)
        source_z = np.random.uniform(params.speaker_min_height, params.speaker_max_height)

        while np.sqrt((mic_x - source_x) ** 2 + (mic_y - source_y) ** 2) > params.speaker_max_distance_from_mic or \
              np.sqrt((mic_x - source_x) ** 2 + (mic_y - source_y) ** 2) < params.speaker_min_distance_from_mic:
            source_x = np.random.uniform(params.speaker_min_distance_from_wall, Lx - params.speaker_min_distance_from_wall)
            source_y = np.random.uniform(params.speaker_min_distance_from_wall, Ly - params.speaker_min_distance_from_wall)

        sources.append({'x': source_x, 'y': source_y, 'z': source_z})

    # Calculate room volume and classify the room based on its dimensions and volume
    V = Lx * Ly * Lz
    XYratio = max(Lx, Ly) / min(Lx, Ly)
    if V < params.small_room_volume_threshold:
        roomClass = 0
    elif V >= params.small_room_volume_threshold and XYratio < params.mid_size_room_XYratio_threshold:
        roomClass = 1
    else:
        roomClass = 2

    # Return a dictionary containing all the generated parameters
    return {'T60': T60, 'Lx': Lx, 'Ly': Ly, 'Lz': Lz, 'mic_x': mic_x, 'mic_y': mic_y, 'mic_z': mic_z, 'sources': sources, 'V': V, 'roomClass': roomClass}


def extract_wav_paths(base_dir):
    """
    Recursively traverse the directories and collect paths of all WAV files.

    Args:
        base_dir (str): The base directory to start the search from.

    Returns:
        list: A list of paths to all the WAV files found in the directory tree.
    """
    # Initialize an empty list to store the file paths
    wav_paths = []
    # Recursively traverse the directories and collect WAV file paths
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_path = os.path.join(root, file)
                wav_paths.append(wav_path)

        return wav_paths


