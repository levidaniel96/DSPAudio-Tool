from dataclasses import dataclass

@dataclass
class Params:
    # General parameters
    num_of_samples: int = 1000
    fs: int = 16000
    M: int = 1
    record_time: int = 5
    num_spk: int = 1
    ref_mic: int = 0
    T60_min: float = 0.1
    T60_max: float = 0.6

    # Speaker parameters
    speaker_min_distance_from_wall: float = 0.5
    speaker_min_distance_from_mic: float = 0.5
    speaker_max_distance_from_mic: float = 1.5
    speaker_min_height: float = 1
    speaker_max_height: float = 1.7

    # Mic parameters
    mic_dis_x: float = 0.5
    mic_dis_y: float = 0.5
    mic_min_distance_from_wall: float = 0.5
    mic_min_height: float = 1
    mic_max_height: float = 1.5

    # Room parameters
    room_min_x: float = 3
    room_max_x: float = 6
    room_min_y: float = 3
    room_max_y: float = 6

    room_min_height: float = 2
    room_max_height: float = 2.5

    # noise
    SNR_white_noise_low: int = 0
    SNR_white_noise_high: int = 10
    SNR_env_noise_low: int = 0
    SNR_env_noise_high: int = 10

    
    # For room size classification
    mid_size_room_XYratio_threshold: float = 1.5
    small_room_volume_threshold: int = 20

@dataclass
class Paths:
    clean_data_set_path: str = ""
    save_data_set_path: str = ""
    env_noise_path: str = ""
@dataclass
class Flags:
    save_RIR: bool = True
    save_wav: bool = True
    env_noise: bool = True
    white_noise: bool = True

@dataclass
class HydraConfig:
    run_dir: str = "${model_path}/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"

@dataclass 
class create_data_config:
    paths: Paths             # Paths configuration
    params: Params           # Parameters configuration
    flags: Flags             # Flags configuration
    hydra: HydraConfig       # Hydra configuration