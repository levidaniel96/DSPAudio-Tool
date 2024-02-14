# Creating Data Set for Audio Signal Processing

The `create_data_set` folder contains scripts and resources for generating a data set for audio signal processing tasks.

## Steps to Create Data Set

To create a data set, follow these steps:

1. **Configure Parameters**: Adjust the parameters of the data set in the `config.yaml` file located in the root of the project. The `config.yaml` file includes various parameters such as the number of samples, sampling frequency, room dimensions, speaker and microphone configurations, and noise levels.

2. **Run the Script**: Execute the `main.py` script to generate the data set based on the configured parameters. This script utilizes the parameters specified in the `config.yaml` file to create audio files and metadata.

3. **Access the Data Set**: Once the script finishes execution, the generated data set will be saved in the `data` folder located in the root of the project. The data set consists of WAV audio files and a CSV file containing metadata describing the parameters of each audio sample.

## Data Set Structure

- For Multiple Speakers: Each speaker's audio files are organized in separate subfolders within the `data` directory. Each audio file is named based on the speaker ID and sample number.

- For Multiple Microphones: Each WAV audio file contains multiple channels corresponding to different microphone positions.

## Configuration Parameters

The `config.yaml` file contains various parameters used to configure the data set generation process. These parameters include:

- General parameters such as the number of samples, sampling frequency, and record time.
- Speaker parameters defining the speaker setup, including distances and heights.
- Microphone parameters specifying the microphone configuration.
- Room parameters determining the dimensions and characteristics of the room.
- Noise parameters controlling the levels of white noise and environmental noise.
- Data set paths specifying the locations for clean data sets and the saved data set.

## Flags

Boolean flags in the `config.yaml` file control various aspects of the data set generation process, such as saving room impulse responses (RIR), audio waveforms, and noise types.
