import os
import numpy as np
from scipy import signal
import soundfile as sf
from pathlib import Path

def create_reverberant_audio(input_folder, output_folder, rir_folder, target_sr=16000):
    """
    Create reverberant audio files by convolving clean speech with RIRs
    
    Args:
        input_folder: Path to folder containing clean speech WAV files
        output_folder: Path to folder where reverberant files will be saved
        rir_folder: Path to folder containing RIR files (seat00.npy to seat07.npy)
        target_sr: Target sample rate (default: 16000 Hz)
    """
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Check if RIR folder exists
    if not os.path.exists(rir_folder):
        print(f"Error: RIR folder '{rir_folder}' does not exist!")
        return
    
    # Load all RIRs with better error handling
    rirs = []
    rir_names = []
    for i in range(8):
        rir_filename = f'seat{i:02d}.npy'
        rir_path = os.path.join(rir_folder, rir_filename)
        
        # Normalize the path to handle mixed slashes
        rir_path = os.path.normpath(rir_path)
        
        if not os.path.exists(rir_path):
            print(f"Error: RIR file '{rir_path}' does not exist!")
            print(f"Looking for: {rir_filename}")
            continue
            
        try:
            rir = np.load(rir_path)
            rirs.append(rir)
            rir_names.append(rir_filename)
            print(f"Loaded RIR: {rir_filename} with shape {rir.shape}")
        except Exception as e:
            print(f"Error loading {rir_path}: {str(e)}")
    
    if len(rirs) == 0:
        print("No RIR files were successfully loaded!")
        return
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return
    
    # Get all WAV files from input folder
    wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.wav')]
    print(f"Found {len(wav_files)} WAV files in {input_folder}")
    
    if len(wav_files) == 0:
        print("No WAV files found in the input folder!")
        return
    
    # Calculate how many files per RIR (roughly equal distribution)
    files_per_rir = len(wav_files) // len(rirs)
    remainder = len(wav_files) % len(rirs)
    
    print(f"Distributing {len(wav_files)} files across {len(rirs)} RIRs")
    print(f"Base files per RIR: {files_per_rir}, Remainder: {remainder}")
    
    file_counter = 0
    
    for rir_idx, (rir, rir_name) in enumerate(zip(rirs, rir_names)):
        # Calculate how many files for this RIR
        if rir_idx < remainder:
            num_files_for_this_rir = files_per_rir + 1
        else:
            num_files_for_this_rir = files_per_rir
            
        print(f"RIR {rir_name} will process {num_files_for_this_rir} files")
        
        for i in range(num_files_for_this_rir):
            if file_counter >= len(wav_files):
                break
                
            input_file = wav_files[file_counter]
            input_path = os.path.join(input_folder, input_file)
            
            # Generate output filename
            name_without_ext = os.path.splitext(input_file)[0]
            output_filename = f"{name_without_ext}_seat{rir_idx:02d}.wav"
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                # Read audio file
                audio, sr = sf.read(input_path)
                
                # Ensure audio is 1D (mono)
                if len(audio.shape) > 1:
                    audio = audio[:, 0]  # Take first channel if stereo
                
                # Resample if necessary
                if sr != target_sr:
                    # Calculate number of samples for target sample rate
                    num_samples = int(len(audio) * target_sr / sr)
                    audio = signal.resample(audio, num_samples)
                
                # Handle multi-channel RIR convolution
                # RIR shape is (samples, 8) - 8 channels
                # We'll create 8-channel output by convolving with each RIR channel
                reverberant_audio_channels = []
                
                for channel in range(rir.shape[1]):
                    # Get single channel from RIR
                    rir_channel = rir[:, channel]
                    
                    # Convolve audio with this RIR channel
                    reverberant_channel = signal.convolve(audio, rir_channel, mode='full')
                    reverberant_audio_channels.append(reverberant_channel)
                
                # Stack channels to create multi-channel audio
                # Transpose to get shape (samples, channels) for soundfile
                reverberant_audio = np.vstack(reverberant_audio_channels).T
                
                # Normalize to prevent clipping
                max_val = np.max(np.abs(reverberant_audio))
                if max_val > 0:
                    reverberant_audio = reverberant_audio / max_val * 0.95
                
                # Save reverberant audio as multi-channel WAV
                sf.write(output_path, reverberant_audio, target_sr)
                
                print(f"Processed: {input_file} -> {output_filename} (8 channels)")
                
            except Exception as e:
                print(f"Error processing {input_file}: {str(e)}")
            
            file_counter += 1
    
    print(f"\nProcessing complete! Created {file_counter} reverberant files in {output_folder}")

if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    # Use raw strings or forward slashes to avoid path issues
    INPUT_FOLDER = r"./data_project/clean"  # Raw string for Windows paths
    OUTPUT_FOLDER = r"./data_project/distant"  # Raw string for Windows paths  
    RIR_FOLDER = r"./data_project/rir"             # Raw string for Windows paths
    
    print("Configuration:")
    print(f"Input folder: {INPUT_FOLDER} (exists: {os.path.exists(INPUT_FOLDER)})")
    print(f"Output folder: {OUTPUT_FOLDER} (exists: {os.path.exists(OUTPUT_FOLDER)})")
    print(f"RIR folder: {RIR_FOLDER} (exists: {os.path.exists(RIR_FOLDER)})")
    
    # Run the processing
    create_reverberant_audio(INPUT_FOLDER, OUTPUT_FOLDER, RIR_FOLDER)