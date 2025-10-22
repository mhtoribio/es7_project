import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
import re

class RETFPSDEstimator(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, retf_output_size=257, psd_output_size=257):
        """
        MLP for estimating RETF and target PSD from reverberant speech
        
        Args:
            input_size: STFT frequency bins (magnitude only)
            hidden_size: Hidden layer size
            retf_output_size: RETF output size (frequency bins * channels)
            psd_output_size: PSD output size (frequency bins)
        """
        super(RETFPSDEstimator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
        )
        
        # Separate heads for RETF and PSD estimation
        self.retf_head = nn.Sequential(
            nn.Linear(hidden_size // 4, retf_output_size),
            nn.Tanh()  # Keep outputs in reasonable range
        )
        
        self.psd_head = nn.Sequential(
            nn.Linear(hidden_size // 4, psd_output_size),
            nn.Softplus()  # PSD must be non-negative
        )
        
    def forward(self, x):
        features = self.encoder(x)
        
        retf_output = self.retf_head(features)
        psd_output = self.psd_head(features)
        
        return retf_output, psd_output

def compute_stft(audio, n_fft=512, hop_length=256):
    """Compute STFT of audio signal"""
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # Take first channel if multi-channel
    
    # Compute STFT using scipy
    f, t, stft = signal.stft(audio, fs=16000, nperseg=n_fft, noverlap=n_fft-hop_length)
    return f, t, stft

def extract_early_rir(rir, early_samples=512):
    """Extract early part of RIR (first 512 samples = 32ms at 16kHz)"""
    if len(rir.shape) > 1:
        return rir[:early_samples, :]
    else:
        return rir[:early_samples]

def compute_true_retf(rir_early, n_fft=512):
    """
    Compute true RETF from early RIR
    Based on paper: RETF is relative to first microphone
    """
    # Ensure rir_early is 2D (samples, channels)
    if len(rir_early.shape) == 1:
        rir_early = rir_early.reshape(-1, 1)
    
    num_channels = rir_early.shape[1]
    
    # Compute frequency response for each channel
    retf_true = []
    for ch in range(num_channels):
        f, h = signal.freqz(rir_early[:, ch], worN=n_fft, fs=16000)
        retf_true.append(h)
    
    retf_true = np.array(retf_true).T  # Shape: (freq_bins, channels)
    
    # Make relative to first microphone (if multiple channels)
    if num_channels > 1:
        retf_relative = retf_true[:, 1:] / (retf_true[:, 0:1] + 1e-10)
    else:
        retf_relative = retf_true  # Single channel case
    
    return retf_relative

def compute_true_psd(clean_speech, rir_early, n_fft=512):
    """Compute true PSD of early speech component"""
    # Convolve clean speech with early RIR (first channel)
    if len(rir_early.shape) > 1:
        rir_mono = rir_early[:, 0]
    else:
        rir_mono = rir_early
        
    early_speech = signal.convolve(clean_speech, rir_mono, mode='same')
    
    # Compute PSD using Welch's method
    f, psd = signal.welch(early_speech, fs=16000, nperseg=n_fft)
    
    return f, psd

def extract_seat_number(filename):
    """Extract seat number from filename using regex"""
    # Look for patterns like _seat00, seat00, _seat0, seat0, etc.
    patterns = [
        r'_seat(\d+)',  # _seat00, _seat01, etc.
        r'seat(\d+)',   # seat00, seat01, etc.
        r'_s(\d+)',     # _s00, _s01, etc.
        r's(\d+)',      # s00, s01, etc.
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    return None

def prepare_training_data(clean_folder, reverberant_folder, rir_folder, n_fft=512, max_files=50):
    """Prepare training data from folders"""
    
    clean_files = [f for f in os.listdir(clean_folder) if f.endswith('.wav')]
    data_pairs = []
    
    print(f"Found {len(clean_files)} clean files")
    
    for clean_file in clean_files[:max_files]:  # Limit files for testing
        try:
            # Load clean speech
            clean_path = os.path.join(clean_folder, clean_file)
            clean_audio, _ = sf.read(clean_path)
            if len(clean_audio.shape) > 1:
                clean_audio = clean_audio[:, 0]  # Take first channel
            
            print(f"Processing {clean_file}...")
            
            # Find corresponding reverberant files
            base_name = os.path.splitext(clean_file)[0]
            reverberant_files = [f for f in os.listdir(reverberant_folder) 
                               if f.startswith(base_name) and f.endswith('.wav')]
            
            if not reverberant_files:
                print(f"  No reverberant file found for {clean_file}")
                continue
            
            print(f"  Found {len(reverberant_files)} reverberant files")
            
            # Process each reverberant file
            for reverberant_file in reverberant_files:
                try:
                    reverberant_path = os.path.join(reverberant_folder, reverberant_file)
                    reverberant_audio, _ = sf.read(reverberant_path)
                    if len(reverberant_audio.shape) > 1:
                        reverberant_audio = reverberant_audio[:, 0]  # Take first channel
                    
                    # Extract seat number from reverberant filename
                    seat_match = extract_seat_number(reverberant_file)
                    
                    if seat_match is None:
                        print(f"    Could not extract seat number from {reverberant_file}")
                        continue
                    
                    # Try different RIR file naming patterns
                    rir_patterns = [
                        f"seat{int(seat_match):02d}.npy",  # seat00.npy, seat01.npy, etc.
                        f"seat{seat_match}.npy",           # seat0.npy, seat1.npy, etc.
                        f"seat{int(seat_match)}.npy",      # seat0.npy, seat1.npy, etc.
                    ]
                    
                    rir_data = None
                    rir_file_used = None
                    
                    for rir_pattern in rir_patterns:
                        rir_path = os.path.join(rir_folder, rir_pattern)
                        if os.path.exists(rir_path):
                            rir_data = np.load(rir_path)
                            rir_file_used = rir_pattern
                            break
                    
                    if rir_data is None:
                        print(f"    No RIR file found for seat {seat_match}")
                        continue
                    
                    print(f"    Loaded RIR: {rir_file_used} with shape {rir_data.shape}")
                    
                    # Extract early RIR
                    rir_early = extract_early_rir(rir_data)
                    
                    # Compute STFT of reverberant speech
                    f, t, stft_reverberant = compute_stft(reverberant_audio, n_fft=n_fft)
                    
                    # Compute true RETF and PSD
                    retf_true = compute_true_retf(rir_early, n_fft=n_fft)
                    f_psd, psd_true = compute_true_psd(clean_audio, rir_early, n_fft=n_fft)
                    
                    # Use only magnitude of STFT as features
                    stft_magnitude = np.abs(stft_reverberant).T  # Shape: (time_frames, freq_bins)
                    
                    data_pairs.append({
                        'reverberant_magnitude': stft_magnitude,
                        'retf_true': retf_true,
                        'psd_true': psd_true,
                        'clean_file': clean_file,
                        'reverberant_file': reverberant_file,
                        'rir_file': rir_file_used
                    })
                    
                    print(f"    Processed: STFT shape {stft_magnitude.shape}, RETF shape {retf_true.shape}, PSD shape {psd_true.shape}")
                    
                except Exception as e:
                    print(f"    Error processing reverberant file {reverberant_file}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing clean file {clean_file}: {e}")
            continue
    
    return data_pairs

def create_dataset(data_pairs, n_fft=512, max_frames_per_file=100):
    """Create PyTorch dataset from prepared data"""
    
    X = []
    y_retf = []
    y_psd = []
    
    for pair in data_pairs:
        stft_magnitude = pair['reverberant_magnitude']  # Shape: (time_frames, freq_bins)
        retf_true = pair['retf_true']  # Shape: (freq_bins, channels)
        psd_true = pair['psd_true']    # Shape: (freq_bins,)
        
        # Limit number of frames per file to avoid memory issues
        num_frames = min(stft_magnitude.shape[0], max_frames_per_file)
        stft_magnitude = stft_magnitude[:num_frames]
        
        # Flatten RETF to 1D array
        retf_flat = retf_true.flatten()  # Shape: (freq_bins * channels,)
        
        # Repeat RETF and PSD for each time frame
        for i in range(num_frames):
            X.append(stft_magnitude[i])  # One frame of STFT magnitude
            y_retf.append(retf_flat)     # Same RETF for all frames of this file
            y_psd.append(psd_true)       # Same PSD for all frames of this file
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(np.array(X))
    y_retf_tensor = torch.FloatTensor(np.array(y_retf))
    y_psd_tensor = torch.FloatTensor(np.array(y_psd))
    
    print(f"Dataset shapes - X: {X_tensor.shape}, y_retf: {y_retf_tensor.shape}, y_psd: {y_psd_tensor.shape}")
    
    return torch.utils.data.TensorDataset(X_tensor, y_retf_tensor, y_psd_tensor)

def train_model(model, dataset, epochs=50, batch_size=32):
    """Train the RETF and PSD estimation model"""
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    mse_loss = nn.MSELoss()
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_retf, batch_psd in dataloader:
            optimizer.zero_grad()
            
            pred_retf, pred_psd = model(batch_X)
            
            # Compute losses
            retf_loss = mse_loss(pred_retf, batch_retf)
            psd_loss = mse_loss(pred_psd, batch_psd)
            
            total_loss = retf_loss + psd_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    return train_losses

def main():
    # Configuration
    CLEAN_FOLDER = "./data_project/clean"
    REVERBERANT_FOLDER = "./data_project/distant" 
    RIR_FOLDER = "./data_project/rir"
    
    n_fft = 512
    hop_length = 256
    
    print("Preparing training data...")
    data_pairs = prepare_training_data(CLEAN_FOLDER, REVERBERANT_FOLDER, RIR_FOLDER, n_fft, max_files=20)
    
    if not data_pairs:
        print("No training data found! Check your file paths and naming conventions.")
        return
    
    print(f"Prepared {len(data_pairs)} training pairs")
    
    print("Creating dataset...")
    dataset = create_dataset(data_pairs, n_fft, max_frames_per_file=50)
    
    if len(dataset) == 0:
        print("No valid training samples created!")
        return
    
    # Calculate output sizes based on RETF and PSD shapes
    sample_retf = data_pairs[0]['retf_true']
    sample_psd = data_pairs[0]['psd_true']
    
    retf_output_size = sample_retf.shape[0] * sample_retf.shape[1]  # freq_bins * channels
    psd_output_size = sample_psd.shape[0]  # freq_bins
    
    print(f"Model output sizes - RETF: {retf_output_size}, PSD: {psd_output_size}")
    
    print("Initializing model...")
    model = RETFPSDEstimator(
        input_size=n_fft//2 + 1,  # STFT magnitude bins
        hidden_size=256,
        retf_output_size=retf_output_size,
        psd_output_size=psd_output_size
    )
    
    print("Training model...")
    losses = train_model(model, dataset, epochs=50, batch_size=32)
    
    # Save model
    torch.save(model.state_dict(), "retf_psd_estimator.pth")
    print("Model saved as 'retf_psd_estimator.pth'")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

if __name__ == "__main__":
    main()