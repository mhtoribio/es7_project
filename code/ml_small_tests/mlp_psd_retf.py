import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
import re

class RETFEstimator(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, output_size=257):
        """
        Separate MLP for estimating RETF from reverberant speech
        
        Args:
            input_size: STFT frequency bins (magnitude + phase concatenated)
            hidden_size: Hidden layer size
            output_size: RETF output size (frequency bins * channels)
        """
        super(RETFEstimator, self).__init__()
        
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
            nn.BatchNorm1d(hidden_size // 4),
        )
        
        self.retf_head = nn.Sequential(
            nn.Linear(hidden_size // 4, output_size),
            nn.Tanh()  # Keep outputs in reasonable range
        )
        
    def forward(self, x):
        features = self.encoder(x)
        retf_output = self.retf_head(features)
        return retf_output

class PSDEstimator(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, output_size=257):
        """
        Separate MLP for estimating PSD from reverberant speech
        
        Args:
            input_size: STFT frequency bins (magnitude + phase concatenated)
            hidden_size: Hidden layer size
            output_size: PSD output size (frequency bins)
        """
        super(PSDEstimator, self).__init__()
        
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
            nn.BatchNorm1d(hidden_size // 4),
        )
        
        self.psd_head = nn.Sequential(
            nn.Linear(hidden_size // 4, output_size),
            nn.Softplus()  # PSD must be non-negative
        )
        
    def forward(self, x):
        features = self.encoder(x)
        psd_output = self.psd_head(features)
        return psd_output

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
                    
                    # Extract magnitude and phase from STFT
                    stft_magnitude = np.abs(stft_reverberant).T  # Shape: (time_frames, freq_bins)
                    stft_phase = np.angle(stft_reverberant).T   # Shape: (time_frames, freq_bins)
                    
                    # Normalize phase to [-1, 1] range (since phase is in [-pi, pi])
                    stft_phase_normalized = stft_phase / np.pi
                    
                    # Concatenate magnitude and phase along feature dimension
                    stft_features = np.concatenate([stft_magnitude, stft_phase_normalized], axis=1)
                    
                    data_pairs.append({
                        'reverberant_features': stft_features,
                        'retf_true': retf_true,
                        'psd_true': psd_true,
                        'clean_file': clean_file,
                        'reverberant_file': reverberant_file,
                        'rir_file': rir_file_used
                    })
                    
                    print(f"    Processed: Features shape {stft_features.shape}, RETF shape {retf_true.shape}, PSD shape {psd_true.shape}")
                    
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
        stft_features = pair['reverberant_features']  # Shape: (time_frames, 2 * freq_bins)
        retf_true = pair['retf_true']  # Shape: (freq_bins, channels)
        psd_true = pair['psd_true']    # Shape: (freq_bins,)
        
        # Limit number of frames per file to avoid memory issues
        num_frames = min(stft_features.shape[0], max_frames_per_file)
        stft_features = stft_features[:num_frames]
        
        # Flatten RETF to 1D array
        retf_flat = retf_true.flatten()  # Shape: (freq_bins * channels,)
        
        # Repeat RETF and PSD for each time frame
        for i in range(num_frames):
            X.append(stft_features[i])  # One frame of STFT features (magnitude + phase)
            y_retf.append(retf_flat)    # Same RETF for all frames of this file
            y_psd.append(psd_true)      # Same PSD for all frames of this file
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(np.array(X))
    y_retf_tensor = torch.FloatTensor(np.array(y_retf))
    y_psd_tensor = torch.FloatTensor(np.array(y_psd))
    
    print(f"Dataset shapes - X: {X_tensor.shape}, y_retf: {y_retf_tensor.shape}, y_psd: {y_psd_tensor.shape}")
    
    return torch.utils.data.TensorDataset(X_tensor, y_retf_tensor, y_psd_tensor)

def train_separate_models(retf_model, psd_model, dataset, epochs=50, batch_size=32):
    """Train the separate RETF and PSD estimation models"""
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Separate optimizers for each model
    retf_optimizer = optim.Adam(retf_model.parameters(), lr=0.001, weight_decay=1e-5)
    psd_optimizer = optim.Adam(psd_model.parameters(), lr=0.001, weight_decay=1e-5)
    
    mse_loss = nn.MSELoss()
    
    retf_losses = []
    psd_losses = []
    total_losses = []
    
    for epoch in range(epochs):
        retf_model.train()
        psd_model.train()
        
        epoch_retf_loss = 0
        epoch_psd_loss = 0
        epoch_total_loss = 0
        
        for batch_X, batch_retf, batch_psd in dataloader:
            # Train RETF model
            retf_optimizer.zero_grad()
            pred_retf = retf_model(batch_X)
            retf_loss = mse_loss(pred_retf, batch_retf)
            retf_loss.backward()
            retf_optimizer.step()
            
            # Train PSD model
            psd_optimizer.zero_grad()
            pred_psd = psd_model(batch_X)
            psd_loss = mse_loss(pred_psd, batch_psd)
            psd_loss.backward()
            psd_optimizer.step()
            
            epoch_retf_loss += retf_loss.item()
            epoch_psd_loss += psd_loss.item()
            epoch_total_loss += (retf_loss.item() + psd_loss.item())
        
        avg_retf_loss = epoch_retf_loss / len(dataloader)
        avg_psd_loss = epoch_psd_loss / len(dataloader)
        avg_total_loss = epoch_total_loss / len(dataloader)
        
        retf_losses.append(avg_retf_loss)
        psd_losses.append(avg_psd_loss)
        total_losses.append(avg_total_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, RETF Loss: {avg_retf_loss:.6f}, PSD Loss: {avg_psd_loss:.6f}, Total Loss: {avg_total_loss:.6f}")
    
    return retf_losses, psd_losses, total_losses

def main():
    # Configuration
    CLEAN_FOLDER = "./data_project/clean"
    REVERBERANT_FOLDER = "./data_project/distant" 
    RIR_FOLDER = "./data_project/rir"
    
    n_fft = 512
    hop_length = 256
    num_freq_bins = n_fft // 2 + 1  # 257 for n_fft=512
    
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
    
    # Input size is now 2 * num_freq_bins (magnitude + phase)
    input_size = 2 * num_freq_bins
    
    print(f"Model input size: {input_size}")
    print(f"RETF output size: {retf_output_size}")
    print(f"PSD output size: {psd_output_size}")
    
    print("Initializing separate models...")
    retf_model = RETFEstimator(
        input_size=input_size,
        hidden_size=512,
        output_size=retf_output_size
    )
    
    psd_model = PSDEstimator(
        input_size=input_size, 
        hidden_size=512,
        output_size=psd_output_size
    )
    
    print("Training separate models...")
    retf_losses, psd_losses, total_losses = train_separate_models(
        retf_model, psd_model, dataset, epochs=50, batch_size=32
    )
    
    # Save models separately
    torch.save(retf_model.state_dict(), "retf_estimator.pth")
    torch.save(psd_model.state_dict(), "psd_estimator.pth")
    print("Models saved as 'retf_estimator.pth' and 'psd_estimator.pth'")
    
    # Plot training losses
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(retf_losses, label='RETF Loss', color='blue')
    plt.plot(psd_losses, label='PSD Loss', color='red')
    plt.plot(total_losses, label='Total Loss', color='green', linestyle='--')
    plt.title('Training Losses - Separate Models')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(retf_losses, label='RETF Loss', color='blue')
    plt.plot(psd_losses, label='PSD Loss', color='red')
    plt.title('Training Losses (Log Scale) - Separate Models')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_losses_separate_models.png')
    plt.show()

if __name__ == "__main__":
    main()