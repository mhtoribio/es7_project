import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
from pathlib import Path
import soundfile as sf
import argparse

# Configuration
CLEAN_FOLDER = "./data_project/clean"
REVERBERANT_FOLDER = "./data_project/distant" 
RIR_FOLDER = "./data_project/rir"
n_fft = 512
hop_length = 256
num_freq_bins = n_fft // 2 + 1

# Dataset Class
class RETFDataset(torch.utils.data.Dataset):
    def __init__(self, clean_folder, reverberant_folder, rir_folder, n_fft=512, hop_length=256):
        self.clean_folder = Path(clean_folder)
        self.reverberant_folder = Path(reverberant_folder) 
        self.rir_folder = Path(rir_folder)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_freq_bins = n_fft // 2 + 1
        
        # Get all clean audio files
        self.clean_files = list(self.clean_folder.glob("*.wav")) + list(self.clean_folder.glob("*.flac"))
        
        # Create mapping to reverberant and RIR files
        self.file_pairs = []
        for clean_file in self.clean_files:
            base_name = clean_file.stem
            reverberant_file = self.reverberant_folder / f"{base_name}.wav"
            rir_file = self.rir_folder / f"{base_name}.wav"
            
            if reverberant_file.exists() and rir_file.exists():
                self.file_pairs.append((clean_file, reverberant_file, rir_file))
        
        print(f"Found {len(self.file_pairs)} valid file pairs")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def load_audio(self, file_path):
        audio, sr = sf.read(str(file_path))
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Take first channel if multi-channel
        return audio, sr
    
    def compute_stft(self, audio):
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        return stft  # Returns (freq_bins, time_frames) complex64
    
    def extract_early_rir(self, rir_audio, sr, early_duration=0.05):
        """Extract early part of RIR (direct path + early reflections)"""
        early_samples = int(early_duration * sr)
        early_rir = rir_audio[:early_samples]
        
        # Pad if shorter than early_duration
        if len(early_rir) < early_samples:
            early_rir = np.pad(early_rir, (0, early_samples - len(early_rir)))
        
        return early_rir
    
    def compute_ground_truth_retf(self, rir_audio, sr):
        """Compute ground truth RETF from RIR"""
        early_rir = self.extract_early_rir(rir_audio, sr)
        
        # Compute transfer function (frequency response) of early RIR
        retf = np.fft.rfft(early_rir, n=self.n_fft)
        
        # Normalize relative to first microphone (paper constraint)
        retf = retf / retf[0]  # Relative to first "microphone"
        
        return retf[:self.num_freq_bins]  # Truncate to STFT frequency bins
    
    def __getitem__(self, idx):
        clean_file, reverberant_file, rir_file = self.file_pairs[idx]
        
        # Load audio files
        clean_audio, sr = self.load_audio(clean_file)
        reverberant_audio, sr = self.load_audio(reverberant_file)
        rir_audio, sr = self.load_audio(rir_file)
        
        # Compute STFT of reverberant signal (input features)
        reverberant_stft = self.compute_stft(reverberant_audio)
        
        # Compute ground truth RETF (target)
        gt_retf = self.compute_ground_truth_retf(rir_audio, sr)
        
        # Convert to magnitude and phase for easier learning
        input_magnitude = np.abs(reverberant_stft)
        input_phase = np.angle(reverberant_stft)
        
        # Stack magnitude and phase as separate channels
        input_features = np.stack([input_magnitude, input_phase], axis=0)
        
        # Ground truth RETF as complex numbers (real + imaginary)
        gt_retf_real = gt_retf.real
        gt_retf_imag = gt_retf.imag
        gt_retf_complex = np.stack([gt_retf_real, gt_retf_imag], axis=0)
        
        return {
            'input_features': torch.FloatTensor(input_features),
            'gt_retf': torch.FloatTensor(gt_retf_complex),
            'file_name': str(clean_file.stem)
        }

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    inputs = [item['input_features'] for item in batch]
    targets = [item['gt_retf'] for item in batch]
    file_names = [item['file_name'] for item in batch]
    
    # Pad sequences to same length
    max_time = max([x.shape[2] for x in inputs])
    
    padded_inputs = []
    for x in inputs:
        pad_size = max_time - x.shape[2]
        if pad_size > 0:
            padded = F.pad(x, (0, pad_size))
        else:
            padded = x
        padded_inputs.append(padded)
    
    inputs_batch = torch.stack(padded_inputs)
    targets_batch = torch.stack(targets)
    
    return {
        'input_features': inputs_batch,
        'gt_retf': targets_batch,
        'file_names': file_names
    }

# Model Definition
class RETFEstimator(nn.Module):
    def __init__(self, num_freq_bins=257, hidden_dim=256, num_layers=4):
        super(RETFEstimator, self).__init__()
        self.num_freq_bins = num_freq_bins
        
        # Frequency-aware encoder
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(3, 3), padding=(1, 1)),  # (mag, phase) -> 64 channels
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((num_freq_bins, None))  # Keep frequency dimension
        )
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Frequency-wise attention
        self.freq_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # RETF decoder (predicts complex RETF coefficients)
        self.retf_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_freq_bins * 2)  # *2 for real and imaginary parts
        )
        
    def forward(self, x):
        # x shape: (batch, 2, freq_bins, time_frames)
        batch_size, _, num_freq, time_frames = x.shape
        
        # Encode frequency-time patterns
        encoded = self.freq_encoder(x)  # (batch, 128, freq_bins, time_frames)
        
        # Rearrange for LSTM: (batch, time_frames, freq_bins, 128) -> (batch, freq_bins, time_frames, 128)
        encoded = encoded.permute(0, 2, 3, 1)  # (batch, freq_bins, time_frames, 128)
        encoded = encoded.reshape(batch_size * num_freq, time_frames, 128)
        
        # Temporal modeling with LSTM
        lstm_out, _ = self.lstm(encoded)  # (batch*freq_bins, time_frames, hidden_dim*2)
        
        # Apply frequency-wise attention
        attention_weights = self.freq_attention(lstm_out)  # (batch*freq_bins, time_frames, 1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)  # (batch*freq_bins, hidden_dim*2)
        
        # Reshape back to frequency dimension
        attended = attended.reshape(batch_size, num_freq, -1)  # (batch, freq_bins, hidden_dim*2)
        
        # Decode to RETF coefficients
        retf_out = self.retf_decoder(attended)  # (batch, freq_bins, num_freq_bins*2)
        retf_out = retf_out.reshape(batch_size, num_freq, self.num_freq_bins, 2)
        
        # Convert to complex numbers
        retf_complex = torch.complex(retf_out[..., 0], retf_out[..., 1])
        
        # Apply RETF constraint: first frequency bin = 1 (relative transfer function)
        retf_complex[:, :, 0] = 1.0 + 0j
        
        return retf_complex

# Loss Function
class RETFLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.01):
        super(RETFLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred_retf, gt_retf, input_magnitude):
        # Complex MSE loss
        complex_loss = F.mse_loss(pred_retf.real, gt_retf.real) + \
                      F.mse_loss(pred_retf.imag, gt_retf.imag)
        
        # Magnitude consistency loss
        pred_magnitude = torch.abs(pred_retf)
        gt_magnitude = torch.abs(gt_retf)
        magnitude_loss = F.mse_loss(pred_magnitude, gt_magnitude)
        
        # Smoothness regularization (RETF should be smooth across frequencies)
        freq_diff = pred_retf[:, :, 1:] - pred_retf[:, :, :-1]
        smoothness_loss = torch.mean(torch.abs(freq_diff))
        
        total_loss = complex_loss + self.alpha * magnitude_loss + self.beta * smoothness_loss
        return total_loss

# Trainer Class
class RETFTrainer:
    def __init__(self, model, train_loader, val_loader, device, lr=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = RETFLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_features = batch['input_features'].to(self.device)
            gt_retf = batch['gt_retf'].to(self.device)
            
            # Extract magnitude for loss computation
            input_magnitude = input_features[:, 0]  # First channel is magnitude
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_retf = self.model(input_features)
            
            # Compute loss
            loss = self.criterion(pred_retf, gt_retf, input_magnitude)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_features = batch['input_features'].to(self.device)
                gt_retf = batch['gt_retf'].to(self.device)
                input_magnitude = input_features[:, 0]
                
                pred_retf = self.model(input_features)
                loss = self.criterion(pred_retf, gt_retf, input_magnitude)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, epochs):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}:')
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            
            print(f'  Train Loss: {train_loss:.6f}')
            print(f'  Val Loss: {val_loss:.6f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch
                }, 'best_retf_model.pth')
                print('  Saved best model!')
            print()

# ML RETF Estimator Wrapper
class MLRETFEstimator:
    """Wrapper to integrate ML RETF estimator into ISCLP framework"""
    def __init__(self, model_path, n_fft=512, hop_length=256):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_freq_bins = n_fft // 2 + 1
        
        # Load trained model
        self.model = RETFEstimator(num_freq_bins=self.num_freq_bins)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def estimate_retf(self, multi_channel_audio):
        """Estimate RETF from multi-channel audio (replaces model-based estimation)"""
        # For now, using first channel as reference
        # You can extend this for multiple microphones
        audio = multi_channel_audio[:, 0] if len(multi_channel_audio.shape) > 1 else multi_channel_audio
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Prepare input features
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        input_features = np.stack([magnitude, phase], axis=0)
        input_tensor = torch.FloatTensor(input_features).unsqueeze(0)  # Add batch dimension
        
        # Predict RETF
        with torch.no_grad():
            retf_complex = self.model(input_tensor)
            retf_np = retf_complex.squeeze(0).numpy()
        
        return retf_np

# Main Function
def main():
    parser = argparse.ArgumentParser(description='Train ML-based RETF Estimator')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--clean_folder', type=str, default=CLEAN_FOLDER, help='Clean audio folder')
    parser.add_argument('--reverberant_folder', type=str, default=REVERBERANT_FOLDER, help='Reverberant audio folder')
    parser.add_argument('--rir_folder', type=str, default=RIR_FOLDER, help='RIR folder')
    args = parser.parse_args()
    
    # Create dataset
    dataset = RETFDataset(
        clean_folder=args.clean_folder,
        reverberant_folder=args.reverberant_folder,
        rir_folder=args.rir_folder,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    print(f"Found {len(dataset)} training examples")
    
    if len(dataset) == 0:
        print("Error: No training examples found. Check your folder paths and file structure.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2
    )
    
    # Create model
    model = RETFEstimator(num_freq_bins=num_freq_bins)
    
    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    trainer = RETFTrainer(model, train_loader, val_loader, device, lr=args.lr)
    trainer.train(epochs=args.epochs)

# Example usage function
def example_usage():
    """Example of how to use the trained model"""
    # Load trained model
    estimator = MLRETFEstimator('best_retf_model.pth')
    
    # Example: Load an audio file and estimate RETF
    audio, sr = sf.read('./data_project/distant/example.wav')
    retf = estimator.estimate_retf(audio)
    
    print(f"Estimated RETF shape: {retf.shape}")
    print("RETF estimation complete!")

if __name__ == "__main__":
    main()
    
    # Uncomment the line below to test the trained model
    # example_usage()
