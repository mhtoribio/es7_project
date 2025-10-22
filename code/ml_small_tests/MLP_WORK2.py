import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
from pathlib import Path
import soundfile as sf
import argparse
import glob

# Configuration
CLEAN_FOLDER = "./data_project/clean"
REVERBERANT_FOLDER = "./data_project/distant" 
RIR_FOLDER = "./data_project/rir"
n_fft = 512
hop_length = 256
num_freq_bins = n_fft // 2 + 1

# Dataset Class with flexible file matching
class RETFDataset(torch.utils.data.Dataset):
    def __init__(self, clean_folder, reverberant_folder, rir_folder, n_fft=512, hop_length=256):
        self.clean_folder = Path(clean_folder)
        self.reverberant_folder = Path(reverberant_folder) 
        self.rir_folder = Path(rir_folder)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_freq_bins = n_fft // 2 + 1
        
        print("Checking folder structure...")
        print(f"Clean folder exists: {self.clean_folder.exists()}")
        print(f"Reverberant folder exists: {self.reverberant_folder.exists()}")
        print(f"RIR folder exists: {self.rir_folder.exists()}")
        
        # Get all files in each folder
        clean_files = list(self.clean_folder.glob("*.*"))
        reverberant_files = list(self.reverberant_folder.glob("*.*"))
        rir_files = list(self.rir_folder.glob("*.*"))
        
        print(f"Found {len(clean_files)} files in clean folder")
        print(f"Found {len(reverberant_files)} files in reverberant folder") 
        print(f"Found {len(rir_files)} files in RIR folder")
        
        # Print first few files in each folder for debugging
        if clean_files:
            print(f"Sample clean files: {[f.name for f in clean_files[:3]]}")
        if reverberant_files:
            print(f"Sample reverberant files: {[f.name for f in reverberant_files[:3]]}")
        if rir_files:
            print(f"Sample RIR files: {[f.name for f in rir_files[:3]]}")
        
        # Create mapping - try different matching strategies
        self.file_pairs = self._find_matching_files(clean_files, reverberant_files, rir_files)
        
        if not self.file_pairs:
            print("No matching files found. Trying alternative matching strategies...")
            self.file_pairs = self._find_files_any_order(clean_files, reverberant_files, rir_files)
    
    def _find_matching_files(self, clean_files, reverberant_files, rir_files):
        """Strategy 1: Exact name matching"""
        file_pairs = []
        
        for clean_file in clean_files:
            base_name = clean_file.stem
            
            # Try to find matching files in other folders
            reverberant_match = None
            rir_match = None
            
            # Look for reverberant file
            for rev_file in reverberant_files:
                if rev_file.stem == base_name:
                    reverberant_match = rev_file
                    break
            
            # Look for RIR file  
            for rir_file in rir_files:
                if rir_file.stem == base_name:
                    rir_match = rir_file
                    break
            
            if reverberant_match and rir_match:
                file_pairs.append((clean_file, reverberant_match, rir_match))
        
        print(f"Found {len(file_pairs)} files with exact name matching")
        return file_pairs
    
    def _find_files_any_order(self, clean_files, reverberant_files, rir_files):
        """Strategy 2: Match any files in order when names don't match"""
        file_pairs = []
        
        # Just pair files by index if counts match
        min_count = min(len(clean_files), len(reverberant_files), len(rir_files))
        
        if min_count > 0:
            for i in range(min_count):
                file_pairs.append((clean_files[i], reverberant_files[i], rir_files[i]))
            print(f"Found {len(file_pairs)} files by index matching")
        else:
            print("Not enough files in all folders for index matching")
            
        return file_pairs
    
    def __len__(self):
        return len(self.file_pairs)
    
    def load_audio(self, file_path):
        try:
            audio, sr = sf.read(str(file_path))
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # Take first channel if multi-channel
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return silent audio as fallback
            return np.zeros(16000), 16000
    
    def compute_stft(self, audio):
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        return stft
    
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
        
        # Avoid division by zero
        if retf[0] == 0:
            retf[0] = 1e-10
            
        # Normalize relative to first microphone (paper constraint)
        retf = retf / retf[0]  # Relative to first "microphone"
        
        return retf[:self.num_freq_bins]  # Truncate to STFT frequency bins
    
    def __getitem__(self, idx):
        clean_file, reverberant_file, rir_file = self.file_pairs[idx]
        
        try:
            # Load audio files
            clean_audio, sr = self.load_audio(clean_file)
            reverberant_audio, sr = self.load_audio(reverberant_file)
            rir_audio, sr = self.load_audio(rir_file)
            
            # Ensure all audio is same length (take shortest)
            min_len = min(len(clean_audio), len(reverberant_audio), len(rir_audio))
            if min_len < n_fft:
                min_len = n_fft
                
            clean_audio = clean_audio[:min_len]
            reverberant_audio = reverberant_audio[:min_len] 
            rir_audio = rir_audio[:min_len]
            
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
                'file_name': f"{clean_file.stem}_{idx}"
            }
            
        except Exception as e:
            print(f"Error processing {clean_file.name}: {e}")
            # Return dummy data
            dummy_features = torch.zeros(2, num_freq_bins, 100)
            dummy_retf = torch.zeros(2, num_freq_bins)
            return {
                'input_features': dummy_features,
                'gt_retf': dummy_retf,
                'file_name': f"dummy_{idx}"
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

# Model Definition (same as before)
class RETFEstimator(nn.Module):
    def __init__(self, num_freq_bins=257, hidden_dim=256, num_layers=4):
        super(RETFEstimator, self).__init__()
        self.num_freq_bins = num_freq_bins
        
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((num_freq_bins, None))
        )
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        self.freq_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.retf_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_freq_bins * 2)
        )
        
    def forward(self, x):
        batch_size, _, num_freq, time_frames = x.shape
        
        encoded = self.freq_encoder(x)
        encoded = encoded.permute(0, 2, 3, 1)
        encoded = encoded.reshape(batch_size * num_freq, time_frames, 128)
        
        lstm_out, _ = self.lstm(encoded)
        attention_weights = self.freq_attention(lstm_out)
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        attended = attended.reshape(batch_size, num_freq, -1)
        
        retf_out = self.retf_decoder(attended)
        retf_out = retf_out.reshape(batch_size, num_freq, self.num_freq_bins, 2)
        
        retf_complex = torch.complex(retf_out[..., 0], retf_out[..., 1])
        retf_complex[:, :, 0] = 1.0 + 0j
        
        return retf_complex

class RETFLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.01):
        super(RETFLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred_retf, gt_retf, input_magnitude):
        complex_loss = F.mse_loss(pred_retf.real, gt_retf.real) + \
                      F.mse_loss(pred_retf.imag, gt_retf.imag)
        
        pred_magnitude = torch.abs(pred_retf)
        gt_magnitude = torch.abs(gt_retf)
        magnitude_loss = F.mse_loss(pred_magnitude, gt_magnitude)
        
        freq_diff = pred_retf[:, :, 1:] - pred_retf[:, :, :-1]
        smoothness_loss = torch.mean(torch.abs(freq_diff))
        
        total_loss = complex_loss + self.alpha * magnitude_loss + self.beta * smoothness_loss
        return total_loss

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
            
            input_magnitude = input_features[:, 0]
            
            self.optimizer.zero_grad()
            pred_retf = self.model(input_features)
            loss = self.criterion(pred_retf, gt_retf, input_magnitude)
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

class MLRETFEstimator:
    def __init__(self, model_path, n_fft=512, hop_length=256):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_freq_bins = n_fft // 2 + 1
        
        self.model = RETFEstimator(num_freq_bins=self.num_freq_bins)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def estimate_retf(self, multi_channel_audio):
        audio = multi_channel_audio[:, 0] if len(multi_channel_audio.shape) > 1 else multi_channel_audio
        
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        input_features = np.stack([magnitude, phase], axis=0)
        input_tensor = torch.FloatTensor(input_features).unsqueeze(0)
        
        with torch.no_grad():
            retf_complex = self.model(input_tensor)
            retf_np = retf_complex.squeeze(0).numpy()
        
        return retf_np

# Test function to check folder structure
def test_folder_structure():
    print("=== Testing Folder Structure ===")
    
    clean_folder = Path(CLEAN_FOLDER)
    reverberant_folder = Path(REVERBERANT_FOLDER)
    rir_folder = Path(RIR_FOLDER)
    
    print(f"Clean folder: {clean_folder.absolute()}")
    print(f"Reverberant folder: {reverberant_folder.absolute()}")
    print(f"RIR folder: {rir_folder.absolute()}")
    
    # Check if folders exist
    for folder, name in [(clean_folder, "Clean"), (reverberant_folder, "Reverberant"), (rir_folder, "RIR")]:
        if folder.exists():
            files = list(folder.glob("*.*"))
            print(f"{name} folder: {len(files)} files")
            if files:
                print(f"  Sample files: {[f.name for f in files[:5]]}")
        else:
            print(f"{name} folder does not exist!")
    
    # Try to create dataset
    print("\n=== Creating Dataset ===")
    dataset = RETFDataset(CLEAN_FOLDER, REVERBERANT_FOLDER, RIR_FOLDER)
    print(f"Dataset size: {len(dataset)}")
    
    return len(dataset) > 0

def main():
    # First test the folder structure
    if not test_folder_structure():
        print("\n❌ Folder structure issue detected!")
        print("Please check:")
        print(f"1. Clean folder exists and has files: {CLEAN_FOLDER}")
        print(f"2. Reverberant folder exists and has files: {REVERBERANT_FOLDER}")
        print(f"3. RIR folder exists and has files: {RIR_FOLDER}")
        print("4. Files have matching names across folders")
        return
    
    parser = argparse.ArgumentParser(description='Train ML-based RETF Estimator')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    # Create dataset
    dataset = RETFDataset(
        clean_folder=CLEAN_FOLDER,
        reverberant_folder=REVERBERANT_FOLDER,
        rir_folder=RIR_FOLDER,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    print(f"Final dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No training examples found. Cannot proceed with training.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    
    # Create model
    model = RETFEstimator(num_freq_bins=num_freq_bins)
    
    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    trainer = RETFTrainer(model, train_loader, val_loader, device, lr=args.lr)
    trainer.train(epochs=args.epochs)

if __name__ == "__main__":
    main()
