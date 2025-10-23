import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split

# ================= PSD MODEL =================
class PSDEstimator(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, output_size=257):
        super(PSDEstimator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Softplus()
        )
    def forward(self, x):
        return self.network(x)

# ================= AUDIO UTILITIES =================
def resample_audio(audio, original_sr, target_sr=16000):
    if original_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / original_sr)
    return signal.resample(audio, num_samples)

def compute_stft(audio, n_fft=512, hop_length=256, sr=16000):
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    f, t, stft = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    return f, t, stft

def extract_early_rir(rir, early_samples=512):
    if len(rir.shape) > 1:
        return rir[:early_samples, :]
    else:
        return rir[:early_samples]

def compute_true_psd(clean_speech, rir_early, n_fft=512, sr=16000):
    if len(rir_early.shape) > 1:
        rir_mono = rir_early[:, 0]
    else:
        rir_mono = rir_early
    early_speech = signal.convolve(clean_speech, rir_mono, mode='same')
    f, psd = signal.welch(early_speech, fs=sr, nperseg=n_fft)
    return f, psd

def extract_seat_number(filename):
    patterns = [r'_seat(\d+)', r'seat(\d+)', r'_s(\d+)', r's(\d+)']
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    return None

# ================= DATA PREPARATION =================
def prepare_psd_data(clean_folder, reverberant_folder, rir_folder, n_fft=512, max_files=100, target_sr=16000):
    clean_files = [f for f in os.listdir(clean_folder) if f.endswith('.wav')]
    data_pairs = []
    print(f"Found {len(clean_files)} clean files")
    
    for clean_file in clean_files[:max_files]:
        try:
            clean_path = os.path.join(clean_folder, clean_file)
            clean_audio, clean_sr = sf.read(clean_path)
            if clean_sr != target_sr:
                clean_audio = resample_audio(clean_audio, clean_sr, target_sr)
            if len(clean_audio.shape) > 1:
                clean_audio = clean_audio[:, 0]
            
            base_name = os.path.splitext(clean_file)[0]
            reverberant_files = [f for f in os.listdir(reverberant_folder) 
                                 if f.startswith(base_name) and f.endswith('.wav')]
            if not reverberant_files:
                continue
            
            for reverberant_file in reverberant_files:
                try:
                    reverberant_path = os.path.join(reverberant_folder, reverberant_file)
                    reverberant_audio, rev_sr = sf.read(reverberant_path)
                    if rev_sr != target_sr:
                        reverberant_audio = resample_audio(reverberant_audio, rev_sr, target_sr)
                    if len(reverberant_audio.shape) > 1:
                        reverberant_audio = reverberant_audio[:, 0]
                    
                    seat_match = extract_seat_number(reverberant_file)
                    if seat_match is None:
                        continue
                    
                    rir_patterns = [f"seat{int(seat_match):02d}.npy", f"seat{seat_match}.npy"]
                    rir_data = None
                    for pattern in rir_patterns:
                        rir_path = os.path.join(rir_folder, pattern)
                        if os.path.exists(rir_path):
                            rir_data = np.load(rir_path)
                            break
                    if rir_data is None:
                        continue
                    
                    rir_early = extract_early_rir(rir_data)
                    f, t, stft_reverberant = compute_stft(reverberant_audio, n_fft=n_fft, sr=target_sr)
                    f_psd, psd_true = compute_true_psd(clean_audio, rir_early, n_fft=n_fft, sr=target_sr)
                    
                    stft_magnitude = np.abs(stft_reverberant).T
                    stft_phase = np.angle(stft_reverberant).T
                    stft_phase_normalized = stft_phase / np.pi
                    stft_features = np.concatenate([stft_magnitude, stft_phase_normalized], axis=1)
                    
                    data_pairs.append({
                        'features': stft_features,
                        'psd_target': psd_true,
                        'clean_file': clean_file,
                        'reverberant_file': reverberant_file
                    })
                except Exception:
                    continue
        except Exception:
            continue
    
    print(f"Prepared {len(data_pairs)} PSD training pairs")
    return data_pairs

# ================= DATASET WITH FRAME MAPPING =================
def create_psd_dataset_with_mapping(data_pairs, n_fft=512, max_frames_per_file=50):
    X_list = []
    y_list = []
    file_names = []
    for pair in data_pairs:
        features = pair['features']
        psd_target = pair['psd_target']
        num_frames = min(features.shape[0], max_frames_per_file)
        features = features[:num_frames]
        psd_target = np.tile(psd_target, (num_frames, 1))
        X_list.append(torch.FloatTensor(features))
        y_list.append(torch.FloatTensor(psd_target))
        file_names.append(pair['reverberant_file'])
    print(f"Prepared dataset with {len(X_list)} files (frame mapping preserved)")
    return X_list, y_list, file_names

def flatten_file_list(X_list, y_list):
    X_all = torch.cat(X_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    return X_all, y_all

# ================= TRAINING =================
def train_psd_model(model, X, y_psd, epochs=100, batch_size=32):
    dataset = torch.utils.data.TensorDataset(X, y_psd)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    losses = []
    print("Training PSD model...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}, PSD Loss: {avg_loss:.6f}")
    return losses

# ================= PLOTTING =================
def plot_psd_comparison(true_psd, estimated_psd, file_name, frame_idx=0):
    plt.figure(figsize=(10,6))
    plt.plot(true_psd[frame_idx], label='Ground Truth PSD')
    plt.plot(estimated_psd[frame_idx], label='Estimated PSD')
    plt.title(f'PSD Comparison for {file_name}, Frame {frame_idx}')
    plt.xlabel('Frequency Bin')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid(True)
    plt.show()

# ================= MAIN =================
def main():
    CLEAN_FOLDER = "./data_project/clean"
    REVERBERANT_FOLDER = "./data_project/distant" 
    RIR_FOLDER = "./data_project/rir"
    TARGET_SR = 16000
    n_fft = 512
    num_freq_bins = n_fft // 2 + 1
    input_size = 2 * num_freq_bins

    print("=== PSD MODEL TRAINING ===")
    
    data_pairs = prepare_psd_data(
        CLEAN_FOLDER, REVERBERANT_FOLDER, RIR_FOLDER, n_fft, max_files=100, target_sr=TARGET_SR
    )
    
    if not data_pairs:
        print("No PSD training data found!")
        return

    # Split into train/test
    train_pairs, test_pairs = train_test_split(data_pairs, test_size=0.1, random_state=42)
    
    # Create datasets with frame mapping
    X_train_list, y_train_list, train_files = create_psd_dataset_with_mapping(train_pairs)
    X_test_list, y_test_list, test_files = create_psd_dataset_with_mapping(test_pairs)
    
    # Flatten frames for training
    X_train, y_train = flatten_file_list(X_train_list, y_train_list)
    
    # Initialize model
    psd_output_size = y_train.shape[1]
    model = PSDEstimator(input_size=input_size, hidden_size=256, output_size=psd_output_size)
    
    # Train model
    losses = train_psd_model(model, X_train, y_train, epochs=100)
    
    # Save model
    torch.save(model.state_dict(), "psd_model.pth")
    print("PSD model saved as 'psd_model.pth'")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('PSD Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.yscale('log')
    plt.show()
    
    # Evaluate on test set and plot PSD comparisons
    model.eval()
    with torch.no_grad():
        for i, (X_file, y_file, file_name) in enumerate(zip(X_test_list, y_test_list, test_files)):
            y_pred = model(X_file).numpy()
            y_true = y_file.numpy()
            plot_psd_comparison(y_true, y_pred, file_name, frame_idx=0)
            if i >= 2:  # plot only first 3 files
                break

if __name__ == "__main__":
    main()
