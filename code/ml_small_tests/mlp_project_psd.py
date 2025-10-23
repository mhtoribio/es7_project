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

class PSDEstimator(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, output_size=257):
        """
        MLP for estimating PSD from reverberant speech
        """
        super(PSDEstimator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_size // 2, output_size),
            nn.Softplus()  # PSD must be non-negative
        )
        
    def forward(self, x):
        return self.network(x)

def resample_audio(audio, original_sr, target_sr=16000):
    """Resample audio to target sample rate"""
    if original_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / original_sr)
    return signal.resample(audio, num_samples)

def compute_stft(audio, n_fft=512, hop_length=256, sr=16000):
    """Compute STFT of audio signal"""
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    f, t, stft = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft-hop_length)
    return f, t, stft

def extract_early_rir(rir, early_samples=512):
    """Extract early part of RIR"""
    if len(rir.shape) > 1:
        return rir[:early_samples, :]
    else:
        return rir[:early_samples]

def compute_true_psd(clean_speech, rir_early, n_fft=512, sr=16000):
    """Compute true PSD using the same method as the STFT features"""
    if len(rir_early.shape) > 1:
        rir_mono = rir_early[:, 0]
    else:
        rir_mono = rir_early
        
    early_speech = signal.convolve(clean_speech, rir_mono, mode='same')
    
    # Compute STFT of early speech (same method as reverberant speech)
    f, t, stft_early = signal.stft(early_speech, fs=sr, nperseg=n_fft, noverlap=n_fft-256)
    
    # Compute PSD as |STFT|^2 (periodogram estimate)
    psd = np.mean(np.abs(stft_early)**2, axis=1)  # Average over time
    
    return f, psd

def extract_seat_number(filename):
    """Extract seat number from filename"""
    patterns = [r'_seat(\d+)', r'seat(\d+)', r'_s(\d+)', r's(\d+)']
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    return None

def prepare_psd_data(clean_folder, reverberant_folder, rir_folder, n_fft=512, max_files=100, target_sr=16000):
    """Prepare training data specifically for PSD estimation"""
    
    clean_files = [f for f in os.listdir(clean_folder) if f.endswith('.wav')]
    data_pairs = []
    
    print(f"Found {len(clean_files)} clean files")
    
    for clean_file in clean_files[:max_files]:
        try:
            # Load clean speech
            clean_path = os.path.join(clean_folder, clean_file)
            clean_audio, clean_sr = sf.read(clean_path)
            
            if clean_sr != target_sr:
                clean_audio = resample_audio(clean_audio, clean_sr, target_sr)
            
            if len(clean_audio.shape) > 1:
                clean_audio = clean_audio[:, 0]
            
            # Find corresponding reverberant files
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
                    
                    # Extract seat number
                    seat_match = extract_seat_number(reverberant_file)
                    if seat_match is None:
                        continue
                    
                    # Find RIR file
                    rir_patterns = [f"seat{int(seat_match):02d}.npy", f"seat{seat_match}.npy"]
                    rir_data = None
                    for pattern in rir_patterns:
                        rir_path = os.path.join(rir_folder, pattern)
                        if os.path.exists(rir_path):
                            rir_data = np.load(rir_path)
                            break
                    
                    if rir_data is None:
                        continue
                    
                    # Extract early RIR
                    rir_early = extract_early_rir(rir_data)
                    
                    # Compute STFT of reverberant speech
                    f, t, stft_reverberant = compute_stft(reverberant_audio, n_fft=n_fft, sr=target_sr)
                    
                    # Compute true PSD
                    f_psd, psd_true = compute_true_psd(clean_audio, rir_early, n_fft=n_fft, sr=target_sr)
                    
                    # Extract magnitude and phase from STFT
                    stft_magnitude = np.abs(stft_reverberant).T
                    stft_phase = np.angle(stft_reverberant).T
                    
                    # Normalize phase to [-1, 1]
                    stft_phase_normalized = stft_phase / np.pi
                    
                    # Concatenate features
                    stft_features = np.concatenate([stft_magnitude, stft_phase_normalized], axis=1)
                    
                    data_pairs.append({
                        'features': stft_features,
                        'psd_target': psd_true,
                        'clean_file': clean_file,
                        'reverberant_file': reverberant_file
                    })
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
    
    print(f"Prepared {len(data_pairs)} PSD training pairs")
    return data_pairs

def create_psd_dataset(data_pairs, n_fft=512, max_frames_per_file=50):
    """Create PyTorch dataset for PSD training"""
    
    X = []
    y_psd = []
    
    for pair in data_pairs:
        features = pair['features']
        psd_target = pair['psd_target']
        
        num_frames = min(features.shape[0], max_frames_per_file)
        features = features[:num_frames]
        
        for i in range(num_frames):
            X.append(features[i])
            y_psd.append(psd_target)
    
    X_tensor = torch.FloatTensor(np.array(X))
    y_psd_tensor = torch.FloatTensor(np.array(y_psd))
    
    print(f"PSD Dataset - Features: {X_tensor.shape}, Targets: {y_psd_tensor.shape}")
    
    return X_tensor, y_psd_tensor

def train_psd_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train the PSD estimation model with validation"""
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    print("Training PSD model...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
    
    return train_losses, val_losses

# ==================== TESTING FUNCTIONS ====================

def test_model_sanity(model, X_test, y_test):
    """Basic sanity checks for the model"""
    
    print("Running model sanity checks...")
    
    # Test 1: Model can make predictions without errors
    try:
        with torch.no_grad():
            predictions = model(X_test[:10])  # Test on small batch
        print("✓ Model can make predictions")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False
    
    # Test 2: Output shape matches expected
    expected_shape = (10, y_test.shape[1])  # (batch_size, freq_bins)
    if predictions.shape == expected_shape:
        print(f"✓ Output shape correct: {predictions.shape}")
    else:
        print(f"✗ Output shape wrong. Expected {expected_shape}, got {predictions.shape}")
        return False
    
    # Test 3: Outputs are non-negative (PSD requirement)
    if torch.all(predictions >= 0):
        print("✓ All outputs are non-negative")
    else:
        print("✗ Some outputs are negative")
        return False
    
    return True

def evaluate_model_performance(model, X_test, y_test):
    """Comprehensive model evaluation"""
    
    print("Evaluating model performance...")
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    # Calculate various metrics
    mse = nn.MSELoss()(predictions, y_test).item()
    mae = nn.L1Loss()(predictions, y_test).item()
    
    # Relative metrics
    relative_mse = (mse / (y_test.std().item() ** 2)) * 100
    signal_to_noise = 10 * torch.log10(torch.mean(y_test ** 2) / torch.mean((predictions - y_test) ** 2)).item()
    
    print(f"Performance Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Relative MSE: {relative_mse:.2f}%")
    print(f"  Signal-to-Noise Ratio: {signal_to_noise:.2f} dB")
    
    # Check if predictions are reasonable
    pred_mean = predictions.mean().item()
    target_mean = y_test.mean().item()
    ratio = pred_mean / target_mean if target_mean > 0 else 0
    
    print(f"  Prediction mean: {pred_mean:.4f}")
    print(f"  Target mean: {target_mean:.4f}")
    print(f"  Ratio: {ratio:.4f}")
    
    return {
        'mse': mse, 
        'mae': mae, 
        'relative_mse': relative_mse,
        'snr': signal_to_noise,
        'prediction_mean': pred_mean,
        'target_mean': target_mean
    }

def visualize_predictions(model, X_test, y_test, num_examples=3):
    """Visual comparison of predicted vs true PSD"""
    
    print("Generating prediction visualizations...")
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    # Convert to numpy for plotting
    pred_np = predictions.numpy()
    target_np = y_test.numpy()
    
    # Select random examples
    indices = np.random.choice(len(X_test), num_examples, replace=False)
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3*num_examples))
    
    if num_examples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        axes[i].plot(target_np[idx], 'b-', label='True PSD', alpha=0.7, linewidth=2)
        axes[i].plot(pred_np[idx], 'r--', label='Predicted PSD', alpha=0.7, linewidth=2)
        axes[i].set_xlabel('Frequency Bin')
        axes[i].set_ylabel('PSD')
        axes[i].set_title(f'Example {i+1} (Sample {idx})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('psd_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot correlation scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(target_np.flatten(), pred_np.flatten(), alpha=0.3, s=1)
    max_val = max(target_np.max(), pred_np.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal', linewidth=2)
    plt.xlabel('True PSD')
    plt.ylabel('Predicted PSD')
    plt.title('Predicted vs True PSD (All Test Samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('psd_scatter.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return indices, pred_np, target_np

def comprehensive_model_test(model, X_test, y_test):
    """Run all tests on the trained model"""
    
    print("\n" + "="*50)
    print("RUNNING COMPREHENSIVE MODEL TESTS")
    print("="*50)
    
    # Test 1: Basic sanity
    print("\n1. BASIC SANITY CHECKS:")
    sanity_ok = test_model_sanity(model, X_test, y_test)
    
    # Test 2: Performance metrics
    print("\n2. PERFORMANCE EVALUATION:")
    metrics = evaluate_model_performance(model, X_test, y_test)
    
    # Test 3: Visual inspection
    print("\n3. VISUAL INSPECTION:")
    indices, pred_np, target_np = visualize_predictions(model, X_test, y_test, num_examples=3)
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Basic Sanity: {'PASS' if sanity_ok else 'FAIL'}")
    print(f"Test MSE: {metrics['mse']:.6f}")
    print(f"Test MAE: {metrics['mae']:.6f}")
    print(f"Relative MSE: {metrics['relative_mse']:.2f}%")
    print(f"SNR: {metrics['snr']:.2f} dB")
    
    # Rough benchmarks
    if metrics['relative_mse'] < 50:  # Less than 50% relative error
        print("✓ Performance is reasonable")
    else:
        print("✗ Performance needs improvement")
    
    if metrics['snr'] > 0:  # Positive SNR means better than random
        print("✓ SNR is positive (better than random)")
    else:
        print("✗ SNR is negative (worse than random)")

def main():
    """Main function for PSD training and testing"""
    # Configuration
    CLEAN_FOLDER = "./data_project/clean"
    REVERBERANT_FOLDER = "./data_project/distant" 
    RIR_FOLDER = "./data_project/rir"
    TARGET_SR = 16000
    
    n_fft = 512
    num_freq_bins = n_fft // 2 + 1
    input_size = 2 * num_freq_bins  # magnitude + phase
    
    print("=== PSD MODEL TRAINING ===")
    
    # Prepare data
    data_pairs = prepare_psd_data(
        CLEAN_FOLDER, REVERBERANT_FOLDER, RIR_FOLDER, n_fft, max_files=100, target_sr=TARGET_SR
    )
    
    if not data_pairs:
        print("No PSD training data found!")
        return
    
    # Split data into train and test sets
    train_pairs, test_pairs = train_test_split(
        data_pairs, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training samples: {len(train_pairs)}")
    print(f"Testing samples: {len(test_pairs)}")
    
    # Create datasets
    X_train, y_train = create_psd_dataset(train_pairs, n_fft)
    X_test, y_test = create_psd_dataset(test_pairs, n_fft)
    
    # Calculate output size
    psd_output_size = y_train.shape[1]
    
    print(f"PSD Model - Input: {input_size}, Output: {psd_output_size}")
    print(f"Training data: {X_train.shape[0]} frames")
    print(f"Testing data: {X_test.shape[0]} frames")
    
    # Initialize model
    model = PSDEstimator(
        input_size=input_size,
        hidden_size=256,
        output_size=psd_output_size
    )
    
    # Train model with validation
    train_losses, val_losses = train_psd_model(
        model, X_train, y_train, X_test, y_test, epochs=100
    )
    
    # Save model
    torch.save(model.state_dict(), "psd_model.pth")
    print("PSD model saved as 'psd_model.pth'")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('PSD Model Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig('psd_training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ==================== TESTING PHASE ====================
    print("\n" + "="*50)
    print("MODEL TESTING PHASE")
    print("="*50)
    
    # Run comprehensive tests on the test set
    comprehensive_model_test(model, X_test, y_test)

    print(f"Input features range: [{X_train.min():.6f}, {X_train.max():.6f}]")
    print(f"Target PSD range: [{y_train.min():.6f}, {y_train.max():.6f}]")
    print(f"Target PSD mean: {y_train.mean():.6f}, std: {y_train.std():.6f}")

if __name__ == "__main__":
    main()