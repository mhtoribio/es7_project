#from seadge import config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from scipy import signal
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Add GPU (Device to this when it works on CPU)

class TemporalBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()

        # First 1D convolution layer
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        # Second 1D convolution layer
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # If the input and output channels differ, use a 1x1 conv to match dimensions for residual connection
        self.downsample = nn.Conv1d(input_size, output_size, 1) if input_size != output_size else None

        # Initialize weights properly (important for stability)
        self.init_weights()

    def init_weights(self):
        # Use Kaiming (He) initialization for ReLU activations
        for conv in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x):
        """
        x: (batch, input_size, seq_len)
        output: (batch, output_size, seq_len)
        """
        out = F.relu(self.conv1(x))
        out = out[:, :, :-self.conv1.padding[0]]    # trim right side for making causal
        out = self.dropout(out)

        out = F.relu(self.conv2(out))
        out = out[:, :, :-self.conv2.padding[0]]    # trim right side for making causal
        out = self.dropout(out)

        # Residual (shortcut) connection
        res = x if self.downsample is None else self.downsample(x)

        # Combine residual and conv outputs
        return F.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_channels, output_channels, num_channels, kernel_size=3, dropout=0.2):
        """
        num_channels: list defining hidden channel sizes, e.g. [64, 128, 256]
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_ch = input_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i  # Exponential dilation increases receptive field
            padding = (kernel_size - 1) * dilation

            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size,
                              stride=1, dilation=dilation, padding=padding, dropout=dropout)
            )

        self.network = nn.Sequential(*layers)

        # 1x1 conv projects to PSD bins (output per frame)
        self.output_layer = nn.Conv1d(num_channels[-1], output_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (batch, input_channels, seq_len)
        output: (batch, output_channels, seq_len)
        """
        y = self.network(x)
        y = F.softplus(self.output_layer(y))  # non-negative PSD
        return y

# Load clean, distant and RIR
CLEAN_FOLDER = r"C:\UNI\7.Semester\mlp_data_rir\clean_speech\resampled"
DISTANT_FOLDER = r"C:\UNI\7.Semester\mlp_data_rir\distant_speech" 
RIR_FOLDER = r"C:\UNI\7.Semester\mlp_data_rir\rir_static"

def prepare_files(clean_folder, distant_folder, rir_folder, max_files):
    # Folder to store resampled clean files
    resampled_clean_folder = os.path.join(clean_folder, "resampled")
    os.makedirs(resampled_clean_folder, exist_ok=True)


    distant_files = [f for f in os.listdir(distant_folder) if f.endswith('.wav')] # Makes a list of all files in the given folder
    file_pairs = [] #Python list

    for distant_file in distant_files[:max_files]:
        try:
            # Load clean speech
            distant_path = os.path.join(distant_folder, distant_file)
            
            # Remove "_seatXX" from distant filename
            base_name = distant_file.split('_seat')[0]  # removes everything after "_seat"
            
            # Construct corresponding clean filename
            clean_file = f"{base_name}.wav"
            clean_path = os.path.join(clean_folder, clean_file)

            # Check if clean file exists
            if not os.path.exists(clean_path):
                print(f"⚠️ No clean match found for {distant_file}")
                continue
            
            distant_fs, _ = wavfile.read(distant_path)
            clean_fs, clean_wav = wavfile.read(clean_path)

            # Check if clean and distant have same fs
            if clean_fs != distant_fs:
                print(f"⚠️ Skipping {clean_file} due to mismatched sampling rate")
                continue  # skip this pair

            file_pairs.append((clean_path, distant_path))

        except Exception as e:
            print(f"Error processing {distant_file}: {e}")

    return file_pairs

def ground_truth_psd(clean_speech, n_fft=512, fs=16000):
    """Compute true PSD"""
    if len(clean_speech.shape) > 1:
        clean_speech = clean_speech[:, 0]
    
    # Compute STFT of early speech (same method as reverberant speech)
    f, _, stft_clean = signal.stft(clean_speech, fs=fs, nperseg=n_fft, noverlap=n_fft*0.5)
    
    # Compute ground truth: PSD = |STFT_clean|^2
    true_psd = np.abs(stft_clean)**2
    
    return f, true_psd

def compute_stft_distant(distant_speech, n_fft=512, fs=16000):
    """Compute stft for distant"""
    if len(distant_speech.shape) > 1:
        distant_speech = distant_speech[:, 0]
    
    # Compute STFT of early speech (same method as reverberant speech)
    f, _, stft_distant = signal.stft(distant_speech, fs=fs, nperseg=n_fft, noverlap=n_fft*0.5)

    return f, stft_distant

def pair_features_PSD_True(clean_speech,distant_speech):
    f_true_psd, true_psd =ground_truth_psd(clean_speech, n_fft=512, fs=16000)
    f, stft_distant =compute_stft_distant(distant_speech, n_fft=512, fs=16000)

    pair_true_psd_stft = [] 

    # Num_frames desired to look at assumed clean shorter signal then distant
    # As distant has reverb
    #num_frames = true_psd.shape[1]
    num_frames = min(stft_distant.shape[1], true_psd.shape[1])

    for i in range(num_frames):
        # Keep in mind features may need to be T (tranposed)

        # Pair Mag,Phase with PSD of the same frame
        magnitude_feature = np.abs(stft_distant[:, i])
        phase_feature = np.angle(stft_distant[:, i])
        psd_frame = true_psd[:, i]
        pair_true_psd_stft.append({
            'mag_feature': magnitude_feature,
            'phase_feature': phase_feature,
            'psd_target': psd_frame
        })

    return pair_true_psd_stft


def compute_tensor_for_pytorch(all_pairs,seq_len = 30):
    # Input and desired output lists
    input_features = []
    true_PSDs = []

    for pair in all_pairs:
        mag = pair['mag_feature']
        phase = pair['phase_feature']
        psd = pair['psd_target']

        # Combine mag and phase into a single feature vector
        feature = np.concatenate((mag, phase))

        # Normalize PSD for network stability
        #psd_norm = np.log1p(psd)  # log(1 + PSD)

        # Appends new pairs to the list
        input_features.append(feature)   # e.g. STFT mag + phase for that frame
        true_PSDs.append(psd)            # corresponding PSD for that same frame
    
    # Convert from python list of numpy arrays to one contiguous numpy arrays
    input_features = np.array(input_features)  # shape: (num_frames, num_features)
    true_PSDs = np.array(true_PSDs)            # shape: (num_frames, num_bins)

    # --- Group into overlapping sequences ---
    x_seq = []
    y_seq = []

    for i in range(len(input_features) - seq_len + 1):
        # Each sample: seq_len consecutive frames
        x_window = input_features[i:i + seq_len].T     # shape: (features, seq_len)
        y_window = true_PSDs[i:i + seq_len].T          # shape: (bins, seq_len)
        x_seq.append(x_window)
        y_seq.append(y_window)
    
    # Convert to tensors
    # implicitly paired by their index position in your tensors.
    x_tensor = torch.FloatTensor(np.array(x_seq))  # (batch, features, seq_len)
    y_tensor = torch.FloatTensor(np.array(y_seq))  # (batch, bins, seq_len)

    return x_tensor, y_tensor


# CREATE TRAIN MODEL AND HAPPY HOPEFULLY
def train_psd_model(model, x_tensor, y_tensor, epochs, batch_size):
    """Train the PSD estimation model"""

    # 80 % training data and 20 % test data
    x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.2) 

    # Create dataset and batches
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.0001) # maybe add weight_decay
    # Scheduler is used to adjust learning rate dynamically
    # If validation loss hasn’t improved in 10 epochs, the learning rate halves.
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss() # For regression (MSELoss()), for classification (CrossEntropyLoss())

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred_psd = model(batch_x)
            #loss = criterion(pred_psd, batch_y)

            # Use this when using trimming for causal convolution
            # Note: Padding is applied to each STFT sequence so that the output length matches the input length.
            #       This means the first and last few frames of each sequence are influenced by zeros (not real STFT data).
            min_len = min(pred_psd.shape[2], batch_y.shape[2])
            loss = criterion(pred_psd[:, :, :min_len], batch_y[:, :, :min_len])

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Used for clipping gradient avoid exploding gradient
            optimizer.step()

            epoch_train_loss += loss.item() # Used for plotting

        # Following is used for plotting average loss per epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:3d} and avg. loss: {avg_train_loss:.6f}')

    # Testing phase (out of the epoch for loop, only evaluating after all epochs)
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test)
        #pred_psd_norm = model(x_test)            # network output in log-scale
        #test_pred = torch.expm1(pred_psd_norm)   # convert back to original PSD
        #y_test = torch.expm1(y_test)
        test_loss = criterion(test_pred, y_test).item()

    return train_losses, test_loss


def debug_file_pairs():
    print("🔍 Starting debug for PSD tensor creation...\n")

    # Step 1: Prepare pairs (file paths)
    data_pairs = prepare_files(CLEAN_FOLDER, DISTANT_FOLDER, RIR_FOLDER, max_files=8000)
    print(f"✅ Found {len(data_pairs)} valid clean/distant pairs.\n")

    all_pairs = []

    # Step 2: Loop through the first few file pairs
    for i, (clean_resampled_file, distant_file) in enumerate(data_pairs[:3]):
        print(f"▶️ Processing Pair {i+1}:")
        print(f" Clean:   {clean_resampled_file}")
        print(f" Distant: {distant_file}")

        _, clean_wav = wavfile.read(clean_resampled_file)
        _, distant_wav = wavfile.read(distant_file)
        # Compute frame-wise features and PSD targets
        frame_pairs = pair_features_PSD_True(clean_wav, distant_wav)
        all_pairs.extend(frame_pairs)

        # Debug print for first frame of this file
        first_frame = frame_pairs[0]
        print(f"  Frames in this file: {len(frame_pairs)}")
        print(f"  Magnitude feature shape: {first_frame['mag_feature'].shape}")
        print(f"  Phase feature shape:     {first_frame['phase_feature'].shape}")
        print(f"  PSD target shape:        {first_frame['psd_target'].shape}")
        print("-" * 60)

    # Step 3: Convert all frame pairs to PyTorch tensors
    x_tensor, y_tensor = compute_tensor_for_pytorch(all_pairs,seq_len=50)

    # Step 4: Final sanity checks
    print("\n✅ Final Tensor Summary:")
    print(f"  Input tensor shape:  {x_tensor.shape}  # (batch, channels, seq_len)")
    print(f"  Target tensor shape: {y_tensor.shape}  # (batch, psd_bins, seq_len)")
    print(f"  Number of sequences: {x_tensor.shape[0]}")
    print(f"  Input channels:      {x_tensor.shape[1]}")
    print(f"  PSD bins per frame:  {y_tensor.shape[1]}")
    print(f"  Sequence length:     {x_tensor.shape[2]}")
    print("=" * 60)

    # Step 5: Define input/output sizes
    input_size = x_tensor.shape[1]
    output_size = y_tensor.shape[1]
    print(f'input size: {input_size}, output size: {output_size}')

    # Step 6: Initialize model
    model = TCN(
        input_size=input_size,
        output_size=output_size,
        num_channels=[256, 256, 256],  # depth of network
        kernel_size=3,
        dropout=0.2
    )

    # Step 7: Train model and get losses
    train_losses, test_loss = train_psd_model(model, x_tensor, y_tensor, epochs=100, batch_size=32)
    print(f"\nFinal test loss: {test_loss:.6f}")

    # Step 8: Plot training losses
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.title("PSD Estimator Training Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    data_pairs = prepare_files(CLEAN_FOLDER, DISTANT_FOLDER, RIR_FOLDER, max_files=100)
    all_pairs = []
  
    # iterate over the data pairs
    for clean_resampled_file, distant_file in data_pairs:
      
        # Mutiple STFT Features paired with PSD for one data pair'
        # Meaning one wave file = STFT Amount of feature pairs
        # THIS probally alot of data keep in mind
        # 10 s / 32 ms = feature pairs for one wave
        frame_pairs = pair_features_PSD_True(clean_resampled_file, distant_file)
        all_pairs.extend(frame_pairs)  # add all frames to global list
    x_tensor, y_psd_tensor = compute_tensor_for_pytorch(all_pairs,seq_len=50)

    input_size = x_tensor.shape[1]
    output_size = y_psd_tensor.shape[1]
    print(f'input size: {input_size}, output size: {output_size}')

    # Initialize model
    model = TCN(
        input_size=input_size,
        output_size=output_size,
        num_channels=[256, 256, 256],  # depth of network
        kernel_size=3,
        dropout=0.2
    )

    train_losses, test_loss = train_psd_model(model, x_tensor, y_psd_tensor, epochs=10, batch_size=32)
    print(f"\nFinal test loss: {test_loss:.6f}")

    # Step 5: Save model
    torch.save(model.state_dict(), "psd_model.pth")
    print("✅ PSD model saved as 'psd_model.pth'")

    # Step 6: Plot training losses
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.title("PSD Estimator Training Loss")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    debug_file_pairs()
