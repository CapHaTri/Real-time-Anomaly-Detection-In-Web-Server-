import streamlit as st
from pinotdb import connect
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import integrate
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
class Custom_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        pass
    def __len__(self):
        pass
    def __getitem(self):
        pass
    def MinMaxScaler(self, x, dim=-2):
        # Calculate the minimum and maximum values along the specified dimension
        min_val, _ = torch.min(x, dim=dim, keepdim=True)
        max_val, _ = torch.max(x, dim=dim, keepdim=True)

        # Scale the tensor to the range [0, 1]
        scaled_x = (x - min_val) / (max_val - min_val)

        return scaled_x
    def Detrend(self, x):
        x_copy = x.detach().clone()
        x_copy = torch.cat((torch.zeros_like(x_copy[0,:]).unsqueeze(-2), x_copy[:-1,:]), dim=-2)
        return x - x_copy
    
    def Fill_nan(self, x):
        median_val = np.nanmedian(x)

        x_filled = np.where(np.isnan(x), median_val, x)

        return x_filled
    
class AER(nn.Module):
    def __init__(self, input_dim, hidden_dim=60):
        super(AER, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoding
        _, (hidden, _) = self.encoder(x)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)  # Concatenate hidden states from both directions
        hidden = self.relu(hidden)

        # Decoding with +2 time steps
        hidden = hidden.unsqueeze(1).repeat(1, x.size(1) + 2, 1)  # Repeat hidden state for the length of input + 2
        decoder_output, _ = self.decoder(hidden)
        decoder_output = self.relu(decoder_output)
        fc_output = self.fc(decoder_output)
        # Predictions
        recon = fc_output[:, 1:-1]  # Reconstruction
        pred_fwd = fc_output[:, -1]  # Forward prediction
        pred_rev = fc_output[:, 0] # Reverse prediction

        return recon, pred_fwd, pred_rev
    
    
def regression_errors(y, y_hat, smoothing_window=0.01, smooth=True,
                      masking_window=0.01, mask=False):
    """Compute an array of absolute errors comparing predictions and expected output.
    If smooth is True, apply EWMA to the resulting array of errors.
    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values.
        smoothing_window (float):
            Optional. Size of the smoothing window, expressed as a proportion of the total
            length of y. If not given, 0.01 is used.
        smooth (bool):
            Optional. Indicates whether the returned errors should be smoothed with EWMA.
            If not given, `True` is used.
        masking_window (float):
            Optional. Size of the masking window, expressed as a proportion of the total
            length of y. If not given, 0.01 is used.
        mask (bool):
            Optional. Mask the start of anomaly scores.
            If not given, `False` is used.
    Returns:
        ndarray:
            Array of errors.
    """
    errors = np.abs(y - y_hat)[:, 0]
    if not smooth:
        return errors

    smoothing_window = max(1, int(len(y) * smoothing_window))
    errors = pd.Series(errors).ewm(span=smoothing_window).mean().values

    if mask:
        mask_length = int(masking_window * len(errors))
        errors[:mask_length] = min(errors)
    return errors

def get_median(y):
    result = []

    padding = np.array([np.pad(y[i], (i, y.shape[0]-i-1), 'constant', constant_values=0).tolist() for i in range(y.shape[0])])
    padding = np.transpose(padding)
    result = [np.median(pad[pad != 0].tolist()) for pad in padding]
    # for i in range(y.shape[0]+y.shape[1]):
    #     padding = np.pad(y[i], (i, y.shape[0]-i+1), 'constant', constant_values=0)
    #     padding = padding[padding != 0].tolist()
    #     result.append(np.median(padding))
    return np.array(result)

def _area_error(y, y_hat, score_window=10, smoothing_window=0.01,smooth=True):
    """Compute area error between predicted and expected values.
    The computed error is calculated as the area difference between predicted
    and expected values with a smoothing factor.
    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.
    Returns:
        ndarray:
            An array of area error.
    """
    smooth_y = pd.Series(y).rolling(score_window, center=True, min_periods=score_window // 2).apply(integrate.trapz)
    smooth_y_hat = pd.Series(y_hat).rolling(score_window, center=True, min_periods=score_window // 2).apply(integrate.trapz)
    errors = abs(smooth_y - smooth_y_hat)

    if smooth:
        if isinstance(smoothing_window, float):
            smoothing_window = max(1, int(len(y) * smoothing_window))
        errors = pd.Series(errors).rolling(smoothing_window, center=True, min_periods=smoothing_window // 2).mean().values

    return np.array(errors)

def bi_regression_errors(y, ry_hat, fy_hat,
                         smoothing_window: float = 0.01, smooth: bool = True, mask: bool = True):
    """Compute an array of absolute errors comparing the forward and reverse predictions with
    the expected output.

    Anomaly scores are created in the forward and reverse directions. Scores in overlapping indices
    are averaged while scores in non-overlapping indices are taken directly from either forward or
    reverse anomaly scores.

    If smooth is True, apply EWMA to the resulting array of errors.

    Args:
        y (ndarray):
            Ground truth.
        ry_hat (ndarray):
            Predicted values (reverse).
        fy_hat (ndarray):
            Predicted values (forward).
        smoothing_window (float):
            Optional. Size of the smoothing window, expressed as a proportion of the total
            length of y. If not given, 0.01 is used.
        smooth (bool):
            Optional. Indicates whether the returned errors should be smoothed with EWMA.
            If not given, `True` is used.
        mask (bool): bool = True
            Optional. Mask anomaly score errors in the beginning.
            If not given, `True` is used.

    Returns:
        ndarray:
            Array of errors.
    """
    time_steps = len(y) - len(fy_hat)
    mask_steps = int(smoothing_window * len(fy_hat)) if mask else 0
    ry, fy = y[:len(ry_hat), :], y[-len(fy_hat):, :]

    f_scores = regression_errors(fy, fy_hat, smoothing_window=smoothing_window, smooth=smooth)
    f_scores[:mask_steps] = 0
    f_scores = np.concatenate([np.zeros(time_steps), f_scores])


    r_scores = regression_errors(ry, ry_hat, smoothing_window=smoothing_window, smooth=smooth)
    r_scores[:mask_steps] = min(r_scores)
    r_scores = np.concatenate([r_scores, np.zeros(time_steps)])

    scores = f_scores + r_scores
    scores[time_steps + mask_steps:-time_steps] /= 2
    return scores

def score_anomalies(y, ry_hat, y_hat, fy_hat,
                    smoothing_window: float = 0.01, smooth: bool = True, mask: bool = True,
                    comb: str = 'mult', lambda_rec: float = 0.5, rec_error_type: str = "dtw"):
    """Compute an array of absolute errors comparing predictions and expected output.

    If smooth is True, apply EWMA to the resulting array of errors.

    Args:
        y (ndarray):
            Ground truth.
        ry_hat (ndarray):
            Predicted values (reverse).
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        fy_hat (ndarray):
            Predicted values (forward).
        smoothing_window (float):
            Optional. Size of the smoothing window, expressed as a proportion of the total
            length of y. If not given, 0.01 is used.
        smooth (bool):
            Optional. Indicates whether the returned errors should be smoothed with EWMA.
            If not given, `True` is used.
        mask (bool): bool = True
            Optional. Mask anomaly score errors in the beginning.
            If not given, `True` is used.
        rec_error_type (str):
            Optional. The method to compute reconstruction error. Can be one of
            `["point", "area", "dtw"]`. If not given, 'dtw' is used.
        comb (str):
            Optional. How to combine critic and reconstruction error. Can be one
            of `["mult", "sum", "rec"]`. If not given, 'mult' is used.
        lambda_rec (float):
            Optional. Used if `comb="sum"` as a lambda weighted sum to combine
            scores. If not given, 0.5 is used.

    Returns:
        ndarray:
            Array of errors.
    """

    reg_scores = bi_regression_errors(y, ry_hat, fy_hat,
                                      smoothing_window=smoothing_window,
                                      smooth=smooth,
                                      mask=mask)

    rec_scores = _area_error(y[1:-1].reshape((-1,)), get_median(y_hat))

    mask_steps = int(smoothing_window * len(y)) if mask else 0
    rec_scores[:mask_steps] = rec_scores.min()
    rec_scores = np.concatenate([np.zeros(1), rec_scores, np.zeros(1)])

    scores = None
    if comb == "mult":
        reg_scores = MinMaxScaler((1, 2)).fit_transform(reg_scores.reshape(-1, 1)).flatten()
        rec_scores = MinMaxScaler((1, 2)).fit_transform(rec_scores.reshape(-1, 1)).flatten()
        scores = np.multiply(reg_scores, rec_scores)

    elif comb == "sum":
        reg_scores = MinMaxScaler((0, 1)).fit_transform(reg_scores.reshape(-1, 1)).flatten()
        rec_scores = MinMaxScaler((0, 1)).fit_transform(rec_scores.reshape(-1, 1)).flatten()
        scores = (1 - lambda_rec) * reg_scores + lambda_rec * rec_scores

    elif comb == "rec":
        scores = rec_scores

    elif comb == "reg":
        scores = reg_scores

    return scores,get_median(y_hat)

def anomaly_identify(anomaly_scores, pruning_threshold=0.13):
    T = len(anomaly_scores)
    window_size = T // 3
    step_size = T // 30

    # Identify anomalies using sliding windows
    anomalies = np.zeros_like(anomaly_scores, dtype=bool)

    for start in range(0, T - window_size + 1, step_size):
        end = start + window_size
        window = anomaly_scores[start:end]
        mean = np.mean(window)
        std_dev = np.std(window)
        threshold = mean + 3 * std_dev
        window_anomalies = (window > threshold)
        anomalies[start:end] = np.logical_or(anomalies[start:end], window_anomalies)

    # Group consecutive anomalies into sequences
    # anomalous_sequences = []
    # current_sequence = []

    # for i, is_anomalous in enumerate(anomalies):
    #     if is_anomalous:
    #         current_sequence.append(i)
    #     elif current_sequence:
    #         anomalous_sequences.append(current_sequence)
    #         current_sequence = []

    # if current_sequence:
    #     anomalous_sequences.append(current_sequence)

    # # Pruning method to reduce false positives
    # K_max = [max(anomaly_scores[seq]) for seq in anomalous_sequences]
    # K_max_sorted_indices = np.argsort(K_max)[::-1]
    # K_max_sorted = [K_max[i] for i in K_max_sorted_indices]

    # # Calculate percentage changes
    # percentage_changes = [0]
    # for i in range(1, len(K_max_sorted)):
    #     percentage_change = (K_max_sorted[i-1] - K_max_sorted[i]) / K_max_sorted[i-1]
    #     percentage_changes.append(percentage_change)

    # # Find the sequence where percentage change exceeds the threshold
    # for j, p_change in enumerate(percentage_changes):
    #     if p_change <= pruning_threshold:
    #         break

    # # Reclassify sequences as normal
    # for idx in K_max_sorted_indices[j:]:
    #     for i in anomalous_sequences[idx]:
    #         anomalies[i] = False
    return anomalies

def demo(model, sequence, device):
    model.eval()
    sequence = sequence.squeeze(0)

    sequence_numpy = sequence.numpy()
    sequence_numpy = np.array([sequence_numpy[i:i+100,:] for i in range(0, sequence_numpy.shape[0]-100+1)])

    sequence_tensor = torch.tensor(sequence_numpy)

    reconstruction, prediction_forward, prediction_reverse = [], [], []

    for tensor in sequence_tensor.chunk(sequence_tensor.shape[0]//30):
        tensor = tensor.to(device)
        with torch.no_grad():
            recon, pred_fwd, pred_rev = model(tensor[:,1:-1,:])

        reconstruction.append(recon.cpu().detach().clone())
        prediction_forward.append(pred_fwd.cpu().detach().clone())
        prediction_reverse.append(pred_rev.cpu().detach().clone())

        # Clear GPU memory cache
        if device=='cuda':
            torch.cuda.empty_cache()

    reconstruction = torch.cat(reconstruction).squeeze(-1).numpy()
    prediction_forward = torch.cat(prediction_forward).numpy()
    prediction_reverse = torch.cat(prediction_reverse).numpy()

    scores, _ = score_anomalies(sequence.numpy(), prediction_reverse, reconstruction, prediction_forward)
    predictions = anomaly_identify(scores)
    
    return predictions
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
st_autorefresh(interval=5000, key='fizzbuzzcounter')



conn = connect(host='172.20.0.6', port=8000)
curs = conn.cursor()

curs.execute("""
SELECT * FROM poc limit 1000    
""")

# Lấy tất cả các dòng kết quả
rows = curs.fetchall()

# In ra màn hình các dòng kết quả
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AER(input_dim=1).to(device)
model.load_state_dict(torch.load('/home/model1.pt', map_location=torch.device('cpu')), strict=True)
model = model.to(device)
Preprocessor = Custom_Dataset()

st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hiển thị tiêu đề căn giữa
st.markdown('<p class="centered-title">Anomaly Detection</p>', unsafe_allow_html=True)

now = datetime.now()
dt_string = now.strftime("%d %B %Y %H:%M:%S")
st.markdown(
    f'<p style="color:blue; font-style:italic;">Last update: {dt_string}</p>',
    unsafe_allow_html=True
)
timestamp = []
value = []
for row in rows:
    timestamp.append(row[0])
    value.append(row[2])
if len(value) >200:
    value = value[-200:]
    timestamp = timestamp[-200:]
indices = []
anomaly_indices = []
fig, ax = plt.subplots(figsize=(20, 10))  # Kích thước đồ thị
ax.plot(timestamp, value, label='Time Series Data', color='blue')
if len(value) > 199:
    value = Preprocessor.MinMaxScaler(Preprocessor.Detrend(torch.tensor(Preprocessor.Fill_nan(value), dtype=torch.float32).unsqueeze(-1)))
    anomaly_flags = demo(model, value, device)
    indices = np.arange(len(value))
    anomaly_indices = indices[anomaly_flags]
    for idx in anomaly_indices:
        ax.axvline(x=idx, color='red', linestyle='--', label='Anomaly' if idx == anomaly_indices[0] else "")

ax.set_title('Time Series Data with Anomalies', fontsize=24)
ax.set_xlabel('Time', fontsize=20)
ax.set_ylabel('Value', fontsize=20)
ax.legend(fontsize=16)
ax.grid(True)

st.pyplot(fig)
if len(anomaly_indices) != 0:
    st.warning("Anomaly detected!")
for i in anomaly_indices:
    st.write("Anomoly at : ", timestamp[i])
# Hiển thị đồ thị với Streamlit

# Đóng kết nối
conn.close()