"""
CGAN_under_deployment.py

Author: Pavan Kumar
Affiliation: Indian Institute of Technology - Madras
Website: pavanmohann.github.io

Description:
    Loads a trained Conditional GAN generator and uses it to predict
    Response Spectra (PGA + PSA at 24 time periods) for a given set of
    conditional inputs (earthquake magnitude, fault type, hypocentral
    depth, distance, Vs30, direction). Produces a mean prediction curve
    with an uncertainty band, plotted against the recorded response
    spectrum for validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

import keras.backend as K
from keras.layers import Layer
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

fontsize = 14

# -----------------------------------------------------------------------
# 1. Load and filter data
# -----------------------------------------------------------------------
DATA_PATH = 'final_data.csv'          # <- update path as needed
MODEL_PATH = 'cgan_gen_1.hdf5'        # <- update path as needed

df = pd.read_csv(DATA_PATH)

columns = ['pga', 'T0.010S', 'T0.020S', 'T0.030S', 'T0.040S', 'T0.050S',
           'T0.060S', 'T0.070S', 'T0.080S', 'T0.090S', 'T0.150S', 'T0.200S',
           'T0.300S', 'T0.500S', 'T0.600S', 'T0.700S', 'T0.800S', 'T0.900S',
           'T1.000S', 'T1.200S', 'T1.500S', 'T2.000S', 'T3.000S', 'T4.000S']

df = df[df[columns].max(axis=1) >= 0.001]

# -----------------------------------------------------------------------
# 2. Feature / label split + scaling
# -----------------------------------------------------------------------
X = df[['eqm', 'ftype', 'hyp', 'dist', 'log_dist', 'log_vs30', 'dir']].dropna()
Y = df[columns].dropna()

X_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)

Y_scaler = MinMaxScaler()
Y_scaled = Y_scaler.fit_transform(Y)

num = 1  # index of the record to analyse
testing_sample_x = X.iloc[num, :]
testing_sample_y = Y.iloc[num, :]

# -----------------------------------------------------------------------
# 3. Custom layer required to load the trained generator
# -----------------------------------------------------------------------
class StochasticInputLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(StochasticInputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(StochasticInputLayer, self).build(input_shape)

    def call(self, x):
        noise = K.random_normal(shape=(self.output_dim,), mean=0., stddev=0.01)
        return x + x * noise

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


custom_objects = {'StochasticInputLayer': StochasticInputLayer}
model = load_model(MODEL_PATH, custom_objects=custom_objects)

time_periods = [0.0099, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.15, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4]

# -----------------------------------------------------------------------
# 4. Single-shot prediction sanity check
# -----------------------------------------------------------------------
x_vals_scaled = X_scaler.transform([testing_sample_x])
y_pred_scaled = model.predict(x_vals_scaled)
y_pred = Y_scaler.inverse_transform(y_pred_scaled)
print("Predicted Immediate output:", y_pred)

# -----------------------------------------------------------------------
# 5. Monte-Carlo predictions (mean + std) via stochastic input layer
# -----------------------------------------------------------------------
def make_predictions(model, X_scaler, Y_scaler, testing_sample_x, num_predictions=75):
    """Run the generator num_predictions times to build a prediction
    distribution (mean + std) for the given input, exploiting the
    stochasticity injected by StochasticInputLayer."""
    x_vals_scaled = X_scaler.transform([testing_sample_x])
    y_pred_scaled = [model.predict(x_vals_scaled, verbose=0) for _ in range(num_predictions)]
    predictions = [Y_scaler.inverse_transform(y)[0] for y in y_pred_scaled]
    return np.array(predictions)


def plot_predictions_and_true_values(predictions, true_values, time_periods):
    """Plot mean prediction ± std-dev band against the recorded spectrum."""
    mean = np.mean(predictions, axis=0)
    std_dev = np.std(predictions, axis=0)

    x = np.array(time_periods)
    xnew = np.linspace(x.min(), x.max(), 200)

    mean_spline = make_interp_spline(x, mean)
    mean_smooth = mean_spline(xnew)

    std_dev_spline = make_interp_spline(x, std_dev)
    std_dev_smooth = std_dev_spline(xnew)

    plt.fill_between(xnew, mean_smooth - std_dev_smooth, mean_smooth + std_dev_smooth,
                      color='b', alpha=1, label='Standard Deviation')
    plt.plot(time_periods, true_values, color='magenta', linewidth=2.5,
              label='Recorded Response Spectra')
    plt.plot(xnew, mean_smooth, label='Mean Prediction', color='lime', linewidth=2.5)

    plt.xscale('log')
    plt.xlabel('Time Period ($s$)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim([0.0099, 4])
    plt.ylabel('Spectral Acceleration, $S_a$ ($g$)', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.show()


if __name__ == '__main__':
    predictions = make_predictions(model, X_scaler, Y_scaler, testing_sample_x)
    plot_predictions_and_true_values(predictions, testing_sample_y, time_periods)
