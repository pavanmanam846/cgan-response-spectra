# -----------------------------------------------------------------------
# CGAN for Ground-Motion Response Spectra Prediction
# Author: Pavan Kumar
# Description: Conditional GAN (generator + discriminator) training
#              pipeline for predicting response spectra (PGA + PSA at
#              24 time periods) conditioned on earthquake magnitude,
#              fault type, hypocentral depth, distance, and Vs30.
# -----------------------------------------------------------------------

# import the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.metrics import r2_score
import math
import warnings
import csv
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import autograd
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

path = <> ## set the path to your file
df_org = pd.read_excel(path)

# create device agnostic code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

# clean the data
df.drop(df[df['Joyner-Boore Dist. (km)'] == -999].index, inplace = True)
df.drop(df[df['Depth Used (km)'] == -999].index, inplace = True)
# df.drop(df[df['Hypocenter Depth (km)'] == -999].index, inplace = True)
df.drop(df[df['Preferred VS30 (m/sec)'] == -999].index, inplace = True)
# Dropping rows with RJB > 1500km (Point 3)
df.drop(df[df['Joyner-Boore Dist. (km)'] > 1500].index, inplace = True)
# Removing questionable Hypocentral Distance (Point 5)
df.drop(df[df['HypD (km)'] <= 0].index, inplace = True)

# rest, follow the steps of data cleaning according to your project

<set up your own data cleaning strategies depending on the data/ project>

"""
SELECT THE COLUMNS CORRESPONDING TO PREDICTOR VARIABLES
"""

res = ['pga',
'T0.010S',
'T0.020S',
'T0.030S',
'T0.040S',
'T0.050S',
'T0.060S',
'T0.070S',
'T0.080S',
'T0.090S',
'T0.150S',
'T0.200S',
'T0.300S',
'T0.500S',
'T0.600S',
'T0.700S',
'T0.800S',
'T0.900S',
'T1.000S',
'T1.200S',
'T1.500S',
'T2.000S',
'T3.000S',
'T4.000S',]

# SCALE THE DATA IN THE RANGE -1 TO 1
from sklearn.preprocessing import MinMaxScaler
# Create the MinMaxScaler object
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler_cond = MinMaxScaler(feature_range=(-1, 1))

# CREATE A NEW DATAFRAME FOR THE CONDITIONAL INPUTS
X_real = pd.DataFrame()
X_real['Mw'] = df['Earthquake Magnitude']
# X['Rjb'] = df['Joyner-Boore Dist. (km)']
### ... Correspondingly add other columns to the new dataframe -> X_real

# log scale the values for predictor variables
for i in PSA_tp:
 X_real[f'PSA{i}'] = np.log10(df[f'T{i}s'])

# TRAIN - TEST - SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test_temp = train_test_split(X_real, test_size=0.3)
X_val, X_test = train_test_split(X_test_temp, test_size=0.5)

# Providing conditional inputs
X_cond_temp = pd.DataFrame(scaler_cond.fit_transform(X_real[['Mw', 'logRjb', 'Vs30', <add other variables>]]))

"""
DISCRIMINATOR CLASS
Input@31: Number of conditional Inputs + Number of predictor variables 
Ouput@1: Whether the data generated is real or synthetic.
[Customize this according to the problem. The generator and discriminator architectures should be modified!]
"""
class Discriminator(nn.Module):
def __init__(self):
super().__init__()
self.model = nn.Sequential(
    nn.Linear(31, 16),  # Change from nn.Dense to nn.Linear and set to 31 input and 16 output neurons
    nn.LeakyReLU(0.2, inplace=True),
    nn.Dropout(0.2),    # Add dropout after the first layer
    nn.Linear(16, 1),   # Output layer with 1 neuron
    nn.Sigmoid()
)


def forward(self, x, c, batch_size):
	c = c.view(batch_size, -1)
	x = x.view(x.size(0), 28)
	x = torch.cat((x, c), 1)
	out = self.model(x.to(torch.float32))
	return out.squeeze()

"""
GENERATOR CLASS
input@n: vector of size -> n
output@24: The number of predictor variables. 
"""


class Generator(nn.Module):
import torch.nn as nn

def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(64, 48),  # Input layer to first hidden layer
        nn.ReLU(0.2, inplace=True),
        nn.Dropout(0.2),    # Dropout layer after first hidden layer
        nn.Linear(48, 32),  # First hidden layer to second hidden layer
        nn.ReLU(0.2, inplace=True),
        nn.Dropout(0.2),    # Dropout layer after second hidden layer
        nn.Linear(32, 24)   # Second hidden layer to output layer
    )


def forward(self, z, c, batch_size):
	# z = torch.from_numpy(z)
	# c = torch.from_numpy(c)
	# c = np.array(c)
	# c = c.view(batch_size, -1)
	c = c.reshape(batch_size, -1)
	z = z.view(batch_size, 10)
	x = torch.cat((z, c), 1)
	out = self.model(x.to(torch.float32))
	return out.view(x.size(0), 24)

"""
GENERATE REAL SAMPLES
"""

def generate_real_samples(df_real, n):
	df_sample = df_real.sample(n)
	X1 = torch.from_numpy(df_sample[[0, 1, 2, ... 6 ]].values)  # 7 conditional variables
	X2 = torch.from_numpy(df_sample[[i for i in range(7,31)]].values) # 24 output variables
	X1 = X1.to(device)
	X2 = X2.to(device)
	return X2, X1

"""
GENERATION OF LATENT DIMENSION (Noise Vector)
"""

def generate_latent_points(latent_dim, n_conditions, batch_size, data):
	z_vector = np.random.randn(latent_dim * batch_size)
	z_vector = torch.from_numpy(z_vector)
	z_vector = z_vector.reshape(batch_size, latent_dim)
	z_vector = z_vector.to(device)
	return z_vector

"""
GENERATION OF SYNTHETIC DATA SAMPLES 
"""

def generate_fake_samples(generator, latent_dim, n_conditions, batch_size,data):
	z_vector = generate_latent_points(latent_dim, n_conditions, batch_size, data)
	cond = np.random.randn(n_conditions * batch_size)
	cond = torch.from_numpy(cond)
	z_vector = z_vector.to(device)
	cond = cond.to(device)
	X = generator(z_vector, cond, batch_size)
	return X, cond

"""
DEFINE GENERATOR AND DISCRIMINATOR
"""
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define your loss function (e.g., binary cross-entropy or anything else for your discriminator)
criterion = nn.BCELoss().to(device)
# criterion = nn.MSELoss().to(device)
torch.set_grad_enabled(True)
# Define your optimizer (e.g., Adam optimizer)
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999)) # DEFINE LEARNING RATE AND CORRESPONDING PARAMETERS FOR GENERATOR AND DISCRIMINATOR NETWORKS
dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

# <DEFINE YOUR PATHS FOR GENERATOR AND DISCRUMINATOR>
generator.load_state_dict(torch.load(CGAN_B_GENERATOR_PATH))
discriminator.load_state_dict(torch.load(CGAN_B_DISCRIMINATOR_PATH))


i=0
with open(res, mode ='r')as file:
csvFile = csv.reader(file)
for lines in csvFile:
i+=1
if i==10000:
latest = lines
float(str(latest[1]))


from tqdm.auto import tqdm
def predict(conditions_test, generator, device):
 generator.eval() # Set the model to evaluation mode
 with torch.no_grad():
		# Generate noise vector
    input = np.random.randn(10 * conditions_test.shape[0])
    input = torch.from_numpy(input)
    input = input.to(device)
    conditions_test = torch.from_numpy(conditions_test)
    conditions_test = conditions_test.to(device)
    # Generate outputs
    predicted_outputs = generator(input, conditions_test, conditions_test.shape[0])
    predicted_outputs = torch.cat((conditions_test, predicted_outputs), 1)
  return predicted_outputs

def train(g_model, d_model, data, val_data, latent_dim, n_epochs = 30000,n_batch = 128, best_r2=-100000):
	batch_per_epoch = int(data.shape[0] / n_batch)
	half_batch = int(n_batch/2)
	y_real = torch.ones((64, 1))
	y_fake = torch.zeros((64, 1))
	y_real = y_real.to(device)
	y_fake = y_fake.to(device)
	for i in range(n_epochs):
		for j in range(batch_per_epoch):
			data_real, cond_real = generate_real_samples(data, half_batch)
			data_fake, cond_fake = generate_fake_samples(generator, latent_dim, 3,↪half_batch, data)
			data_real = data_real.to(device)
			# y_real = y_real.to(device)
			data_fake = data_fake.to(device)
			# y_fake = y_fake.to(device)
			d_model.zero_grad()
			# Fake
			fake_output = discriminator(data_fake, cond_fake, half_batch)
			# print(fake_output, y_fake)
			fake_output = fake_output.to(device)
			d_total_loss = 0
			d_loss = criterion(fake_output, y_fake.squeeze())
			d_total_loss+=d_loss
			d_loss.backward(retain_graph=True)
			dis_optimizer.step()
			# Real
			real_output = discriminator(data_real, cond_real, half_batch)
			real_output = real_output.to(device)
			d_loss = criterion(real_output, y_real.squeeze())
			d_total_loss+=d_loss
			d_loss.backward(retain_graph=True)
			dis_optimizer.step()
			generator.zero_grad()
			fake_output = discriminator(data_fake, cond_fake, half_batch)
			g_loss = criterion(fake_output, y_real.squeeze())
			g_loss.backward()
			gen_optimizer.step()
			if j % 100 == 0 and i%100 == 0:
				print('[%d/%d][%d/%d] Loss_D: %.10f || Loss_G: %.10f' % (i, n_epochs,j, batch_per_epoch, d_total_loss.item(), g_loss.item()))
				X_val_temp = pd.DataFrame(scaler.transform(val_data))
				conditions_val = X_val_temp[[0, 1, 2]]
			conditions_val = conditions_val.to_numpy()
			pred_outputs = predict(conditions_val, generator, device)
			true_values = scaler.inverse_transform(X_val_temp)
			pred_outputs = scaler.inverse_transform(pred_outputs)
			curr_r_squared_test = r2_score(true_values, pred_outputs)
			torch.save(generator.state_dict(), CGAN_GENERATOR_PATH)
			torch.save(discriminator.state_dict(), CGAN_DISCRIMINATOR_PATH)
			if(curr_r_squared_test > best_r2):
			torch.save(generator.state_dict(), CGAN_B_GENERATOR_PATH)
			torch.save(discriminator.state_dict(), CGAN_B_DISCRIMINATOR_PATH)
			best_r2 = curr_r_squared_test
			res_ls = [i, best_r2, curr_r_squared_test, d_total_loss.item(), g_loss.item()]
			with open(res,'a') as fd:
				writer = csv.writer(fd)
				writer.writerow(res_ls)
			if i%100 == 0:
				print(f'Validation r2 Score: {curr_r_squared_test} || Best r2 Score:{best_r2}')


X_test_temp = pd.DataFrame(scaler.transform(X_test))
true_values = scaler.inverse_transform(X_test_temp)
conditions_test = X_test_temp[[0, 1, ... , ]] # <Provide your conditional inputs>
conditions_test = conditions_test.to_numpy()

generator.load_state_dict(torch.load(CGAN_GENERATOR_PATH))
discriminator.load_state_dict(torch.load(CGAN_DISCRIMINATOR_PATH))
# <LOAD THE G & D MODELS FROM THE PATHS>
"""
CHECK THE MEAN OF PREDICTIONS
"""

 mean_output = []
for i in range(1000):
	output = predict(conditions_test, generator, device)
	output = scaler.inverse_transform(output)
	mean_output.append(output)
	mean_array = np.mean(mean_output, axis=0)

"""
COMPUTE THE R2 SCORE, MEAN VALUES AND THE MEAN OF THE PREDS
"""

r_squared_test = r2_score(true_values, mean_array)
true_values = true_values[:,7:]
mean_array = mean_array[:,7:]


plt.xlabel(r'$y-rec$')
plt.ylabel(r'$y-pred$')
#plt.scatter(X_test.data.numpy(), y_test.data.numpy(), color='k', s=2)
#serial = range(0,37)
# serial = 20
# XX = y_test_tensor[:, serial].data.numpy()
XX = true_values
YY = mean_array
# YY = mean_prediction[:, ser
plt.scatter(XX, YY, color='r', s=10)
#plt.scatter( y_test,y_predict.data.numpy(), color='r', s=10)
min_val = min(np.min(XX), np.min(YY))
max_val = max(np.max(XX), np.max(YY))
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--',label='r2=0.75')
plt.title('R2 Plot')
plt.ylabel('Rec-test')
plt.xlabel('Pre-test')
plt.legend()
plt.show()


X_train_temp = pd.DataFrame(X_train)
true_values = scaler.inverse_transform(X_train_temp)
conditions_train = X_train_temp[[0, 1, 2]]
conditions_train = conditions_train.to_numpy()
X_train_temp


mean_output = []
for i in range(1000):
output = predict(conditions_train, generator, device)
output = scaler.inverse_transform(output)
mean_output.append(output)
mean_array = np.mean(mean_output, axis=0)
mean_array

r_squared_test = r2_score(true_values, mean_array)
r_squared_test

true_values = true_values[:,7:]
mean_array = mean_array[:,7:]

plt.xlabel(r'$y-rec$')
plt.ylabel(r'$y-pred$')
#plt.scatter(X_train.data.numpy(), y_train.data.numpy(), color='k', s=2)
#serial = range(0,37)
# serial = 20
# XX = y_train_tensor[:, serial].data.numpy()
XX = true_values
YY = mean_array
# YY = mean_prediction[:, ser
plt.scatter(XX, YY, color='r', s=10)
#plt.scatter( y_train,y_predict.data.numpy(), color='r', s=10)
min_val = min(np.min(XX), np.min(YY))
max_val = max(np.max(XX), np.max(YY))
plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--',label='r2=0.76')
plt.title('R2 Plot')
plt.ylabel('Rec-train')
plt.xlabel('Pre-train')
plt.legend()
plt.show()

import sklearn.model_selection as sk

mask = df['USGS Potentially Induced Event (PIE) Flag'] == True # <Add the flag corresponding to your data>
inter = df[mask]
intra = df[~mask]

inter.reset_index(inplace = True)
intra.reset_index(inplace = True)

inter['Mw/Rjb'] = np.array(inter['Earthquake Magnitude']) / np.
↪array(inter['Joyner-Boore Dist. (km)'])
inter['log Rjb'] = np.log10(inter['Joyner-Boore Dist. (km)'])
yy = pd.DataFrame()
yy['log PGA'] = np.log10(inter['PGA-H RotDnn (g)'])
yy['log PGV'] = np.log10(inter['PGV-H RotDnn (cm/s)'])
t = [0.0099, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,0.09, 0.15, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4]
for i in t:
	yy[f"log PSA {i}s"] = np.log10(inter[f"T{i}s"])


	XX = inter[['Earthquake Magnitude', 'Joyner-Boore Dist. (km)', ... 'Preferred VS30 (m/sec)']] # change accordong to your input

	conditions_inter = XX[['Earthquake Magnitude', ... 'Preferred VS30 (m/sec)']] # <change the conditional inputs according to your>
	conditions_inter = conditions_inter.to_numpy()
	conditions_inter_pred = scaler_cond.transform(conditions_inter)


mean_output = []
for i in range(1000):
output = predict(conditions_inter_pred, generator, device)
output = scaler.inverse_transform(output)
mean_output.append(output)
inter_y_pred = np.mean(mean_output, axis=0)

residue_inter = yy - inter_y_pred[:,6:]


import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XX['Joyner-Boore Dist. (km)'].max() - XX['Joyner-Boore␣
↪Dist. (km)'].min()) / 200) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_inter', 'Rjb (km)'])
for i in range(num_intervals):
	start_dist = i * 200
	end_dist = start_dist + 200
	intervals_X = XX[(XX['Joyner-Boore Dist. (km)'] >= start_dist) &(XX['Joyner-Boore Dist. (km)'] < end_dist)]
	interval_data = residue_inter[(XX['Joyner-Boore Dist. (km)'] >= start_dist) &(XX['Joyner-Boore Dist. (km)'] < end_dist)]
	residue_intervals = interval_data['log PSA 0.01s']
	df2 = pd.DataFrame()
	df2['residue_inter']=residue_intervals
	df2['Rjb (km)']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True)
ax = sns.stripplot(x="Rjb (km)", y="residue_inter", data=df1, ax= ax,marker="$\circ$", color=".25")
ax = sns.boxplot(x="Rjb (km)", y="residue_inter", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PSA Residual (T=0.01s) vs Rjb(km)')
plt.show()


import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XX['Joyner-Boore Dist. (km)'].max() - XX['Joyner-BooreDist. (km)'].min()) / 200) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_inter', 'Rjb (km)'])
for i in range(num_intervals):
	start_dist = i * 200
	end_dist = start_dist + 200
	intervals_X = XX[(XX['Joyner-Boore Dist. (km)'] >= start_dist) &(XX['Joyner-Boore Dist. (km)'] < end_dist)]
	interval_data = residue_inter[(XX['Joyner-Boore Dist. (km)'] >= start_dist) &(XX['Joyner-Boore Dist. (km)'] < end_dist)]
	residue_intervals = interval_data['log PSA 0.2s']
	df2 = pd.DataFrame()
	df2['residue_inter']=residue_intervals
	df2['Rjb (km)']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True)
ax = sns.stripplot(x="Rjb (km)", y="residue_inter", data=df1, ax= ax,marker="$\circ$", color=".25")
ax = sns.boxplot(x="Rjb (km)", y="residue_inter", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PSA Residual (T=0.2s) vs Rjb(km)')
plt.show()


import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XX['Joyner-Boore Dist. (km)'].max() - XX['Joyner-BooreDist. (km)'].min()) / 200) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_inter', 'Rjb (km)'])
for i in range(num_intervals):
	start_dist = i * 200
	end_dist = start_dist + 200
	intervals_X = XX[(XX['Joyner-Boore Dist. (km)'] >= start_dist) &(XX['Joyner-Boore Dist. (km)'] < end_dist)]
	interval_data = residue_inter[(XX['Joyner-Boore Dist. (km)'] >= start_dist) &(XX['Joyner-Boore Dist. (km)'] < end_dist)]
	residue_intervals = interval_data['log PGA']
	df2 = pd.DataFrame()
	df2['residue_inter']=residue_intervals
	df2['Rjb (km)']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True)
ax = sns.stripplot(x="Rjb (km)", y="residue_inter", data=df1, ax= ax,marker="$\circ$", color=".25")
ax = sns.boxplot(x="Rjb (km)", y="residue_inter", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PGA Residual vs Rjb(km)')
plt.show()

import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XX['Earthquake Magnitude'].max() - XX['EarthquakeMagnitude'].min())) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_inter', 'Mw'])
for i in range(num_intervals):
	start_dist = (i+3) * 1
	end_dist = start_dist + 1
	intervals_X = XX[(XX['Earthquake Magnitude'] >= start_dist) & (XX['EarthquakeMagnitude'] < end_dist)]
	interval_data = residue_inter[(XX['Earthquake Magnitude'] >= start_dist) &(XX['Earthquake Magnitude'] < end_dist)]
	residue_intervals = interval_data['log PSA 0.01s']
	df2 = pd.DataFrame()
	df2['residue_inter']=residue_intervals
	df2['Mw']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True
ax = sns.stripplot(x="Mw", y="residue_inter", data=df1, ax= ax,marker="$\circ$", color=".25")
ax = sns.boxplot(x="Mw", y="residue_inter", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PSA Residual (T=0.01s) vs Mw')
plt.show()

import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XX['Earthquake Magnitude'].max() - XX['EarthquakeMagnitude'].min())) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_inter', 'Mw'])
for i in range(num_intervals):
	start_dist = (i+3) * 1
	end_dist = start_dist + 1
	intervals_X = XX[(XX['Earthquake Magnitude'] >= start_dist) & (XX['EarthquakeMagnitude'] < end_dist)]
	interval_data = residue_inter[(XX['Earthquake Magnitude'] >= start_dist) &(XX['Earthquake Magnitude'] < end_dist)]

	residue_intervals = interval_data['log PSA 0.2s']
	df2 = pd.DataFrame()
	df2['residue_inter']=residue_intervals
	df2['Mw']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True)
ax = sns.stripplot(x="Mw", y="residue_inter", data=df1, ax= ax,marker="$\circ$", color=".25")
ax = sns.boxplot(x="Mw", y="residue_inter", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PSA Residual (T=0.2s) vs Mw')
plt.show()


import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XX['Earthquake Magnitude'].max() - XX['EarthquakeMagnitude'].min())) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_inter', 'Mw'])
for i in range(num_intervals):
	start_dist = (i+3) * 1
	end_dist = start_dist + 1
	intervals_X = XX[(XX['Earthquake Magnitude'] >= start_dist) & (XX['EarthquakeMagnitude'] < end_dist)]
	interval_data = residue_inter[(XX['Earthquake Magnitude'] >= start_dist) &(XX['Earthquake Magnitude'] < end_dist)]
	residue_intervals = interval_data['log PGA']
	df2 = pd.DataFrame()
	df2['residue_inter']=residue_intervals
	df2['Mw']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True)
ax = sns.stripplot(x="Mw", y="residue_inter", data=df1, ax= ax,marker="$\circ$", color=".25")
ax = sns.boxplot(x="Mw", y="residue_inter", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PGA Residual vs Mw')
plt.show()

import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XX['Earthquake Magnitude'].max() - XX['EarthquakeMagnitude'].min())) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_inter', 'Mw'])
for i in range(num_intervals):
	start_dist = (i+3) * 1
	end_dist = start_dist + 1
	intervals_X = XX[(XX['Earthquake Magnitude'] >= start_dist) & (XX['EarthquakeMagnitude'] < end_dist)]
	interval_data = residue_inter[(XX['Earthquake Magnitude'] >= start_dist) &(XX['Earthquake Magnitude'] < end_dist)]
	residue_intervals = interval_data['log PSA 0.01s']
	df2 = pd.DataFrame()
	df2['residue_inter']=residue_intervals
	df2['Mw']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True)
ax = sns.stripplot(x="Mw", y="residue_inter", data=df1, ax= ax,marker="$\circ$", color=".25")
ax = sns.boxplot(x="Mw", y="residue_inter", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PSA Residual (T=0.01s) vs Mw')
plt.show()


intra['Mw/Rjb'] = np.array(intra['Earthquake Magnitude']) / np.array(intra['Joyner-Boore Dist. (km)'])
intra['log Rjb'] = np.log10(intra['Joyner-Boore Dist. (km)'])
yyy = pd.DataFrame()
yyy['log PGA'] = np.log10(intra['PGA-H RotDnn (g)'])
yyy['log PGV'] = np.log10(intra['PGV-H RotDnn (cm/s)'])
t = [0.0099, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,0.09, 0.15, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4]
for i in t:
	yyy[f"log PSA {i}s"] = np.log10(intra[f"T{i}s"])
	XXX = intra[['Earthquake Magnitude', 'Joyner-Boore Dist. (km)', ..., 'Preferred VS30 (m/sec)']

conditions_intra = XXX[['Earthquake Magnitude', ...'Preferred VS30 (m/sec)']]
conditions_intra = conditions_intra.to_numpy()
conditions_intra_pred = scaler_cond.transform(conditions_intra)

mean_output = []
for i in range(1000):
output = predict(conditions_intra_pred, generator, device)
output = scaler.inverse_transform(output)
mean_output.append(output)
intra_y_pred = np.mean(mean_output, axis=0)


residue_intra = yyy - intra_y_pred[:,7:]
import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XXX['Joyner-Boore Dist. (km)'].max() - XXX['Joyner-BooreDist. (km)'].min()) / 200) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_intra', 'Rjb (km)'])
for i in range(num_intervals):
	start_dist = i * 200
	end_dist = start_dist + 200
	intervals_X = XXX[(XXX['Joyner-Boore Dist. (km)'] >= start_dist) &(XXX['Joyner-Boore Dist. (km)'] < end_dist)]
	interval_data = residue_intra[(XXX['Joyner-Boore Dist. (km)'] >= start_dist)& (XXX['Joyner-Boore Dist. (km)'] < end_dist)]
	residue_intervals = interval_data['log PSA 0.01s']
	df2 = pd.DataFrame()
	df2['residue_intra']=residue_intervals
	df2['Rjb (km)']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True)
ax = sns.stripplot(x="Rjb (km)", y="residue_intra", data=df1, ax= ax,marker="$\circ$", color=".25")
ax = sns.boxplot(x="Rjb (km)", y="residue_intra", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PSA Residual (T=0.01s) vs Rjb(km)')
plt.show()


import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XXX['Joyner-Boore Dist. (km)'].max() - XXX['Joyner-BooreDist. (km)'].min()) / 200) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_intra', 'Rjb (km)'])
for i in range(num_intervals):
	start_dist = i * 200
	end_dist = start_dist + 200
	intervals_X = XXX[(XXX['Joyner-Boore Dist. (km)'] >= start_dist) &(XXX['Joyner-Boore Dist. (km)'] < end_dist)]
	interval_data = residue_intra[(XXX['Joyner-Boore Dist. (km)'] >= start_dist)& (XXX['Joyner-Boore Dist. (km)'] < end_dist)]
	residue_intervals = interval_data['log PSA 0.2s']
	df2 = pd.DataFrame()
	df2['residue_intra']=residue_intervals
	df2['Rjb (km)']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True)
ax = sns.stripplot(x="Rjb (km)", y="residue_intra", data=df1, ax= ax,marker="$\circ$", color=".25")
ax = sns.boxplot(x="Rjb (km)", y="residue_intra", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PSA Residual (T=0.2s) vs Rjb(km)')
plt.show()


import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XXX['Joyner-Boore Dist. (km)'].max() - XXX['Joyner-BooreDist. (km)'].min()) / 200) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_intra', 'Rjb (km)'])
for i in range(num_intervals):
	start_dist = i * 200
	end_dist = start_dist + 200
	intervals_X = XXX[(XXX['Joyner-Boore Dist. (km)'] >= start_dist) &(XXX['Joyner-Boore Dist. (km)'] < end_dist)]
	interval_data = residue_intra[(XXX['Joyner-Boore Dist. (km)'] >= start_dist)& (XXX['Joyner-Boore Dist. (km)'] < end_dist)]
	residue_intervals = interval_data['log PGA']
	df2 = pd.DataFrame()
	df2['residue_intra']=residue_intervals
	df2['Rjb (km)']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True)
ax = sns.stripplot(x="Rjb (km)", y="residue_intra", data=df1, ax= ax,marker="$\circ$", color=".25")
ax = sns.boxplot(x="Rjb (km)", y="residue_intra", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PGA Residual vs Rjb(km)')
plt.show()


import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XXX['Earthquake Magnitude'].max() - XXX['EarthquakeMagnitude'].min())) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_intra', 'Mw'])
for i in range(num_intervals):
	start_dist = (i+3) * 1
	end_dist = start_dist + 1
	intervals_X = XXX[(XXX['Earthquake Magnitude'] >= start_dist) &(XXX['Earthquake Magnitude'] < end_dist)]
	intraval_data = residue_intra[(XXX['Earthquake Magnitude'] >= start_dist) &(XXX['Earthquake Magnitude'] < end_dist)]

	residue_intervals = intraval_data['log PSA 0.01s']
	df2 = pd.DataFrame()
	df2['residue_intra']=residue_intervals
	df2['Mw']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True)
ax = sns.stripplot(x="Mw", y="residue_intra", data=df1, ax= ax,arker="$\circ$", color=".25")
ax = sns.boxplot(x="Mw", y="residue_intra", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PSA Residual (T=0.01s) vs Mw')
plt.show()

import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XXX['Earthquake Magnitude'].max() - XXX['EarthquakeMagnitude'].min())) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_intra', 'Mw'])
for i in range(num_intervals):
	start_dist = (i+3) * 1
	end_dist = start_dist + 1
	intervals_X = XXX[(XXX['Earthquake Magnitude'] >= start_dist) &(XXX['Earthquake Magnitude'] < end_dist)]
	intraval_data = residue_intra[(XXX['Earthquake Magnitude'] >= start_dist) &(XXX['Earthquake Magnitude'] < end_dist)]
	residue_intervals = intraval_data['log PSA 0.2s']
	df2 = pd.DataFrame()
	df2['residue_intra']=residue_intervals
	df2['Mw']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True)
ax = sns.stripplot(x="Mw", y="residue_intra", data=df1, ax= ax,marker="$\circ$", color=".25")
ax = sns.boxplot(x="Mw", y="residue_intra", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PSA Residual (T=0.2s) vs Mw')
plt.show()


import matplotlib.pyplot as plt
import pylab as P
num_intervals = int((XXX['Earthquake Magnitude'].max() - XXX['EarthquakeMagnitude'].min())) + 1
fig, ax = plt.subplots(figsize = (12,6))
df1 = pd.DataFrame(columns = ['residue_intra', 'Mw'])
for i in range(num_intervals):
	start_dist = (i+3) * 1
	end_dist = start_dist + 1
	intervals_X = XXX[(XXX['Earthquake Magnitude'] >= start_dist) &(XXX['Earthquake Magnitude'] < end_dist)]
	intraval_data = residue_intra[(XXX['Earthquake Magnitude'] >= start_dist) &(XXX['Earthquake Magnitude'] < end_dist)]
	residue_intervals = intraval_data['log PGA']
	df2 = pd.DataFrame()
	df2['residue_intra']=residue_intervals
	df2['Mw']=(start_dist+end_dist)/2
	df1 = pd.concat([df1,df2], ignore_index=True)
ax = sns.stripplot(x="Mw", y="residue_intra", data=df1, ax= ax,marker="$\circ$", color=".25")
ax = sns.boxplot(x="Mw", y="residue_intra", data=df1, ax=ax)
ax.axhline(0, ls='--', color='r')
plt.title('PGA Residual vs Mw')
plt.show()


X_plot = pd.DataFrame(columns = ['Mw', ..., 'Vs30']) # <conditional inputs>
X_plot.loc[0] = [3, math.log10(10), 760]
X_plot.loc[1] = [4, math.log10(10), 760]
X_plot.loc[2] = [5, math.log10(10), 760]
X_plot = scaler_cond.transform(X_plot.to_numpy())


mean_output = []
for i in range(1000):
output = predict(X_plot, generator, device)
output = scaler.inverse_transform(output)
mean_output.append(output)
mean_array = np.mean(mean_output, axis=0)
mean_array = np.power(10, mean_array)

plt.plot(PSA_tp, mean_array[0][5:], linestyle='--', marker = 'd', label="Mw=3")
plt.plot(PSA_tp, mean_array[1][5:], linestyle='--', marker = 's', label="Mw=4")
plt.plot(PSA_tp, mean_array[2][5:], linestyle='--', marker = '^', label="Mw=5")
# <You can add more earthquake magnitudes>
plt.grid(True, which="both")
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="upper right")
plt.title("PSA vs Time Period(s)")
plt.xlabel("Time Period(s)")
plt.ylabel("PSA(g)")


X_plot = pd.DataFrame(columns = ['Mw', 'logRjb', 'Vs30']) # <You can experiment with other variables>
X_plot.loc[0] = [4, math.log10(10), 760]
X_plot.loc[1] = [4, math.log10(50), 760]
X_plot.loc[2] = [4, math.log10(100), 760]
X_plot.loc[3] = [4, math.log10(150), 760]
X_plot = scaler_cond.transform(X_plot.to_numpy())


mean_output = []
for i in range(1000):
output = predict(X_plot, generator, device)
output = scaler.inverse_transform(output)
mean_output.append(output)
mean_array = np.mean(mean_output, axis=0)
mean_array = np.power(10, mean_array)

plt.plot(PSA_tp, mean_array[0][5:], linestyle='--', marker = 'd',label="Rjb=10km")
plt.plot(PSA_tp, mean_array[1][5:], linestyle='--', marker = 's',label="Rjb=50km")
plt.plot(PSA_tp, mean_array[2][5:], linestyle='--', marker = '^',label="Rjb=100km")
plt.plot(PSA_tp, mean_array[3][5:], linestyle='--', marker = '*',label="Rjb=150km")
plt.grid(True, which="both")
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="upper right")
plt.title("PSA vs Time Period(s)")
plt.xlabel("Time Period(s)")
plt.ylabel("PSA(g)")

dict_ = {0:'d', 1:'s', 2:'^', 3:'*', 4:'o', 5:'+'}


X_plot = pd.DataFrame(columns = ['Mw', ... 'Vs30'])
X_plot.loc[0] = [4, math.log10(100), 540]
X_plot.loc[1] = [4, math.log10(100), 760]
X_plot.loc[2] = [4, math.log10(100), 1080]
X_plot = scaler_cond.transform(X_plot.to_numpy())


mean_output = []
for i in range(1000):
	output = predict(X_plot, generator, device)
	output = scaler.inverse_transform(output)
	mean_output.append(output)
mean_array = np.mean(mean_output, axis=0)
mean_array = np.power(10, mean_array)

plt.plot(PSA_tp, mean_array[0][5:], linestyle='--', marker = 'd',label="Vs30=760km")
plt.plot(PSA_tp, mean_array[1][5:], linestyle='--', marker = 's',label="Vs30=1080km")
plt.plot(PSA_tp, mean_array[2][5:], linestyle='--', marker = '^',label="Vs30=1540km")
plt.grid(True, which="both")
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="upper right")
plt.title("PSA vs Time Period(s)")
plt.xlabel("Time Period(s)")
plt.ylabel("PSA(g)")

temp_Rjb = []
for i in range(200):
temp_Rjb.append((i+1)*5)
X_plot = pd.DataFrame(columns = ['Mw', ... 'Vs30']) # <Experiment with log RJB>
for i in range(200):
X_plot.loc[i] = [5, np.log10((i+1)*5), 2000]
X_plot = scaler_cond.transform(X_plot)


mean_predictions = []
for i in range(1000):
	predictions = predict(X_plot, generator, device)
	predictions = scaler.inverse_transform(predictions)
	mean_predictions.append(predictions)
mean_array = np.mean(mean_predictions, axis=0)
mean_array = np.power(10, mean_array)

multi_predictions = []
pred_value = []
for i in range(200):
	multi_predictions.append([])
	pred_value.append([])
	for i in range(1000):
		predictions = predict(X_plot, generator, device)
		predictions = scaler.inverse_transform(predictions)
		mean_predictions.append(predictions)
		for j in range(200):
			multi_predictions[j].append(predictions[j])
for i in range(200):
	pred_value[i] = np.array(multi_predictions[i]).mean(axis=0)


y_plot = predict(X_plot, generator, device)
y_plot = scaler.inverse_transform(y_plot)
y_plot_new = []
for i in range(200):
	y_plot_new.append(y_plot[i][2:])


temp_plot = []
for i in range(200):
	temp_plot.append(pred_value[i][9])
plt.plot(temp_Rjb, temp_plot)
plt.title("PSA (T=0.1) vs Rjb(km)")
plt.ylabel("PSA (g)")
plt.xlabel("Rjb(km)")

X_plot = pd.DataFrame(columns = ['Mw', 'logRjb', 'Vs30'])
temp_Mw = []
for i in range(20,60):
	temp_Mw.append((i+1)*0.1)
j=0
for i in range(20,60):
	X_plot.loc[j] = [(i+1)*0.1, np.log10(100), 760]
	j+=1
X_plot = scaler_cond.transform(X_plot)


mean_predictions = []
for i in range(1000):
	predictions = predict(X_plot, generator, device)
	predictions = scaler.inverse_transform(predictions)
	mean_predictions.append(predictions)
mean_array = np.mean(mean_predictions, axis=0)
mean_array = np.power(10, mean_array)

multi_predictions = []
pred_value = []
for i in range(40):
	multi_predictions.append([])
	pred_value.append([])
for i in range(40):
	predictions = predict(X_plot, generator, device)
	predictions = scaler.inverse_transform(predictions)
	mean_predictions.append(predictions)
	for j in range(40):
		multi_predictions[j].append(predictions[j])
for i in range(40):
	pred_value[i] = np.array(multi_predictions[i]).mean(axis=0)
y_plot = predict(X_plot, generator, device)
y_plot = scaler.inverse_transform(y_plot)
y_plot_new = []
for i in range(40):
	y_plot_new.append(y_plot[i][2:])


temp_plot = []
for i in range(40):
	temp_plot.append(pred_value[i][9])
plt.plot(temp_Mw, temp_plot)
plt.title("PSA (g) vs Mw")
plt.ylabel("PSA (T=0.1s)")
plt.xlabel("Mw")


his = pd.read_csv('<path to results csv>')
his.columns = ['i', 'best_r2', 'curr_r2', 'd_loss', 'g_loss'] # or any other metrics that you defined above
his # print them



r2_his = his['curr_r2']
d_loss_his = his['d_loss']
g_loss_his = his['g_loss']

"""
GENERATOR LOSS PER EPOCH
"""

g_loss_mini = []
g_loss_idx = []
for i in range(len(g_loss_his)):
	if(i%1000==0):
		g_loss_mini.append(g_loss_his.iloc[i])
		g_loss_idx.append(i)
plt.plot(g_loss_idx, g_loss_mini)
plt.xlabel('Number of Epochs')
plt.ylabel('Generator Loss')
plt.title('Generator Loss vs Number of Epochs')

d_loss_mini = []
d_loss_idx = []
for i in range(len(d_loss_his)):
	if(i%1000==0):
		d_loss_mini.append(d_loss_his.iloc[i])
		d_loss_idx.append(i)
plt.plot(d_loss_idx, d_loss_mini)
plt.xlabel('Number of Epochs')
plt.ylabel('Discriminator Loss')
plt.title('Discriminator Loss vs Number of Epochs')
