import copy
import time

import torch
import geospaNN
import numpy as np
import pandas as pd

path = '/Users/zhanwentao/Documents/Abhi/Conference/IBC2024/data/BCEF/'

combined_data = pd.read_csv(path + 'BCEF.csv')

import geopandas as gpd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Create a GeoDataFrame from the combined data

# Plot using geopandas
gdf = gpd.GeoDataFrame(
    combined_data,
    geometry=gpd.points_from_xy(combined_data.x, combined_data.y)
)
gdf = gdf.set_crs("+proj=aea +lat_1=55 +lat_2=65 +lat_0=50 +lon_0=-154 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=km +no_defs")
gdf = gdf.to_crs("EPSG:4326")

# Geospatial distribution plot for BCEF data.
for variable in ['FCH', 'PTC']:
    plt.clf()
    gdf.plot(column=variable, cmap='viridis', legend=True, figsize=(10, 5), markersize=1)
    plt.title('Geographical Plot of ' + variable + 'for BCEF data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(path + 'BCEF' + variable + '.png')

# Scatter plot for BCEF data, PTC vs FCH.
plt.clf()
plt.figure(figsize=(8, 6))
plt.scatter(np.array(combined_data['FCH']), np.array(combined_data['PTC']),
            s = 1.5, alpha = 0.7)
# Add titles and labels
plt.title('PTC vs FCH', fontsize=14)
plt.xlabel('FCH', fontsize=12)
plt.ylabel('PTC', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
# Show the plot
plt.tight_layout()
# plt.show()
plt.savefig(path + 'scatter.png')

# Subsample and filter the data
combined_data_subsample = gdf.sample(n=30000, random_state=42)
combined_data_subsample = combined_data_subsample.dropna()
combined_data_subsample = combined_data_subsample[combined_data_subsample['PTC']
                                                  < np.quantile(combined_data_subsample['PTC'], 0.95)]
combined_data_subsample["latitude"] = combined_data_subsample.geometry.y
combined_data_subsample["longitude"] = combined_data_subsample.geometry.x

X = torch.from_numpy(combined_data_subsample[['FCH']].to_numpy()).float()
Y = torch.from_numpy(combined_data_subsample[['PTC']].to_numpy()).float().reshape(-1)
coord = torch.from_numpy(combined_data_subsample[["longitude", "latitude"]].to_numpy()).float()
coord[:,0] = (coord[:,0] - min(coord[:,0]))/(max(coord[:,0]) - min(coord[:,0]))
coord[:,1] = (coord[:,1] - min(coord[:,1]))/(max(coord[:,1]) - min(coord[:,1]))

# Scatter plot for subsampled BCEF data, PTC vs FCH.
plt.clf()
plt.figure(figsize=(8, 6))
plt.scatter(np.array(combined_data_subsample['FCH']), np.array(combined_data_subsample['PTC']),
            s = 1.5, alpha = 0.7)
plt.title('PTC vs FCH', fontsize=14)
plt.xlabel('FCH', fontsize=12); plt.ylabel('PTC', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(path + 'scatter_subsample.png')

# Parameter settings
p = X.shape[1]
n = X.shape[0]
nn = 20
batch_size = 50

X, Y, coord, _ = geospaNN.spatial_order(X, Y, coord, method = 'max-min')
data = geospaNN.make_graph(X, Y, coord, nn)

# Training-testing split
z = (combined_data_subsample["latitude"] - 64.68 - 0.8 * 148.4 - 0.8 * combined_data_subsample["longitude"]) > 0
test_mask = torch.logical_and(Y < 60,
                              torch.logical_and(X.reshape(-1) < torch.quantile(X, 0.3),
                                                torch.tensor(z.values)))
combined_data_test = combined_data_subsample[test_mask.detach().numpy()]
np.random.seed(2024)
id_val = np.random.choice(np.where(~test_mask)[0], int(0.2 * n), replace=False)
train_mask = ~test_mask
train_mask[id_val] = False
val_mask = ~torch.logical_or(train_mask, test_mask)

# Geospatial distribution plot for BCEF test data.
for variable in ['FCH', 'PTC']:
    plt.clf()
    combined_data_test.plot(column=variable, cmap='viridis', legend=True, figsize=(10, 5), markersize=1)
    plt.title('Geographical Plot of ' + variable + 'for BCEF data')
    plt.ylim(combined_data_subsample['latitude'].min(), combined_data['latitude'].max())
    plt.xlim(combined_data['longitude'].min(), combined_data['longitude'].max())
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(path + 'BCEF' + variable + '_test.png')

data_train = geospaNN.make_graph(X[train_mask,:], Y[train_mask], coord[train_mask,:], nn)
data_val = geospaNN.make_graph(X[val_mask,:], Y[val_mask], coord[val_mask,:], nn)
data_test = geospaNN.make_graph(X[test_mask,:], Y[test_mask], coord[test_mask,:], nn)

# NN training
torch.manual_seed(2024)
mlp_nn = torch.nn.Sequential(
    torch.nn.Linear(p, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1),
)
nn_model = geospaNN.nn_train(mlp_nn, lr =  0.1, min_delta = 0.001)
training_log = nn_model.train(data_train, data_val, data_test)

theta = geospaNN.theta_update(mlp_nn(data_train.x).squeeze() - data_train.y,
                              data_train.pos, neighbor_size = 20)
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_nn, theta=torch.tensor(theta))
estimate_nn = model.estimate(data_test.x)
predict_nn = model.predict(data_train, data_test)

# NN training
torch.manual_seed(2024)
mlp_nngls = copy.deepcopy(mlp_nn)
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_nngls, theta=torch.tensor(theta, dtype=float))
nngls_model = geospaNN.nngls_train(model, lr = 0.1, min_delta = 0.001)
training_log = nngls_model.train(data_train, data_val, data_test, batch_size = batch_size,
                                 Update_init = 100, Update_step = 5, epoch_num = 20, vignette=False)
theta_hat = geospaNN.theta_update(mlp_nngls(data_train.x).squeeze() - data_train.y,
                                  data_train.pos, neighbor_size = 20)
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_nngls, theta=torch.tensor(theta_hat))
estimate_nngls = model.estimate(data_test.x)
predict_nngls = model.predict(data_train, data_test)

beta, theta_hat_BRISC = geospaNN.BRISC_estimation(data_train.y.detach().numpy(),
                                            torch.concat([torch.ones(data_train.x.shape[0], 1), data_train.x], axis = 1).detach().numpy(),
                                            data_train.pos.detach().numpy())
def mlp_BRISC(X):
    return beta[0] + beta[1]*X
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_BRISC, theta=torch.tensor(theta_hat))
estimate_BRISC = model.estimate(data_test.x)
predict_BRISC = model.predict(data_train, data_test)

X_plt = data_train.x
Y_plt = data_train.y
plt.clf()
plt.scatter(X_plt.detach().numpy(), Y_plt.detach().numpy(), s=1, label='data')
plt.scatter(X_plt.detach().numpy(), mlp_nn(X_plt).detach().numpy(), s=1, label='NN')
plt.scatter(X_plt.detach().numpy(), mlp_BRISC(X_plt).detach().numpy(), s=1, label='linear')
plt.scatter(X_plt.detach().numpy(), mlp_nngls(X_plt).detach().numpy(), s=1, label='NNGLS')
lgnd = plt.legend(fontsize=10)
plt.xlabel('FCH', fontsize=10)
plt.ylabel('PTC', fontsize=10)
plt.title('Estimation')

for handle in lgnd.legend_handles:
    handle.set_sizes([10.0])
plt.savefig(path + 'Estimation_BCEF.png')

plt.clf()
plt.scatter(data_test.y.detach().numpy(), data_test.y.detach().numpy(), s=1, alpha = 0.5, label='Truth')
plt.scatter(data_test.y.detach().numpy(), predict_nn, s=1, alpha = 0.5, label='NN + kriging')
plt.scatter(data_test.y.detach().numpy(), predict_nngls, s=1, alpha = 0.5, label='NNGLS')
plt.scatter(data_test.y.detach().numpy(), predict_BRISC, s=1, alpha = 0.5, label='Linear kriging')
lgnd = plt.legend(fontsize=10)
plt.xlabel('Observed PTC', fontsize=10)
plt.ylabel('Predicted PTC from FCH', fontsize=10)
plt.title('Prediction')

for handle in lgnd.legend_handles:
    handle.set_sizes([10.0])
plt.show()
plt.savefig(path + 'Prediction_BCEF.png')

print(torch.mean((data_test.y - predict_nn)**2))
print(torch.mean((data_test.y - predict_nngls)**2))
print(torch.mean((data_test.y - predict_BRISC)**2))
