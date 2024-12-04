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

for variable in ['FCH', 'PTC']:
    plt.clf()
    gdf.plot(column=variable, cmap='viridis', legend=True, figsize=(10, 5), markersize=1)
    plt.title('Geographical Plot of ' + variable + 'for BCEF data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(path + 'BCEF' + variable + '.png')

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

combined_data_subsample = gdf.sample(n=30000, random_state=42)
combined_data_subsample = combined_data_subsample.dropna()
combined_data_subsample = combined_data_subsample[combined_data_subsample['PTC']
                                                  < np.quantile(combined_data_subsample['PTC'], 0.95)]
combined_data_subsample["latitude"] = combined_data_subsample.geometry.y
combined_data_subsample["longitude"] = combined_data_subsample.geometry.x

X = torch.from_numpy(combined_data_subsample[['PTC']].to_numpy()).float()
Y = torch.from_numpy(combined_data_subsample[['FCH']].to_numpy()).float().reshape(-1)

plt.clf()
plt.figure(figsize=(8, 6))
plt.scatter(np.array(combined_data_subsample['FCH']), np.array(combined_data_subsample['PTC']),
            s = 1.5, alpha = 0.7)
plt.title('PTC vs FCH', fontsize=14)
plt.xlabel('FCH', fontsize=12); plt.ylabel('PTC', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
# Show the plot
plt.tight_layout()
plt.savefig(path + 'scatter_subsample.png')

coord = torch.from_numpy(combined_data_subsample[["longitude", "latitude"]].to_numpy()).float()
coord[:,0] = (coord[:,0] - min(coord[:,0]))/(max(coord[:,0]) - min(coord[:,0]))
coord[:,1] = (coord[:,1] - min(coord[:,1]))/(max(coord[:,1]) - min(coord[:,1]))

p = X.shape[1]

n = X.shape[0]
nn = 20
batch_size = 50

X, Y, coord, _ = geospaNN.spatial_order(X, Y, coord, method = 'max-min')
data = geospaNN.make_graph(X, Y, coord, nn)

torch.manual_seed(2024)
np.random.seed(0)
data_train, data_val, data_test = geospaNN.split_data(X, Y, coord, neighbor_size = 20,
                                                   test_proportion = 0.2)

def block_rand(n, k):
    lx = np.empty(0)
    ly = np.empty(0)
    for i in range(k):
        if i == 0:
            ix = np.random.choice(range(n), 1)
            iy = np.random.choice(range(n), 1)
        else:
            ix = np.random.choice(np.delete(range(n), lx),1)
            iy = np.random.choice(np.delete(range(n), ly),1)
        lx = np.append(lx, ix).astype(int)
        ly = np.append(ly, iy).astype(int)
    return lx, ly


if True:
    split = 5
    n_temp = split
    k_temp = split
    lx, ly = block_rand(n_temp, k_temp)
    np.random.seed(2024)
    xspc = np.linspace(0, 1, n_temp + 1)
    yspc = np.linspace(0, 1, n_temp + 1)
    test_mask = np.zeros(n, dtype=bool)
    coord_np = coord.detach().numpy()
    for i in range(k_temp):
        mask_temp = np.logical_and((coord_np[:, 0] > xspc[lx[i]]) * (coord_np[:, 0] <= xspc[lx[i] + 1]),
                                   (coord_np[:, 1] > yspc[ly[i]]) * (coord_np[:, 1] <= yspc[ly[i] + 1]))
        test_mask = np.logical_or(test_mask, mask_temp)
if True:
    z = (combined_data_subsample["latitude"] - 64.68 - 0.8 * 148.4 - 0.8 * combined_data_subsample["longitude"]) > 0
    test_mask = torch.logical_and(Y < 60,
                                  torch.logical_and(X.reshape(-1) < torch.quantile(X, 0.3),
                                  torch.tensor(z.values)))
    print(test_mask.sum())
if True:
    n_test = 2000
    k = 3
    test_mask = np.zeros(n, dtype=bool)
    x_space = torch.linspace(Y.min(), Y.max(), k + 1)
    for j in range(k):
        x_l, x_r = x_space[j], x_space[j+1]
        np.random.seed(j)
        id_test_temp = np.random.choice(torch.where(torch.logical_and(Y >= x_l, Y < x_r))[0], int(n_test/k), replace = False)
        test_mask[id_test_temp] = True
    print(test_mask.sum())
id_train = np.random.choice(np.where(~test_mask)[0], int(0.2 * n), replace=False)
train_mask = ~test_mask
train_mask[id_train] = False
val_mask = ~np.logical_or(train_mask, test_mask)

combined_data_subsample = combined_data_subsample[combined_data_subsample['PTC']
                                                  < np.quantile(combined_data_subsample['PTC'], 0.95)]
z = (combined_data_subsample["latitude"] - 64.68 - 0.8 * 148.4 - 0.8 * combined_data_subsample["longitude"]) > 0
test_mask = torch.logical_and(Y < 60,
                              torch.logical_and(X.reshape(-1) < torch.quantile(X, 0.3),
                                                torch.tensor(z.values)))
combined_data_test = combined_data_subsample[test_mask.detach().numpy()]
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

mlp_nn = torch.nn.Sequential(
    torch.nn.Linear(p, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1),
)
#mlp_nn0 = copy.deepcopy(mlp_nn)
nn_model = geospaNN.nn_train(mlp_nn, lr =  0.1, min_delta = 0.001)
training_log = nn_model.train(data_train, data_val, data_test)

start = time.time()
n_sub = data_train.x.shape[0]
theta = geospaNN.theta_update(torch.tensor([10, 100, 30]),
                               mlp_nn(data_train.x).squeeze()[range(n_sub)] - data_train.y[range(n_sub)],
                               data_train.pos[range(n_sub)], neighbor_size = 20)
theta[2] = max(1e-03, theta[2])
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_nn, theta=torch.tensor(theta))
estimate0 = model.estimate(data_test.x)
predict0 = model.predict(data_train, data_test)
print(theta)
print(time.time() - start)

start = time.time()
#theta0 = [theta[0]*(1+theta[2]) * 0.98, theta[1], theta[0]*(1+theta[2]) * 0.02]
mlp_nngls = copy.deepcopy(mlp_nn)
#model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_nngls, theta=torch.tensor([10, 150, 20], dtype=float))
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_nngls, theta=torch.tensor(theta, dtype=float))
nngls_model = geospaNN.nngls_train(model, lr = 0.1, min_delta = 0.001)
training_log = nngls_model.train(data_train, data_val, data_test, batch_size = batch_size,
                                 Update_init = 100, Update_step = 5, epoch_num = 20, vignette=False)
theta_hat = geospaNN.theta_update(torch.tensor(theta),
                                  mlp_nngls(data_train.x).squeeze() - data_train.y,
                                  data_train.pos, neighbor_size = 20)
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_nngls, theta=torch.tensor(theta_hat))
estimate = model.estimate(data_test.x)
predict = model.predict(data_train, data_test)
print(time.time() - start)

torch.save(mlp_nngls.state_dict(), 'model_test')

#mlp_nngls_archive = copy.deepcopy(mlp_nngls)

beta, theta_hat_BRISC = geospaNN.BRISC_estimation(data_train.y.detach().numpy(),
                                            torch.concat([torch.ones(data_train.x.shape[0], 1), data_train.x], axis = 1).detach().numpy(),
                                            data_train.pos.detach().numpy())
def mlp_BRISC(X):
    return beta[0] + beta[1]*X
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_BRISC, theta=torch.tensor(theta_hat))
estimate_BRISC = model.estimate(data_test.x)
predict_BRISC = model.predict(data_train, data_test)

start = time.time()
#theta0 = [theta[0]*(1+theta[2]) * 0.98, theta[1], theta[0]*(1+theta[2]) * 0.02]
mlp_linear = torch.nn.Sequential(
    torch.nn.Linear(p, 1)
)
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_linear, theta=torch.tensor(theta))
nngls_model = geospaNN.nngls_train(model, lr = 0.01, min_delta = 0.001)
training_log = nngls_model.train(data_train, data_val, data_test, batch_size = batch_size,
                                 Update_init = 100, Update_step = 5, epoch_num = 20, vignette=True)
theta_hat = geospaNN.theta_update(torch.tensor(theta),
                                  mlp_linear(data_train.x).squeeze() - data_train.y,
                                  data_train.pos, neighbor_size = 20)
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_linear, theta=torch.tensor(theta_hat))
predict2 = model.predict(data_train, data_test)
print(time.time() - start)

X_plt = data_train.x
Y_plt = data_train.y
plt.clf()
plt.scatter(X_plt.detach().numpy(), Y_plt.detach().numpy(), s=1, label='data')
#plt.scatter(X.detach().numpy(), mlp_nn0(X).detach().numpy(), s=1, label='NN0')
plt.scatter(X_plt.detach().numpy(), mlp_nn(X_plt).detach().numpy(), s=1, label='NN')
#plt.scatter(X.detach().numpy(), mlp_linear(X).detach().numpy(), s=1, label='linear')
plt.scatter(X_plt.detach().numpy(), mlp_BRISC(X_plt).detach().numpy(), s=1, label='linear')
plt.scatter(X_plt.detach().numpy(), mlp_nngls(X_plt).detach().numpy(), s=1, label='NNGLS')
#plt.scatter(X.detach().numpy(), mlp_nngls2(X).detach().numpy(), s=1, label='NNGLS nugget 50%')
#plt.scatter(X.detach().numpy(), mlp_nngls3(X).detach().numpy(), s=1, label='NNGLS nugget 97%')
lgnd = plt.legend(fontsize=10)
plt.xlabel('FCH', fontsize=10)
plt.ylabel('PTC', fontsize=10)
plt.title('Estimation')

for handle in lgnd.legend_handles:
    handle.set_sizes([10.0])
plt.show()
plt.savefig(path + 'Estimation_block_bin_Y.png')

plt.clf()
plt.scatter(data_test.y.detach().numpy(), data_test.y.detach().numpy(), s=1, alpha = 0.5, label='Truth')
#plt.scatter(X.detach().numpy(), mlp_nn0(X).detach().numpy(), s=1, label='NN0')
plt.scatter(data_test.y.detach().numpy(), predict0, s=1, alpha = 0.5, label='NN + kriging')
plt.scatter(data_test.y.detach().numpy(), predict, s=1, alpha = 0.5, label='NNGLS')
plt.scatter(data_test.y.detach().numpy(), predict_BRISC, s=1, alpha = 0.5, label='Linear kriging')
#plt.scatter(data_test.y.detach().numpy(), predict3, s=1, alpha = 0.5, label='NNGLS nugget 97%')
lgnd = plt.legend(fontsize=10)
plt.xlabel('Observed PTC', fontsize=10)
plt.ylabel('Predicted PTC from FCH', fontsize=10)
plt.title('Prediction')

plt.clf()
plt.scatter(data_test.x.detach().numpy(), data_test.y.detach().numpy(), s=1, alpha = 0.5, label='Truth')
#plt.scatter(X.detach().numpy(), mlp_nn0(X).detach().numpy(), s=1, label='NN0')
#plt.scatter(data_test.x.detach().numpy(), predict0, s=1, alpha = 0.5, label='NN + kriging')
plt.scatter(data_test.x.detach().numpy(), predict, s=1, alpha = 0.5, label='NNGLS')
plt.scatter(data_test.x.detach().numpy(), predict_BRISC, s=1, alpha = 0.5, label='Linear kriging')
#plt.scatter(data_test.y.detach().numpy(), predict3, s=1, alpha = 0.5, label='NNGLS nugget 97%')
lgnd = plt.legend(fontsize=10)
plt.xlabel('Observed PTC', fontsize=10)
plt.ylabel('Predicted PTC from FCH', fontsize=10)
plt.title('Prediction')

plt.clf()
#plt.scatter(data_test.x.detach().numpy(), data_test.y.detach().numpy(), s=1, alpha = 0.5, label='Truth')
#plt.scatter(X.detach().numpy(), mlp_nn0(X).detach().numpy(), s=1, label='NN0')
#plt.scatter(data_test.x.detach().numpy(), predict0, s=1, alpha = 0.5, label='NN + kriging')
plt.scatter(data_test.x.detach().numpy(), abs(predict - data_test.y.detach().numpy()), s=1, alpha = 0.5, label='NNGLS')
plt.scatter(data_test.x.detach().numpy(), abs(predict_BRISC - data_test.y.detach().numpy()), s=1, alpha = 0.5, label='Linear kriging')
#plt.scatter(data_test.y.detach().numpy(), predict3, s=1, alpha = 0.5, label='NNGLS nugget 97%')
lgnd = plt.legend(fontsize=10)
plt.xlabel('Observed PTC', fontsize=10)
plt.ylabel('Predicted PTC from FCH', fontsize=10)
plt.title('Prediction')

for handle in lgnd.legend_handles:
    handle.set_sizes([10.0])
plt.show()
plt.savefig(path + 'Prediction_block_bin_Y.png')

plt.clf()
plt.scatter(data_train.x.detach().numpy(), data_train.y.detach().numpy(), s=1, alpha = 0.5, label='Training')
plt.scatter(data_test.x.detach().numpy(), data_test.y.detach().numpy(), s=1, alpha = 0.5, label='Testing')
lgnd = plt.legend(fontsize=10)
plt.xlabel('Observed PTC', fontsize=10)
plt.ylabel('Observed FCH', fontsize=10)
plt.title('Training-testing split')

for handle in lgnd.legend_handles:
    handle.set_sizes([10.0])
plt.show()

print(torch.mean((data_test.y - mlp_nn(data_test.x).detach().numpy())**2))
print(torch.mean((data_test.y - predict0)**2))
print(torch.mean((data_test.y - predict)**2))
#print(torch.mean((data_test.y - predict2)**2))
print(torch.mean((data_test.y - predict_BRISC)**2))
# Random 82.98 62.59 50.36

print(torch.mean((data_test.y - estimate0)**2))
print(torch.mean((data_test.y - estimate)**2))
print(torch.mean((data_test.y - estimate_BRISC)**2))
