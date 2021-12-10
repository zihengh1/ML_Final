import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluation(pred, gt, min_index):
    avg_mae = mean_absolute_error(pred[:, min_index], gt[:, min_index])
    avg_rmse = np.sqrt(mean_squared_error(pred[:, min_index], gt[:, min_index]))
    return avg_mae, avg_rmse

adj_mx_la = pd.read_pickle("../DCRNN/data/sensor_graph/adj_mx_la.pkl")
geo_neighbor_dict = pd.read_pickle("../Results/geo_neighbor.pickle")
dtw_neighbor_dict = pd.read_pickle("../Results/dtw_neighbor.pickle")
l2_neighbor_dict = pd.read_pickle("../Results/l2_neighbor.pickle")

train_LA = np.load("../DCRNN/data/METR-LA/train.npz")
test_LA = np.load("../DCRNN/data/METR-LA/test.npz")

sensor_id_list_la = adj_mx_la[0]
neighbor_list = [geo_neighbor_dict, l2_neighbor_dict, dtw_neighbor_dict]
neighbor_type = ["GEO", "L2", "DTW"]

results = []
K = 5
for t, neighbor_dict in enumerate(neighbor_list):
    # print(neighbor_type[t])
    for i in range(len(sensor_id_list_la)):
        # print(sensor_id_list_la[i], end=" | ")
        for k in range(1, len(neighbor_dict[i])):
            if k > K: break
            # print(k, end=", ")
            X_index = np.insert(neighbor_dict[i][:k], 0, i)
            train_X = train_LA["x"][:, :, [X_index], 0].reshape(23974, 12 * (k + 1))
            train_y = train_LA["y"][:, :, i, 0]
            test_X = test_LA["x"][:, :, [X_index], 0].reshape(6850, 12 * (k + 1))
            test_y = test_LA["y"][:, :, i, 0]
            # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

            ### Training SVR with top-5 Neighbors ###
            regr_wo = make_pipeline(StandardScaler(), MultiOutputRegressor(LinearSVR(max_iter=5000)))
            regr_wo = regr_wo.fit(train_X, train_y)
            prediction = regr_wo.predict(test_X)
            MAE_15_mins, RMSE_15_mins = evaluation(prediction, test_y, 2)
            MAE_30_mins, RMSE_30_mins = evaluation(prediction, test_y, 5)
            MAE_60_mins, RMSE_60_mins = evaluation(prediction, test_y, 11)
            # print(round(MAE_15_mins, 2), round(RMSE_15_mins, 2), \
            #       round(MAE_30_mins, 2), round(RMSE_30_mins, 2), \
            #       round(MAE_60_mins, 2), round(RMSE_60_mins, 2))
            results.append([sensor_id_list_la[i], neighbor_type[t], k, MAE_15_mins, RMSE_15_mins, MAE_30_mins, RMSE_30_mins, MAE_60_mins, RMSE_60_mins])
            ###########################################

result_df = pd.DataFrame(results, columns=["Sensor_ID", "Neighbor_Type", "k", "MAE_15mins", "RMSE_15mins", "MAE_30mins", "RMSE_30mins", "MAE_60mins", "RMSE_60mins"])
result_df.to_csv("../Results/Neighbor_SVR_performance_table.csv")
