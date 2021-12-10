import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.fftpack import dct
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from tsaug import TimeWarp, AddNoise, Dropout, Drift, Reverse, Pool

def evaluation(pred, gt, min_index):
    avg_mae = mean_absolute_error(pred[:, min_index], gt[:, min_index])
    avg_rmse = np.sqrt(mean_squared_error(pred[:, min_index], gt[:, min_index]))
    return avg_mae, avg_rmse

adj_mx_la = pd.read_pickle("../DCRNN/data/sensor_graph/adj_mx_la.pkl")
sensor_list_la = adj_mx_la[0]
train_LA = np.load("../DCRNN/data/METR-LA/train.npz")
test_LA = np.load("../DCRNN/data/METR-LA/test.npz")

print(train_LA["x"].shape, train_LA["y"].shape, test_LA["x"].shape, test_LA["y"].shape)

my_augmenter1 = (AddNoise(scale=0.1, distr='gaussian', kind='additive') * 5)
my_augmenter2 = (TimeWarp() * 5)
my_augmenter3 = (Reverse() @ 0.5 * 5)
my_augmenter4 = (Drift(max_drift=(0, 0.2)) @ 0.5 * 5)
my_augmenter5 = (Pool(size=3) @ 0.5 * 5)
my_augmenter6 = (Dropout(fill="mean") @ 0.5 * 5)
augmenters = [my_augmenter1, my_augmenter2, my_augmenter3, my_augmenter4, my_augmenter5, my_augmenter6]
augmenter_names = ["Add Noise", "Time Warp", "Reverse", "Drift", "Pool", "Drop"]

results = []
for i in tqdm(range(len(sensor_list_la))):
    # if i > 4:
    #     break
    print(i, " Sensor: ", sensor_list_la[i], end=" | ")
    train_X = train_LA["x"][:, :, i, 0]
    train_y = train_LA["y"][:, :, i, 0]
    test_X = test_LA["x"][:, :, i, 0]
    test_y = test_LA["y"][:, :, i, 0]
    
    ### Training PLSR without Augmentation ###
    # print("Without Augmentation")
    regr_wo = make_pipeline(StandardScaler(), PLSRegression(n_components=6))
    regr_wo = regr_wo.fit(train_X, train_y)
    prediction = regr_wo.predict(test_X)
    MAE_15_mins, RMSE_15_mins = evaluation(prediction, test_y, 2)
    MAE_30_mins, RMSE_30_mins = evaluation(prediction, test_y, 5)
    MAE_60_mins, RMSE_60_mins = evaluation(prediction, test_y, 11)
    # print(round(MAE_15_mins, 2), round(RMSE_15_mins, 2), round(MAE_30_mins, 2), round(RMSE_30_mins, 2), round(MAE_60_mins, 2), round(RMSE_60_mins, 2))
    results.append([sensor_list_la[i], "Without Augmentation", MAE_15_mins, RMSE_15_mins, MAE_30_mins, RMSE_30_mins, MAE_60_mins, RMSE_60_mins])
    ###########################################

    ### Training PLSR with DCT ###
    # print("Discrete Cosine Transform")
    train_X_fft = dct(train_X)
    test_X_fft = dct(test_X)
    train_X_fft = np.concatenate([train_X, train_X_fft], axis=1)
    test_X_fft = np.concatenate([test_X, test_X_fft], axis=1)
    regr = make_pipeline(StandardScaler(), PLSRegression(n_components=6))
    regr = regr.fit(train_X_fft, train_y)
    prediction = regr.predict(test_X_fft)
    MAE_15_mins, RMSE_15_mins = evaluation(prediction, test_y, 2)
    MAE_30_mins, RMSE_30_mins = evaluation(prediction, test_y, 5)
    MAE_60_mins, RMSE_60_mins = evaluation(prediction, test_y, 11)
    # print(round(MAE_15_mins, 2), round(RMSE_15_mins, 2), round(MAE_30_mins, 2), round(RMSE_30_mins, 2), round(MAE_60_mins, 2), round(RMSE_60_mins, 2))
    results.append([sensor_list_la[i], "DCT", MAE_15_mins, RMSE_15_mins, MAE_30_mins, RMSE_30_mins, MAE_60_mins, RMSE_60_mins])
    ###########################################
    
    ### Training PLSR with Augmentation ###
    for a, aug in enumerate(augmenters):
        # print(augmenter_names[a])
        
        """
        train_X_copy = train_X.copy()
        test_X_copy = test_X.copy()
        train_X_aug = aug.augment(train_X_copy)
        test_X_aug = aug.augment(test_X_copy)
        
        for j in range(5):
            train_ts = train_X_aug[j::5, :]
            test_ts = test_X_aug[j::5, :]
            train_X_copy = np.concatenate([train_X_copy, train_ts], axis=1)
            test_X_copy = np.concatenate([test_X_copy, test_ts], axis=1)
        """
        
        train_X_aug, train_y_aug = aug.augment(train_X, train_y)
        regr = make_pipeline(StandardScaler(), PLSRegression(n_components=6))
        regr = regr.fit(train_X_aug, train_y_aug)
        prediction_aug = regr.predict(test_X)
        MAE_15_mins, RMSE_15_mins = evaluation(prediction_aug, test_y, 2)
        MAE_30_mins, RMSE_30_mins = evaluation(prediction_aug, test_y, 5)
        MAE_60_mins, RMSE_60_mins = evaluation(prediction_aug, test_y, 11)
        # print(round(MAE_15_mins, 2), round(RMSE_15_mins, 2), round(MAE_30_mins, 2), round(RMSE_30_mins, 2), round(MAE_60_mins, 2), round(RMSE_60_mins, 2))
        results.append([sensor_list_la[i], augmenter_names[a], MAE_15_mins, RMSE_15_mins, MAE_30_mins, RMSE_30_mins, MAE_60_mins, RMSE_60_mins])
    ########################################

result_df = pd.DataFrame(results, columns=["Sensor_ID", "Method", "MAE_15mins", "RMSE_15mins", "MAE_30mins", "RMSE_30mins", "MAE_60mins", "RMSE_60mins"])
result_df.to_csv("../Results/PLSR_performance_table.csv")