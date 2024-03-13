from sklearn.base import OutlierMixin
from sklearn.metrics.pairwise import KERNEL_PARAMS
from xgboost import XGBRegressor
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.impute import KNNImputer
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.kernel_ridge import KernelRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.gaussian_process import GaussianProcessRegressor

###############READ THE DATA#######################
x_data = pd.read_csv('X_train.csv')
y_data = pd.read_csv('y_train.csv')
x_test = pd.read_csv('X_test.csv')

print('Pandas read shape = ' + str(x_data.shape))

print('Test Set',np.argwhere(np.any(np.isnan(x_test))))

x_data = x_data.to_numpy()[:, 1:]  #convert the data frame to numpy array
y_data = y_data.to_numpy()[:, 1]  #convert the data frame to numpy array\
x_test = x_test.to_numpy()[:, 1:]

print('Numpy shape = ' + str(x_data.shape))

#USED TO SPLIT HERE

###############PREPROCESS THE DATA################

#Eliminate samples and fearures with too many Nans
# samples_nan = np.zeros(x_data.shape[0])
# features_nan = np.zeros(x_data.shape[1])
# samples_threshold = 90
# features_threshold = 120

# counter=0
# for row in x_data:
#     samples_nan[counter] = np.sum(np.isnan(row)) 
#     counter += 1

# counter=0
# for col in x_data.T:
#     features_nan[counter] = np.sum(np.isnan(col)) 
#     counter += 1

# #print('For samples',np.sort(samples_nan)[-100:][::-1])
# #print('for features',np.sort(features_nan)[-100:][::-1])
# # plt.figure()
# # plt.plot(np.sort(samples_nan)[::-1])
# # plt.show()
# # plt.figure()
# # plt.plot(np.sort(features_nan)[::-1])
# # plt.show()

# x_data = x_data[:,features_nan < features_threshold]
# x_data = x_data[samples_nan < samples_threshold,:]
# y_data = y_data[samples_nan < samples_threshold]
# print('After Deleting Too Many Nans', x_data.shape)

#Eliminate Features with Abnormal Variance
std_dev_upper_bound = 1e5

#selector = VarianceThreshold(variance_upper_bound)
#x_train = selector.fit_transform(x_train)
#print(x_train.shape)

feature_std_devs = np.nanstd(x_data, axis=0)
plt.figure()
plt.plot(feature_std_devs)
plt.show()
#print(feature_std_devs)

x_data = x_data[:,feature_std_devs <= std_dev_upper_bound]
x_test = x_test[:,feature_std_devs <= std_dev_upper_bound]

feature_std_devs = np.nanstd(x_data, axis=0)
plt.figure()
plt.plot(feature_std_devs)
plt.show()
#print(feature_std_devs)

print('After Removing Highly Deviating Features',x_data.shape)

################SPLIT THE DATA####################
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.001, random_state=36)


################SCALING THE DATA###############
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

###############IMPUTATION####################
imp = SimpleImputer(missing_values=np.nan, strategy='median')
#imp = KNNImputer(n_neighbors = 10, weights = 'distance')
x_train_imputed = imp.fit_transform(x_train)
x_val_imputed = imp.transform(x_val)
x_test = imp.transform(x_test)

#imp_estimator = XGBRegressor(n_estimators = 120, learning_rate=0.11,subsample=0.8, colsample_bynode=0.3, reg_alpha=1.0, reg_lambda=0.01, gamma=0.1, random_state=42, max_depth = 5)
#kernel = ConstantKernel(constant_value=5, constant_value_bounds="fixed")*RBF(5, length_scale_bounds="fixed")+ConstantKernel(constant_value=5, constant_value_bounds="fixed")
#imp_estimator = SVR(kernel=kernel,C=5)
#imp_estimator = SVR(kernel='rbf', C=40, gamma=0.005)
#imp_estimator = XGBRegressor(n_estimators = 120, learning_rate=0.05,subsample=0.6, colsample_bynode=0.2, reg_alpha=1.0, reg_lambda=10.0, gamma=0.1, random_state=42, max_depth = 4)
#imp_estimator = RandomForestRegressor(n_estimators = 800, max_depth=24, random_state=42, max_features = int(0.5*x_train.shape[1]), min_samples_leaf=2, min_samples_split=5)
# imp_estimator = XGBRegressor(n_estimators = 120, learning_rate=0.05,subsample=0.7, colsample_bynode=0.6, reg_alpha=5.0, reg_lambda=1.0, gamma=0.15, random_state=42, max_depth = 6)
# kernel = ConstantKernel(constant_value=0.224**2, constant_value_bounds="fixed")*RBF(5, length_scale_bounds="fixed")+ConstantKernel(constant_value=2.4**2, constant_value_bounds="fixed")
# imp_estimator = GaussianProcessRegressor(kernel=kernel, alpha=1e-3)
# iterative_imputer = IterativeImputer(random_state=0, estimator=imp_estimator, initial_strategy='median',verbose=2,max_iter = 10)
# iterative_imputer.fit(x_train)
# x_train_imputed = iterative_imputer.transform(x_train)
# x_val_imputed = iterative_imputer.transform(x_val)
# x_test = iterative_imputer.transform(x_test)
# print('imputation completed')
################OUTLIER ELIMINATION###############
print(np.any(np.isnan(x_train_imputed)))
kept_indices = IsolationForest(n_estimators=15,random_state=36).fit_predict(x_train_imputed) == 1
#outlier_detector = LocalOutlierFactor(n_neighbors=10)
#kept_indices = outlier_detector.fit_predict(x_train_imputed)==1

#outlier_factor = LocalOutlierFactor(n_neighbors=40)
#kept_indices = outlier_factor.fit_predict(x_train) == 1

print('Removed ' + str(len(kept_indices) - kept_indices.sum()))
x_train_eliminated = x_train_imputed[kept_indices]
y_train_eliminated = y_train[kept_indices]

print('Remaining ' + str(x_train_eliminated.shape[0]))

###############FEATURE SELECTION################
variance_lower_bound = np.max(np.var(x_train_eliminated, axis=0))*1e-1

selector = VarianceThreshold(variance_lower_bound)
x_train_high_var = selector.fit_transform(x_train_eliminated)
x_val_high_var = selector.transform(x_val_imputed)
x_test = selector.transform(x_test)

feature_vars = np.var(x_train_high_var, axis=0)
plt.figure()
plt.plot(feature_vars)
plt.show()

print('After Removing Low Variance Features:', x_train_high_var.shape)

correlation_threshold = 0.1
cov_matrix = np.matmul(np.transpose(x_train_high_var), x_train_high_var) / float(x_train_high_var.shape[0])
print('Maxmimum of cov matrix', np.max(cov_matrix))

correlated_feature_indices = []
for i in range(int(cov_matrix.shape[1])):
    for j in range(i):
        if (cov_matrix[i,j] >= cov_matrix[i,i] - correlation_threshold and \
            cov_matrix[i,j] <= cov_matrix[i,i] + correlation_threshold) or \
        (cov_matrix[i,j] <= -cov_matrix[i,i] + correlation_threshold and \
            cov_matrix[i,j] >= -cov_matrix[i,i] - correlation_threshold) and i != j:
            correlated_feature_indices.append(j)

#print(correlated_feature_indices)

print(list(set(correlated_feature_indices)))
x_train_high_var = np.delete(x_train_high_var, list(set(correlated_feature_indices)), axis=1)
x_val_high_var = np.delete(x_val_high_var, list(set(correlated_feature_indices)), axis=1)
x_test = np.delete(x_test, list(set(correlated_feature_indices)), axis=1)

print('After Removing Highly Correlated Features:', x_train_high_var.shape)
            

#feature_selector = RandomForestRegressor(n_estimators = 100, max_depth=15, random_state=0, max_features = 0.4*x_train_high_var.shape[1])

#feature_selector_1 = XGBRegressor(n_estimators = 30, learning_rate=0.05,subsample=0.3, colsample_bynode=0.2, reg_alpha=10.0, reg_lambda=2.0, gamma=0.1, random_state=42, max_depth = 4)
feature_selector_1 = XGBRegressor(n_estimators = 120, learning_rate=0.05,subsample=0.7, colsample_bynode=0.6, reg_alpha=5.0, reg_lambda=1.0, gamma=0.15, random_state=42, max_depth = 6)
feature_selector_1.fit(x_train_high_var, y_train_eliminated)
print('1')
#feature_selector_2 = XGBRegressor(n_estimators = 120, learning_rate=0.11,subsample=0.9, colsample_bynode=0.3, reg_alpha=5.0, reg_lambda=2.0, gamma=0.1, random_state=45, max_depth = 5)
feature_selector_2 = XGBRegressor(n_estimators = 200, learning_rate=0.05,subsample=0.6, colsample_bynode=0.4, reg_alpha=3.0, reg_lambda=1.0, gamma=0.15, random_state=42, max_depth = 5)
feature_selector_2.fit(x_train_high_var, y_train_eliminated)
print('2')
#feature_selector_3 = XGBRegressor(n_estimators = 50, learning_rate=0.01,subsample=0.3, colsample_bynode=0.2, reg_alpha=20.0, reg_lambda=5.0, gamma=0.1, random_state=67, max_depth = 2)
feature_selector_3 = XGBRegressor(n_estimators = 240, learning_rate=0.05,subsample=0.7, colsample_bynode=0.4, reg_alpha=5.0, reg_lambda=1.0, gamma=0.18, random_state=42, max_depth = 6)
feature_selector_3.fit(x_train_high_var, y_train_eliminated)
print('3')
#feature_selector_4 = XGBRegressor(n_estimators = 90, learning_rate=0.09,subsample=0.6, colsample_bynode=0.5, reg_alpha=1.0, reg_lambda=1.0, gamma=0.1, random_state=58, max_depth = 5)
feature_selector_4 = XGBRegressor(n_estimators = 150, learning_rate=0.05,subsample=0.7, colsample_bynode=0.6, reg_alpha=5.0, reg_lambda=1.0, gamma=0.1, random_state=42, max_depth = 5)
feature_selector_4.fit(x_train_high_var, y_train_eliminated)
print('4')
#feature_selector_5 = XGBRegressor(n_estimators = 200, learning_rate=0.15,subsample=0.7, colsample_bynode=0.5, reg_alpha=1.0, reg_lambda=1.0, gamma=0.1, random_state=45, max_depth = 6)
feature_selector_5 = XGBRegressor(n_estimators = 120, learning_rate=0.05,subsample=0.6, colsample_bynode=0.2, reg_alpha=1.0, reg_lambda=10.0, gamma=0.1, random_state=42, max_depth = 4)
feature_selector_5.fit(x_train_high_var, y_train_eliminated)
print('5')

# feature_selector_1 = LassoCV(cv=6)
# feature_selector_1.fit(x_train_high_var, y_train_eliminated)
# print('1')
# feature_selector_2 = LassoCV(cv=6)
# feature_selector_2.fit(x_train_high_var, y_train_eliminated)
# print('2')
# feature_selector_3 = LassoCV(cv=6)
# feature_selector_3.fit(x_train_high_var, y_train_eliminated)
# print('3')
# feature_selector_4 = LassoCV(cv=6)
# feature_selector_4.fit(x_train_high_var, y_train_eliminated)
# print('4')
# feature_selector_5 = LassoCV(cv=6)
# feature_selector_5.fit(x_train_high_var, y_train_eliminated)
# print('5')
#Use ENSEMBLES TO DECIDE ON FEATURES AS SEEN ON THE NET

averaged_feature_importances = (feature_selector_1.feature_importances_ + \
    feature_selector_2.feature_importances_ + \
        feature_selector_3.feature_importances_ + \
            feature_selector_4.feature_importances_ + \
                feature_selector_5.feature_importances_ ) / 5.0


plt.figure()
#plt.plot(np.sort(feature_selector_1.feature_importances_)[::-1])
plt.plot(np.sort(averaged_feature_importances)[::-1])
plt.show()

num_selected_features = 70
#selected_feature_indices = feature_selector.feature_importances_.argsort()[-num_selected_features:][::-1]
selected_feature_indices = averaged_feature_importances.argsort()[-num_selected_features:][::-1]

plt.figure()
#plt.plot(np.sort(feature_selector_1.feature_importances_)[::-1])
plt.plot(averaged_feature_importances[selected_feature_indices])
plt.show()


x_train_final = x_train_high_var[:,selected_feature_indices]
x_val_final = x_val_high_var[:,selected_feature_indices]
x_test = x_test[:,selected_feature_indices]

###################TRAINING#######################

#model_1 = XGBRegressor(n_estimators = 120, learning_rate=0.09,subsample=0.6, colsample_bynode=0.2, reg_alpha=1.0, reg_lambda=10.0, gamma=0.1, random_state=42, max_depth = 4)
model_1 = XGBRegressor(n_estimators = 120, learning_rate=0.05,subsample=0.7, colsample_bynode=0.6, reg_alpha=5.0, reg_lambda=1.0, gamma=0.15, random_state=42, max_depth = 6)
model_1.fit(x_train_final,y_train_eliminated)
y_train_pred_1 = model_1.predict(x_train_final)
y_val_pred_1 = model_1.predict(x_val_final)
y_test_pred_1 = model_1.predict(x_test)
print('Val_score_xgboost1',r2_score(y_val, y_val_pred_1))
print('Train_score_xgboost1',r2_score(y_train_eliminated, y_train_pred_1))

#model_3 = XGBRegressor(n_estimators = 100, learning_rate=0.09,subsample=0.6, colsample_bynode=0.2, reg_alpha=5.0, reg_lambda=10.0, gamma=0.1, random_state=42, max_depth = 6)
model_3 = XGBRegressor(n_estimators = 200, learning_rate=0.05,subsample=0.6, colsample_bynode=0.4, reg_alpha=3.0, reg_lambda=1.0, gamma=0.15, random_state=42, max_depth = 5)
model_3.fit(x_train_final,y_train_eliminated)
y_train_pred_3 = model_3.predict(x_train_final)
y_val_pred_3 = model_3.predict(x_val_final)
y_test_pred_3 = model_3.predict(x_test)
print('Val_score_xgboost_2',r2_score(y_val, y_val_pred_3))
print('Train_score_xgboost_2',r2_score(y_train_eliminated, y_train_pred_3))

#kernel = ConstantKernel(constant_value=5, constant_value_bounds="fixed")*RBF(5, length_scale_bounds="fixed")+ConstantKernel(constant_value=5, constant_value_bounds="fixed")
#model_2 = SVR(kernel=kernel,C=5)
model_2 = XGBRegressor(n_estimators = 240, learning_rate=0.05,subsample=0.7, colsample_bynode=0.4, reg_alpha=5.0, reg_lambda=1.0, gamma=0.18, random_state=42, max_depth = 6)
model_2.fit(x_train_final,y_train_eliminated)
y_train_pred_2 = model_2.predict(x_train_final)
y_val_pred_2 = model_2.predict(x_val_final)
y_test_pred_2 = model_2.predict(x_test)
print('Val_score_svr',r2_score(y_val, y_val_pred_2))
print('Train_score_svr',r2_score(y_train_eliminated, y_train_pred_2))

#model_4 = LassoCV(cv=6)
#model_4 = XGBRegressor(n_estimators = 120, learning_rate=0.09,subsample=0.7, colsample_bynode=0.3, reg_alpha=5.0, reg_lambda=10.0, gamma=0.1, random_state=42, max_depth = 4)
model_4 = XGBRegressor(n_estimators = 150, learning_rate=0.05,subsample=0.7, colsample_bynode=0.6, reg_alpha=5.0, reg_lambda=1.0, gamma=0.1, random_state=42, max_depth = 5)
model_4.fit(x_train_final,y_train_eliminated)
y_train_pred_4 = model_4.predict(x_train_final)
y_val_pred_4 = model_4.predict(x_val_final)
y_test_pred_4 = model_4.predict(x_test)
print('Val_score_svr_2',r2_score(y_val, y_val_pred_4))
print('Train_score_svr_2',r2_score(y_train_eliminated, y_train_pred_4))

model_5 = XGBRegressor(n_estimators = 120, learning_rate=0.05,subsample=0.6, colsample_bynode=0.2, reg_alpha=1.0, reg_lambda=10.0, gamma=0.1, random_state=42, max_depth = 4)
#model = RandomForestRegressor(n_estimators = 200, max_depth=15, random_state=0)
model_5.fit(x_train_final,y_train_eliminated)
y_train_pred_5 = model_5.predict(x_train_final)
y_val_pred_5 = model_5.predict(x_val_final)
y_test_pred_5 = model_5.predict(x_test)
print('Val_score_xgboost_3',r2_score(y_val, y_val_pred_5))
print('Train_score_xgboost_3',r2_score(y_train_eliminated, y_train_pred_5))

#model_6 = RandomForestRegressor(n_estimators = 80, max_depth=10, random_state=0, max_features = int(0.4*x_train_final.shape[1]))
#model_6 = RandomForestRegressor(n_estimators = 800, max_depth=24, random_state=42, max_features = int(0.5*x_train_final.shape[1]), min_samples_leaf=2, min_samples_split=5)
#model_6 = XGBRegressor(n_estimators = 240, learning_rate=0.05,subsample=0.7, colsample_bynode=0.7, reg_alpha=5.0, reg_lambda=0.5, gamma=0.19, random_state=42, max_depth = 7)
#kernel_6 = ConstantKernel(constant_value=0.224**2, constant_value_bounds="fixed")*RBF(5, length_scale_bounds="fixed")+ConstantKernel(constant_value=2.4**2, constant_value_bounds="fixed")
#model_6 = GaussianProcessRegressor(kernel=kernel_6, alpha=1e-3)
kernel_6 = ConstantKernel(constant_value=0.447**2, constant_value_bounds="fixed")*RBF(3, length_scale_bounds="fixed")+ConstantKernel(constant_value=2.65**2, constant_value_bounds="fixed")+Matern(length_scale = 10, nu = 2.5, length_scale_bounds="fixed")
model_6 = GaussianProcessRegressor(kernel = kernel_6, alpha=5e-3)
model_6.fit(x_train_final,y_train_eliminated)
y_train_pred_6 = model_6.predict(x_train_final)
y_val_pred_6 = model_6.predict(x_val_final)
y_test_pred_6 = model_6.predict(x_test)
print('Val_score_RF',r2_score(y_val, y_val_pred_6))
print('Train_score_RF',r2_score(y_train_eliminated, y_train_pred_6))

#model_7 = RandomForestRegressor(n_estimators = 100, max_depth=12, random_state=10, max_features = int(0.3*x_train_final.shape[1]))
#model_7 = RandomForestRegressor(n_estimators = 800, max_depth=20, random_state=42, max_features = 31, min_samples_split=6, min_samples_leaf=2)
#model_7 = XGBRegressor(n_estimators = 320, learning_rate=0.05,subsample=0.7, colsample_bynode=0.6, reg_alpha=5.0, reg_lambda=1.0, gamma=0.18, random_state=42, max_depth = 6)
#kernel_7 = ConstantKernel(constant_value=0.316**2, constant_value_bounds="fixed")*RBF(5, length_scale_bounds="fixed")+ConstantKernel(constant_value=2.4**2, constant_value_bounds="fixed")
#model_7 = GaussianProcessRegressor(kernel=kernel_7, alpha=5e-3)
kernel_7 = ConstantKernel(constant_value=0.548**2, constant_value_bounds="fixed")*RBF(4, length_scale_bounds="fixed")+ConstantKernel(constant_value=2.45**2, constant_value_bounds="fixed")+Matern(length_scale = 10, nu = 5, length_scale_bounds="fixed")
model_7 = GaussianProcessRegressor(kernel = kernel_7, alpha=2e-3)
model_7.fit(x_train_final,y_train_eliminated)
y_train_pred_7 = model_7.predict(x_train_final)
y_val_pred_7 = model_7.predict(x_val_final)
y_test_pred_7 = model_7.predict(x_test)
print('Val_score_RF',r2_score(y_val, y_val_pred_7))
print('Train_score_RF',r2_score(y_train_eliminated, y_train_pred_7))

#model_8 = RandomForestRegressor(n_estimators = 120, max_depth=15, random_state=42, max_features = int(0.2*x_train_final.shape[1]))
#model_8 = RandomForestRegressor(n_estimators = 625, max_depth=66, random_state=42, max_features = 28, min_samples_leaf=2, min_samples_split=2)
#kernel_8 = ConstantKernel(constant_value=0.265**2, constant_value_bounds="fixed")*RBF(6, length_scale_bounds="fixed")+ConstantKernel(constant_value=2.83**2, constant_value_bounds="fixed")
#model_8 = GaussianProcessRegressor(kernel=kernel_8, alpha=6e-3)
kernel_8 = ConstantKernel(constant_value=0.548**2, constant_value_bounds="fixed")*RBF(4, length_scale_bounds="fixed")+ConstantKernel(constant_value=1.73**2, constant_value_bounds="fixed")+Matern(length_scale = 10, nu = 2.5, length_scale_bounds="fixed")
model_8 = GaussianProcessRegressor(kernel = kernel_8, alpha=1e-2)
model_8.fit(x_train_final,y_train_eliminated)
y_train_pred_8 = model_8.predict(x_train_final)
y_val_pred_8 = model_8.predict(x_val_final)
y_test_pred_8 = model_8.predict(x_test)
print('Val_score_RF',r2_score(y_val, y_val_pred_8))
print('Train_score_RF',r2_score(y_train_eliminated, y_train_pred_8))

#model_9 = RandomForestRegressor(n_estimators = 900, max_depth=20, random_state=42, max_features = 49, min_samples_leaf=2, min_samples_split=5)
#model_9 = XGBRegressor(n_estimators = 360, learning_rate=0.03,subsample=0.7, colsample_bynode=0.5, reg_alpha=5.0, reg_lambda=1.0, gamma=0.2, random_state=42, max_depth = 6)
#kernel_9 = ConstantKernel(constant_value=0.316**2, constant_value_bounds="fixed")*RBF(5, length_scale_bounds="fixed")+ConstantKernel(constant_value=0.316**2, constant_value_bounds="fixed")
#model_9 = GaussianProcessRegressor(kernel=kernel_9, alpha=1e-2)
kernel_9 = ConstantKernel(constant_value=0.632**2, constant_value_bounds="fixed")*RBF(4, length_scale_bounds="fixed")+ConstantKernel(constant_value=2.45**2, constant_value_bounds="fixed")+Matern(length_scale = 10, nu = 5, length_scale_bounds="fixed")
model_9 = GaussianProcessRegressor(kernel = kernel_9, alpha=2e-3)
model_9.fit(x_train_final,y_train_eliminated)
y_train_pred_9 = model_9.predict(x_train_final)
y_val_pred_9 = model_9.predict(x_val_final)
y_test_pred_9 = model_9.predict(x_test)
print('Val_score_RF',r2_score(y_val, y_val_pred_9))
print('Train_score_RF',r2_score(y_train_eliminated, y_train_pred_9))

#model_10 = XGBRegressor(n_estimators = 360, learning_rate=0.05,subsample=0.7, colsample_bynode=0.6, reg_alpha=5.0, reg_lambda=1.0, gamma=0.18, random_state=42, max_depth = 6)
#kernel_10 = ConstantKernel(constant_value=0.707**2, constant_value_bounds="fixed")*RBF(5, length_scale_bounds="fixed")+ConstantKernel(constant_value=2.4**2, constant_value_bounds="fixed")
#model_10 = GaussianProcessRegressor(kernel=kernel_10, alpha=5e-3)
kernel_10 = ConstantKernel(constant_value=0.632**2, constant_value_bounds="fixed")*RBF(4, length_scale_bounds="fixed")+ConstantKernel(constant_value=2.65**2, constant_value_bounds="fixed")+Matern(length_scale = 10, nu = 5, length_scale_bounds="fixed")
model_10 = GaussianProcessRegressor(kernel = kernel_10, alpha=6e-3)
model_10.fit(x_train_final,y_train_eliminated)
y_train_pred_10 = model_10.predict(x_train_final)
y_val_pred_10 = model_10.predict(x_val_final)
y_test_pred_10 = model_10.predict(x_test)
print('Val_score_RF',r2_score(y_val, y_val_pred_10))
print('Train_score_RF',r2_score(y_train_eliminated, y_train_pred_10))

y_train_predictions = np.array([y_train_pred_1, y_train_pred_2, y_train_pred_3, y_train_pred_4, y_train_pred_5,y_train_pred_6, y_train_pred_7, y_train_pred_8, y_train_pred_9, y_train_pred_10])
y_val_predictions = np.array([y_val_pred_1, y_val_pred_2, y_val_pred_3, y_val_pred_4, y_val_pred_5,y_val_pred_6, y_val_pred_7, y_val_pred_8, y_val_pred_9, y_val_pred_10])
y_test_predictions = np.array([y_test_pred_1, y_test_pred_2, y_test_pred_3, y_test_pred_4, y_test_pred_5,y_test_pred_6, y_test_pred_7, y_test_pred_8, y_test_pred_9, y_test_pred_10])
print(y_val_predictions.shape)
#eights = (1.0/10)*np.ones(10)
weight_array = [1,1,1,1,1,2,2,2,2,2]
weights = weight_array/(np.sum(weight_array))
y_train_overall = np.average(y_train_predictions, weights = weights, axis=0)
y_val_overall = np.average(y_val_predictions, weights = weights, axis=0)
y_test_overall = np.average(y_test_predictions, weights = weights, axis=0)
#y_train_overall = np.mean(y_train_predictions, axis=0)
#y_val_overall = np.mean(y_val_predictions, axis=0)

#y_train_overall = 0.2 * y_train_pred_1 + 0.2 * y_train_pred_2 + 0.2 * y_train_pred_3 + 0.2 * y_train_pred_5 + 0.2 * y_train_pred_4
#y_val_overall = 0.2 * y_val_pred_1 + 0.2 * y_val_pred_2 + 0.2 * y_val_pred_3 + + 0.2 * y_val_pred_5 + 0.2 * y_val_pred_4
print('Val_score_overall',r2_score(y_val, y_val_overall))
print('Train_score_overall',r2_score(y_train_eliminated, y_train_overall))

header = ['id','y']
index_array = np.arange(y_test_overall.shape[0])
submission = np.vstack((index_array,y_test_overall))
print(submission)
print(submission.shape)
df = pd.DataFrame(submission.T)
df.to_csv('submission.csv', index=False, header=header)