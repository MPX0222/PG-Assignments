import torch

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from DataPreprocess import *

def machine_prediction():
    train_original_x, train_original_y = train_data_preprocess()
    test_original_x = data_preprocess()

    # 将数据集分为训练集和测试集
    X_train, X_valid, y_train, y_valid = train_test_split(train_original_x, train_original_y, test_size=0.3, random_state=0)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_valid = torch.tensor(X_train, dtype=torch.float32)
    y_valid = torch.tensor(y_train, dtype=torch.float32)
    test_original_x = torch.tensor(test_original_x, dtype=torch.float32)

    # RVR
    print('\n RVR Model')

    linear_model = RVR(kernel="linear")
    linear_model.fit(X_train, y_train)
    linear_model_valid_prediction = linear_model.predict(X_valid)
    print('\nVALID MAE:{}'.format(mean_absolute_error(y_valid, [int(i) for i in linear_model_valid_prediction])))
    print('VALID Variance {}'.format(np.var([int(i) for i in linear_model_valid_prediction])))
    linear_model_test_prediction = linear_model.predict(test_original_x)
    print([int(i) for i in linear_model_test_prediction])
    print('TEST Variance {}'.format(np.var([int(i) for i in linear_model_test_prediction])))

    rbf_model = RVR(kernel="rbf")
    rbf_model.fit(X_train, y_train)
    rbf_model_valid_prediction = rbf_model.predict(X_valid)
    print('\nVALID MAE:{}'.format(mean_absolute_error(y_valid, [int(i) for i in rbf_model_valid_prediction])))
    print('VALID Variance {}'.format(np.var([int(i) for i in rbf_model_valid_prediction])))
    rbf_model_test_prediction = rbf_model.predict(test_original_x)
    print([int(i) for i in rbf_model_test_prediction])
    print('TEST Variance {}'.format(np.var([int(i) for i in rbf_model_test_prediction])))

    poly_model = RVR(kernel="poly")
    poly_model.fit(X_train, y_train)
    poly_model_valid_prediction = poly_model.predict(X_valid)
    print('\nVALID MAE:{}'.format(mean_absolute_error(y_valid, [int(i) for i in poly_model_valid_prediction])))
    print('VALID Variance {}'.format(np.var([int(i) for i in poly_model_valid_prediction])))
    poly_model_test_prediction = rbf_model.predict(test_original_x)
    print([int(i) for i in poly_model_test_prediction])
    print('TEST Variance {}'.format(np.var([int(i) for i in poly_model_test_prediction])))

    # 随机森林
    print('\n RandomForest Model')

    rf_model = RandomForestRegressor(n_estimators=12, random_state=0)
    rf_model.fit(X_train, y_train)
    rf_model_valid_prediction = rf_model.predict(X_valid)
    print('\nVALID MAE:{}'.format(mean_absolute_error(y_valid, [int(i) for i in rf_model_valid_prediction])))
    print('VALID Variance {}'.format(np.var([int(i) for i in rf_model_valid_prediction])))
    rf_model_test_prediction = rf_model.predict(test_original_x)
    print([int(i) for i in rf_model_test_prediction])
    print('TEST Variance {}'.format(np.var([int(i) for i in rf_model_test_prediction])))

    # Bagging
    print('\n Bagging Model')

    bag_model = BaggingRegressor()
    bag_model.fit(X_train, y_train)
    bag_model_valid_prediction = bag_model.predict(X_valid)
    print('\nVALID MAE:{}'.format(mean_absolute_error(y_valid, [int(i) for i in bag_model_valid_prediction])))
    print('VALID Variance {}'.format(np.var([int(i) for i in bag_model_valid_prediction])))
    bag_model_test_prediction = bag_model.predict(test_original_x)
    print([int(i) for i in bag_model_test_prediction])
    print('TEST Variance {}'.format(np.var([int(i) for i in bag_model_test_prediction])))

    # XGBoost
    print('\n XGBoost Model')

    xgb_model = XGBRegressor(learning_rate=0.2)
    xgb_model.fit(X_train, y_train)
    xgb_model_valid_prediction = xgb_model.predict(X_valid)
    print('\nVALID MAE:{}'.format(mean_absolute_error(y_valid, [int(i) for i in xgb_model_valid_prediction])))
    print('LABEL Variance {}'.format(np.var([int(i) for i in y_valid])))
    print('VALID Variance {}'.format(np.var([int(i) for i in xgb_model_valid_prediction])))
    xgb_model_test_prediction = xgb_model.predict(test_original_x)
    print([int(i) for i in xgb_model_test_prediction])
    print('TEST Variance {}'.format(np.var([int(i) for i in xgb_model_test_prediction])))

    # LightGBM
    print('\n LightGBM Model')

    lgbm_model = LGBMRegressor()
    lgbm_model.fit(X_train, y_train)
    lgbm_model_valid_prediction = lgbm_model.predict(X_valid)
    print('\nVALID MAE:{}'.format(mean_absolute_error(y_valid, [int(i) for i in lgbm_model_valid_prediction])))
    print('LABEL Variance {}'.format(np.var([int(i) for i in y_valid])))
    print('VALID Variance {}'.format(np.var([int(i) for i in lgbm_model_valid_prediction])))
    lgbm_model_test_prediction = lgbm_model.predict(test_original_x)
    print([int(i) for i in lgbm_model_test_prediction])
    print('TEST Variance {}'.format(np.var([int(i) for i in lgbm_model_test_prediction])))

    # 集成模型
    print('\n Ensemble Model')

    new_linear_pred, new_rbf_pred, new_poly_pred = [int(i) for i in linear_model_valid_prediction], [int(i) for i in rbf_model_valid_prediction], [int(i) for i in poly_model_valid_prediction]
    new_rf_pred = [int(i) for i in rf_model_valid_prediction]
    new_bag_pred = [int(i) for i in bag_model_valid_prediction]
    new_xgb_pred = [int(i) for i in xgb_model_valid_prediction]
    new_lgbm_pred = [int(i) for i in lgbm_model_valid_prediction]

    # ensemble_prediction = [(i + j) / 2 for i, j in zip(new_rf_pred, new_rbf_pred)]
    ensemble_prediction = [(i + j + k) / 3 for i, j, k in zip(new_xgb_pred, new_bag_pred, new_linear_pred)]
    print('\nENSEMBLE VALID MAE:{}'.format(mean_absolute_error(y_valid, ensemble_prediction)))
    print('ENSEMBLE VALID Variance {}'.format(np.var([int(i) for i in ensemble_prediction])))

    test_new_linear_pred, test_new_rbf_pred, test_new_poly_pred = [int(i) for i in linear_model_test_prediction], [
        int(i) for i in rbf_model_test_prediction], [int(i) for i in poly_model_test_prediction]
    test_new_bag_pred = [int(i) for i in bag_model_test_prediction]
    test_new_rf_pred = [int(i) for i in rf_model_test_prediction]
    test_new_xgb_pref = [int(i) for i in xgb_model_test_prediction]

    # test_ensemble_prediction = [(i + j) / 2 for i, j in zip(test_new_rf_pred, test_new_rbf_pred)]
    test_ensemble_prediction = [(i + j + k) / 3 for i, j, k in
                                zip(test_new_xgb_pref, test_new_bag_pred, test_new_linear_pred)]
    print([int(i) for i in test_ensemble_prediction])
    print('TEST Variance {}'.format(np.var([int(i) for i in test_ensemble_prediction])))

    # 写结果
    # final_list = [int(i) for i in [int(i) for i in lgbm_model_test_prediction]]
    # submission_version_name = 'Ensemble-DataAug3-XGBoost_Bagging_RVR'
    # write_submission(final_list, submission_version_name)


machine_prediction()