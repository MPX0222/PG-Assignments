from RVM import *


def dataaug_rollingavg(ori_df, rolling_avg_size):
    dataaug_df = (ori_df.rolling(rolling_avg_size).sum())
    return dataaug_df


def train_data_preprocess():
    data_path_1 = r'data/train/lh.MeanCurv - 1600.csv'
    data_path_2 = r'data/train/lh.GausCurv - 1600.csv'
    data_path_3 = r'data/train/rh.MeanCurv- 1600.csv'
    data_path_4 = r'data/train/rh.GausCurv- 1600.csv'

    data_path_5 = r'data/train/lh.ThickAvg - 1600.csv'
    data_path_6 = r'data/train/lh.SurfArea - 1600.csv'
    data_path_7 = r'data/train/rh.ThickAvg- 1600.csv'
    data_path_8 = r'data/train/rh.SurfArea - 1600.csv'

    data_path_9 = r'data/train/lh.GrayVol - 1600.csv'
    data_path_10 = r'data/train/rh.GrayVol- 1600.csv'

    data_path_11 = r'data/train/wmparc - 1600.csv'
    data_path_12 = r'data/train/aseg - 1600.csv'

    label_path = r'data/train/subject_info - 1600.csv'

    original_x1, original_x2, original_x3, original_x4, original_y = pd.read_csv(data_path_1), pd.read_csv(data_path_2), pd.read_csv(data_path_3), pd.read_csv(data_path_4), pd.read_csv(label_path)
    original_x5, original_x6, original_x7, original_x8 = pd.read_csv(data_path_5), pd.read_csv(data_path_6), pd.read_csv(data_path_7), pd.read_csv(data_path_8)
    original_x9, original_x10 = pd.read_csv(data_path_9), pd.read_csv(data_path_10)
    original_x11, original_x12 = pd.read_csv(data_path_11), pd.read_csv(data_path_12)

    original_x1, original_x2, original_x3, original_x4 = original_x1.iloc[:, 1:], original_x2.iloc[:,1:], original_x3.iloc[:,1:], original_x4.iloc[:, 1:]
    original_x5, original_x6, original_x7, original_x8 = original_x5.iloc[:, 1:], original_x6.iloc[:,1:], original_x7.iloc[:,1:], original_x8.iloc[:, 1:]
    original_x9, original_x10 = original_x9.iloc[:, 1:], original_x10.iloc[:, 1:]
    original_x11, original_x12 = original_x11.iloc[:, 1:], original_x12.iloc[:, 1:]
    original_y = original_y.iloc[:, 3]

    # 目前经验是需要标准化
    original_x1 = (original_x1 - original_x1.min()) / (original_x1.max() - original_x1.min())
    original_x2 = (original_x2 - original_x2.min()) / (original_x2.max() - original_x2.min())
    original_x3 = (original_x3 - original_x3.min()) / (original_x3.max() - original_x3.min())
    original_x4 = (original_x4 - original_x4.min()) / (original_x4.max() - original_x4.min())
    original_x5 = (original_x5 - original_x5.min()) / (original_x5.max() - original_x5.min())
    original_x6 = (original_x6 - original_x6.min()) / (original_x6.max() - original_x6.min())
    original_x7 = (original_x7 - original_x7.min()) / (original_x7.max() - original_x7.min())
    original_x8 = (original_x8 - original_x8.min()) / (original_x8.max() - original_x8.min())
    original_x9 = (original_x9 - original_x9.min()) / (original_x9.max() - original_x9.min())
    original_x10 = (original_x10 - original_x10.min()) / (original_x10.max() - original_x10.min())
    original_x11 = (original_x11 - original_x11.min()) / (original_x11.max() - original_x11.min())
    original_x12 = (original_x12 - original_x12.min()) / (original_x12.max() - original_x12.min())

    # 数据增强
    is_dataaug = True

    if is_dataaug:
        rolling_size = 3

        aug_x1, aug_x2, aug_x3, aug_x4, aug_x5, aug_x6 = dataaug_rollingavg(original_x1,rolling_size), dataaug_rollingavg(original_x2, rolling_size), dataaug_rollingavg(original_x3, rolling_size), dataaug_rollingavg(original_x4,rolling_size), dataaug_rollingavg(original_x5, rolling_size), dataaug_rollingavg(original_x6, rolling_size)
        aug_x7, aug_x8, aug_x9, aug_x10, aug_x11, aug_x12 = dataaug_rollingavg(original_x7,rolling_size), dataaug_rollingavg(original_x8, rolling_size), dataaug_rollingavg(original_x9, rolling_size), dataaug_rollingavg(original_x10,rolling_size), dataaug_rollingavg(original_x11, rolling_size), dataaug_rollingavg(original_x12, rolling_size)
        aug_y = dataaug_rollingavg(original_y, rolling_size)

        aug_x1, aug_x2, aug_x3, aug_x4, aug_x5, aug_x6 = aug_x1.dropna(), aug_x2.dropna(), aug_x3.dropna(), aug_x4.dropna(), aug_x5.dropna(), aug_x6.dropna()
        aug_x7, aug_x8, aug_x9, aug_x10, aug_x11, aug_x12 = aug_x7.dropna(), aug_x8.dropna(), aug_x9.dropna(), aug_x10.dropna(), aug_x11.dropna(), aug_x12.dropna()
        aug_y = aug_y.dropna()

        original_x1, original_x2 = pd.concat([original_x1, aug_x1], ignore_index=True), pd.concat([original_x2, aug_x2],ignore_index=True)
        original_x3, original_x4 = pd.concat([original_x3, aug_x3], ignore_index=True), pd.concat([original_x4, aug_x4],ignore_index=True)
        original_x5, original_x6 = pd.concat([original_x5, aug_x5], ignore_index=True), pd.concat([original_x6, aug_x6],ignore_index=True)
        original_x7, original_x8 = pd.concat([original_x7, aug_x7], ignore_index=True), pd.concat([original_x8, aug_x8],ignore_index=True)
        original_x9, original_x10 = pd.concat([original_x9, aug_x9], ignore_index=True), pd.concat([original_x10, aug_x10], ignore_index=True)
        original_x11, original_x12 = pd.concat([original_x11, aug_x11], ignore_index=True), pd.concat([original_x12, aug_x12], ignore_index=True)
        original_y = pd.concat([original_y, aug_y], ignore_index=True)

    # # PCA
    # pca_fitter = PCA(n_components=10)
    # original_x1, original_x2, original_x3, original_x4 = pca_fitter.fit_transform(original_x1), pca_fitter.fit_transform(original_x2), pca_fitter.fit_transform(original_x3), pca_fitter.fit_transform(original_x4)
    # original_x5, original_x6, original_x7, original_x8 = pca_fitter.fit_transform(original_x5), pca_fitter.fit_transform(original_x6), pca_fitter.fit_transform(original_x7), pca_fitter.fit_transform(original_x8)
    # original_x9, original_x10, original_x11, original_x12 = pca_fitter.fit_transform(original_x9), pca_fitter.fit_transform(original_x10), pca_fitter.fit_transform(original_x11), pca_fitter.fit_transform(original_x12)

    #
    # # SELECT-K-BEST
    # k = 10  # 选择最相关的特征数量
    # selector = SelectKBest(f_classif, k=k)
    # original_x1, original_x2, original_x3, original_x4 = selector.fit_transform(original_x1, original_y), selector.fit_transform(original_x2, original_y), selector.fit_transform(original_x3, original_y), selector.fit_transform(original_x4, original_y)

    original_x1, original_x2, original_x3, original_x4 = np.array(original_x1), np.array(original_x2), np.array(
        original_x3), np.array(original_x4)
    original_x5, original_x6, original_x7, original_x8 = np.array(original_x5), np.array(original_x6), np.array(
        original_x7), np.array(original_x8)
    original_x9, original_x10 = np.array(original_x9), np.array(original_x10)
    original_x11, original_x12 = np.array(original_x11), np.array(original_x12)
    original_y = np.array(original_y)

    # original_x = np.concatenate([original_x1 + original_x3, original_x2 + original_x4], axis=1)
    original_x_thick = (original_x5 + original_x7) * 0.5
    original_x_gauscurv = (original_x2 + original_x4) * 0.5
    original_x_meancurv = (original_x1 + original_x3) * 0.5
    original_x_surfarea = (original_x6 + original_x8) * 0.5
    original_x_grayvol = (original_x9 + original_x10) * 0.5
    original_x_wmparc = original_x11
    original_x_aseg = original_x12

    # original_x = np.hstack([original_x_thick, original_x_gauscurv, original_x_meancurv, original_x_surfarea, original_x_grayvol, original_x_wmparc, original_x_aseg])
    # original_x = np.hstack([original_x1, original_x2, original_x3, original_x4, original_x5, original_x6, original_x7, original_x8])
    original_x = np.hstack(
        [original_x_thick, original_x_gauscurv, original_x_meancurv, original_x_surfarea, original_x_grayvol,
         original_x_wmparc, original_x_aseg, original_x1, original_x2, original_x3, original_x4, original_x5,
         original_x6, original_x7, original_x8, original_x9, original_x10])

    return original_x, original_y


def data_preprocess():
    data_path_1 = r'data/test/lh.MeanCurv- 389.csv'
    data_path_2 = r'data/test/lh.GausCurv- 389.csv'
    data_path_3 = r'data/test/rh.MeanCurv- 389.csv'
    data_path_4 = r'data/test/rh.GausCurv- 389.csv'

    data_path_5 = r'data/test/lh.ThickAvg- 389.csv'
    data_path_6 = r'data/test/lh.SurfArea - 389.csv'
    data_path_7 = r'data/test/rh.ThickAvg- 389.csv'
    data_path_8 = r'data/test/rh.SurfArea - 389.csv'

    data_path_9 = r'data/test/lh.GrayVol - 389.csv'
    data_path_10 = r'data/test/rh.GrayVol- 389.csv'

    data_path_11 = r'data/test/wmparc - 389.csv'
    data_path_12 = r'data/test/aseg - 389.csv'

    # label_path = r'data/test/subject_info - 389.csv'

    original_x1, original_x2, original_x3, original_x4 = pd.read_csv(data_path_1), pd.read_csv(
        data_path_2).dropna(), pd.read_csv(data_path_3), pd.read_csv(data_path_4).dropna()
    original_x5, original_x6, original_x7, original_x8 = pd.read_csv(data_path_5), pd.read_csv(
        data_path_6), pd.read_csv(data_path_7), pd.read_csv(data_path_8)
    original_x9, original_x10 = pd.read_csv(data_path_9), pd.read_csv(data_path_10)
    original_x11, original_x12 = pd.read_csv(data_path_11), pd.read_csv(data_path_12)

    original_x1, original_x2, original_x3, original_x4 = original_x1.iloc[:, 1:], original_x2.iloc[:,1:], original_x3.iloc[:,1:], original_x4.iloc[:, 1:]
    original_x5, original_x6, original_x7, original_x8 = original_x5.iloc[:, 1:], original_x6.iloc[:,1:], original_x7.iloc[:,1:], original_x8.iloc[:, 1:]
    original_x9, original_x10 = original_x9.iloc[:, 1:], original_x10.iloc[:, 1:]
    original_x11, original_x12 = original_x11.iloc[:, 1:], original_x12.iloc[:, 1:]
    # original_y = original_y.iloc[:, 3]

    # 目前经验是需要标准化
    original_x1 = (original_x1 - original_x1.min()) / (original_x1.max() - original_x1.min())
    original_x2 = (original_x2 - original_x2.min()) / (original_x2.max() - original_x2.min())
    original_x3 = (original_x3 - original_x3.min()) / (original_x3.max() - original_x3.min())
    original_x4 = (original_x4 - original_x4.min()) / (original_x4.max() - original_x4.min())
    original_x5 = (original_x5 - original_x5.min()) / (original_x5.max() - original_x5.min())
    original_x6 = (original_x6 - original_x6.min()) / (original_x6.max() - original_x6.min())
    original_x7 = (original_x7 - original_x7.min()) / (original_x7.max() - original_x7.min())
    original_x8 = (original_x8 - original_x8.min()) / (original_x8.max() - original_x8.min())
    original_x9 = (original_x9 - original_x9.min()) / (original_x9.max() - original_x9.min())
    original_x10 = (original_x10 - original_x10.min()) / (original_x10.max() - original_x10.min())
    original_x11 = (original_x11 - original_x11.min()) / (original_x11.max() - original_x11.min())
    original_x12 = (original_x12 - original_x12.min()) / (original_x12.max() - original_x12.min())

    original_x1, original_x2, original_x3, original_x4 = np.array(original_x1), np.array(original_x2), np.array(
        original_x3), np.array(original_x4)
    original_x5, original_x6, original_x7, original_x8 = np.array(original_x5), np.array(original_x6), np.array(
        original_x7), np.array(original_x8)
    original_x9, original_x10 = np.array(original_x9), np.array(original_x10)
    original_x11, original_x12 = np.array(original_x11), np.array(original_x12)
    # original_y = np.array(original_y)

    # original_x = np.concatenate([original_x1 + original_x3, original_x2 + original_x4], axis=1)
    original_x_thick = (original_x5 + original_x7) * 0.5
    original_x_gauscurv = (original_x2 + original_x4) * 0.5
    original_x_meancurv = (original_x1 + original_x3) * 0.5
    original_x_surfarea = (original_x6 + original_x8) * 0.5
    original_x_grayvol = (original_x9 + original_x10) * 0.5
    original_x_wmparc = original_x11
    original_x_aseg = original_x12

    # original_x = np.hstack([original_x_thick, original_x_gauscurv, original_x_meancurv, original_x_surfarea, original_x_grayvol, original_x_wmparc, original_x_aseg])
    # original_x = np.hstack([original_x1, original_x2, original_x3, original_x4, original_x5, original_x6, original_x7, original_x8])
    original_x = np.hstack(
        [original_x_thick, original_x_gauscurv, original_x_meancurv, original_x_surfarea, original_x_grayvol,
         original_x_wmparc, original_x_aseg, original_x1, original_x2, original_x3, original_x4, original_x5,
         original_x6, original_x7, original_x8, original_x9, original_x10])

    return original_x


def write_submission(prediction_list, submit_name):
    ori_submission_path = r'subject_info - 389.csv'

    predict_df = pd.read_csv(ori_submission_path, encoding='gbk')
    predict_df['年龄'] = prediction_list

    print('\nSumission CSV Preview')
    print(predict_df.head(10))
    predict_df.to_csv(submit_name + '.csv')

    print('Successfully Build a Submission CSV File')
