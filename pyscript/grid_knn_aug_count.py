import scipy
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import time
import multiprocessing
from sklearn.neighbors import KNeighborsClassifier

NUM_CORES = multiprocessing.cpu_count()/2

TRAIN_FILE = '../input/train.csv'
TEST_FILE = '../input/test.csv'

TRAIN_TEST_HASH_FILE = '../input/train_test_hash.csv'

timestamp =  datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# VALID_LOG_FILE = '../valid/valid_log_knn_20_{0:s}.csv'.format(timestamp)
# VALID_OUTPUT_FILE = '../valid/valid_score_knn_20_{0:s}.csv'.format(timestamp)
TEST_OUTPUT_FILE = '../train/test_score_knn_count_20_{0:s}.csv'.format(timestamp)
# SUBMISSION_FILE = '../output/submission_knn_20_{0:s}.csv'.format(timestamp)

# # writing header
# with open(SUBMISSION_FILE, 'w') as f:
#     f.write("row_id,place_id\n")

# with open(VALID_OUTPUT_FILE, 'w') as f:
#     f.write("row_id,label,prob\n")

with open(TEST_OUTPUT_FILE, 'w') as f:
    f.write("row_id,label,prob\n")

# with open(VALID_LOG_FILE, 'w') as f:
#     f.write("hash,score\n")

# periodic encoder
weekday_encoder = np.zeros((7,7))
for i in range(7):
    weekday_encoder[i, i] = 0.5
    for j in range(7):
        if j == i:
            continue
        if (j > i and j <= i + 3) or (j <= i-4):
            weekday_encoder[i, j] = 0
        else:
            weekday_encoder[i, j] = 1

month_encoder = np.zeros((12,12))
for i in range(12):
#     test_mat[i, i] = 0.5
    for j in range(12):
        if j == i:
            continue
        if (j > i and j < i + 6) or (j < i-6):
            month_encoder[i, j] = 0
        else:
            month_encoder[i, j] = 1

# minute_encoder = np.zeros((1440,1440))
# for i in range(1440):
# #     test_mat[i, i] = 0.5
#     for j in range(1440):
#         if j == i:
#             continue
#         if (j > i and j < i + 720) or (j < i - 720):
#             minute_encoder[i, j] = 0
#         else:
#             minute_encoder[i, j] = 1

minute_encoder = np.zeros((288, 288))
for i in range(288):
#     test_mat[i, i] = 0.5
    for j in range(288):
        if j == i:
            continue
        if (j > i and j < i + 144) or (j < i - 144):
            minute_encoder[i, j] = 0
        else:
            minute_encoder[i, j] = 1            

def update_validation_log(cur_hash, valid_score):
    with open(VALID_LOG_FILE,'a') as f:
        f.write("%d,%f\n" %(cur_hash, valid_score))

def update_validation_output(y_valid_sort, yprob_valid, local_row_id_train, valid_test_rows, local_place_id_invtrans_dict):
    with open(VALID_OUTPUT_FILE,'a') as f:
        for i, irows in enumerate(valid_test_rows):
            f.write("{0:d},".format(local_row_id_train[irows]))
            for j in xrange(20):
                cur_place_id =  local_place_id_invtrans_dict[y_valid_sort[i, j]]
                cur_prob = yprob_valid[i, y_valid_sort[i, j]]
                f.write("%d,%f" %(cur_place_id, cur_prob))
                if (j < 19):
                    f.write(",")
                else:
                    f.write("\n")

def update_test_output(y_sort, yprob, local_row_id_test, local_place_id_invtrans_dict):
    with open(TEST_OUTPUT_FILE,'a') as f:
        for i in range(local_row_id_test.shape[0]):
            f.write("{0:d},".format(local_row_id_test[i]))
            for j in xrange(20):
                cur_place_id =  local_place_id_invtrans_dict[y_sort[i, j]]
                cur_prob = yprob[i, y_sort[i, j]]
                f.write("%d,%f" %(cur_place_id, cur_prob))
                if (j < 19):
                    f.write(",")
                else:
                    f.write("\n")

def update_submission(y_sort, local_row_id_test, local_place_id_invtrans_dict):
    with open(SUBMISSION_FILE, 'a') as f:
        for i in range(local_row_id_test.shape[0]):
            f.write("{0:d},".format(local_row_id_test[i]))
            for j in xrange(3):
                cur_place_id =  local_place_id_invtrans_dict[y_sort[i, j]]
                f.write("%d" %(cur_place_id))
                if (j < 2):
                    f.write(" ")
                else:
                    f.write("\n")


def map3eval(preds, dtrain):
    actual = dtrain.get_label()
    predicted = preds.argsort(axis=1)[:,-np.arange(1,4)]
    metric = 0.
    for i in range(3):
        metric += float(np.sum(actual==predicted[:,i]))/(i+1)
    metric /= actual.shape[0]
    return 'map@3', metric

def map3(local_yvalid_test, y_valid_sort):   
    metric = 0.
    for i in range(3):
        metric += float(np.sum(local_yvalid_test==y_valid_sort[:,i]))/(i+1)
    metric /= local_yvalid_test.shape[0]

    return metric

def load_data(data_name):
    types = {'row_id': np.dtype(np.int32),
         'x': np.dtype(float),
         'y' : np.dtype(float),
         'accuracy': np.dtype(np.int16),
         'place_id': np.int64,
         'time': np.dtype(np.int32)}
    df = pd.read_csv(data_name, dtype=types, index_col = 0, na_filter=False)
    return df

def calculate_distance(distances):
    return distances ** -2

def mydist(x, y):
    dist = abs(x[0] - y[0]) * 512
    dist += abs(x[1] - y[1]) * 1024
    dist += abs(x[2] - y[2]) * 10
    dist += min(abs(x[3] - y[3]) % 24, 24 - abs(y[3] - x[3]) % 24) * 4.0
    dist += min(abs(x[4] - y[4]) % 7, 7 - abs(y[4] - x[4]) % 7) * 3.0
    dist += min(abs(x[5] - y[5]) % 12, 12 - abs(y[5] - x[5]) % 12) * 2.0
    dist += abs(x[6] - y[6]) * 10.0
    
    return dist


def fetch_grid_1d_inner(range_x, lower, upper):
    for ix in range(range_x):
        x_min = lower + float(ix) / range_x * (upper - lower)
        x_max = lower + (float(ix) + 1) / range_x * (upper - lower)

        if x_max == 10:
            x_max += 0.0001

        yield [x_min, x_max]

def fetch_grid_1d(range_x):
    for ix in range(range_x):
        x_min = float(ix) / range_x * 10
        x_max = (float(ix) + 1) / range_x * 10

        if ix == range_x - 1:
            x_max += 0.0001

        yield [x_min, x_max]

def fetch_grid_1d_shift(range_x):
    for ix in range(range_x - 1):
        x_min = (float(ix) + 0.5) / range_x * 10
        x_max = (float(ix) + 1.5) / range_x * 10
        
        x_min = x_min if ix > 0 else 0
        x_max = x_max if ix < range_x - 2 else 10

        if ix == range_x - 2:
            x_max += 0.0001

        yield [x_min, x_max]


def period_encode(local_X, encode_mat, target, off_set):
    num_code = encode_mat.shape[0]
    for i in range(num_code):
        cur_enocder = np.vectorize(lambda x: encode_mat[i, x])
        local_X[:, i + off_set] = cur_enocder(target)


def run_local_knn(df_train_local, df_test_local, anchors, local_histogram):
    [x_min, x_max, y_min, y_max] = anchors

    num_sample = df_train_local.shape[0]
    num_test = df_test_local.shape[0]

    # generate features for training
    local_Xtrain = np.zeros((num_sample, 311))
    local_Xtrain_time = df_train_local['time'].values
    local_Xtrain[:, :3] =  df_train_local[['x', 'y', 'accuracy']].values
    local_Xtrain[:, 2] = np.log10(local_Xtrain[:, 2])

    local_Xtrain_day = np.floor_divide(local_Xtrain_time, 1440) # day
    # local_Xtrain_month = np.floor_divide(local_Xtrain_day * 12, 365) # month
    local_Xtrain_month = np.floor_divide(local_Xtrain_day, 30) # month
    local_Xtrain[:, 310] = np.floor_divide(local_Xtrain_day, 365) # year
    local_Xtrain_minute = np.floor_divide(np.remainder(local_Xtrain_time, 1440), 5) # 5 minute
    local_Xtrain_weekday = np.remainder(local_Xtrain_day, 7) 
    local_Xtrain_month = np.remainder(local_Xtrain_month, 12)
    period_encode(local_Xtrain, minute_encoder, local_Xtrain_minute, 3)
    period_encode(local_Xtrain, weekday_encoder, local_Xtrain_weekday, 291)
    period_encode(local_Xtrain, month_encoder, local_Xtrain_month, 298)

    local_Xtest = np.zeros((num_test, 311))
    local_Xtest_time = df_test_local['time'].values
    local_Xtest[:, :3] =  df_test_local[['x', 'y', 'accuracy']].values
    local_Xtest[:, 2] = np.log10(local_Xtest[:, 2]) # accuracy

    local_Xtest_day = np.floor_divide(local_Xtest_time, 1440) # day
    # local_Xtest_month = np.floor_divide(local_Xtest_day * 12, 365) # month
    local_Xtest_month = np.floor_divide(local_Xtest_day, 30) # month
    local_Xtest[:, 310] = np.floor_divide(local_Xtest_day, 365) # year
    local_Xtest_minute = np.floor_divide(np.remainder(local_Xtest_time, 1440), 5) # 5 minute
    local_Xtest_weekday = np.remainder(local_Xtest_day, 7) 
    local_Xtest_month = np.remainder(local_Xtest_month, 12)
    period_encode(local_Xtest, minute_encoder, local_Xtest_minute, 3)
    period_encode(local_Xtest, weekday_encoder, local_Xtest_weekday, 291)
    period_encode(local_Xtest, month_encoder, local_Xtest_month, 298)


    local_row_id_train = df_train_local.index.values
    local_row_id_test = df_test_local.index.values

    # generate local place id dict and ytrain
    local_place_id_train = df_train_local['place_id'].values

    local_place_id_unique = list(np.unique(local_place_id_train))

    local_place_id_trans_dict = dict(zip(local_place_id_unique, range(len(local_place_id_unique))))
    local_place_id_invtrans_dict = dict(zip(range(len(local_place_id_unique)), local_place_id_unique))

    local_place_id_encoder = np.vectorize(lambda x:local_place_id_trans_dict[x])
    local_place_id_decoder = np.vectorize(lambda x:local_place_id_invtrans_dict[x])

    local_ytrain = local_place_id_encoder(local_place_id_train)

    valid_split = 655200

    valid_train_rows = np.argwhere(local_Xtrain_time <= valid_split).flatten()
    valid_test_rows = np.argwhere( (local_Xtrain_time > valid_split) & \
        (local_Xtrain[:, 0] >= x_min) & (local_Xtrain[:, 0] < x_max) & (local_Xtrain[:, 1] >= y_min) & (local_Xtrain[:, 1] < y_max) ).flatten()

    # give different weight
    fw = np.array([462.0, 975.0, 10.0, 1.0 / 6.0, 1.56, 1.06, 10.0])
    f_offset = np.array([0, 1, 2, 3, 291, 298, 310, 311])
    for i in range(7):
        for col in range(f_offset[i], f_offset[i+1]):
            local_Xtrain[:, col] *= fw[i]
            local_Xtest[:, col] *= fw[i]    

    local_Xvalid_train = local_Xtrain[valid_train_rows, :]
    local_Xvalid_test = local_Xtrain[valid_test_rows, :]

    local_yvalid_train = local_ytrain[valid_train_rows]
    local_yvalid_test = local_ytrain[valid_test_rows]


    # filter place id with low appearance
    thresh = 5
    local_yvalid_train_unique, local_yvalid_train_count = np.unique(local_yvalid_train, return_counts = True)
    local_yvalid_train_count_dict = dict(zip(local_yvalid_train_unique, local_yvalid_train_count))
    local_yvalid_train_count_func = np.vectorize(lambda x: local_yvalid_train_count_dict[x])
    local_yvalid_train_counts = local_yvalid_train_count_func(local_yvalid_train)
    filtered_valid_train_rows = np.argwhere(local_yvalid_train_counts >= thresh).flatten()
    filtered_local_yvalid_train = local_yvalid_train[filtered_valid_train_rows]
    filtered_local_Xvalid_train = local_Xvalid_train[filtered_valid_train_rows]

    local_ytrain_unique, local_ytrain_count = np.unique(local_ytrain, return_counts = True)
    local_ytrain_count_dict = dict(zip(local_ytrain_unique, local_ytrain_count))
    local_ytrain_count_func = np.vectorize(lambda x: local_ytrain_count_dict[x])
    local_ytrain_counts = local_ytrain_count_func(local_ytrain)
    filtered_train_rows = np.argwhere(local_ytrain_counts >= thresh).flatten()
    filtered_local_ytrain = local_ytrain[filtered_train_rows]
    filtered_local_Xtrain = local_Xtrain[filtered_train_rows]


    num_neigbhor_valid = np.floor_divide(np.sqrt(filtered_local_yvalid_train.shape[0]), 5.3).astype(int)
    # num_neigbhor_valid = 36

    print "valid train: ", local_yvalid_train.shape[0], "valid test: ", local_yvalid_test.shape[0], "neighbors:", num_neigbhor_valid

    # setup parameters for knn
    knn_param = dict()
    knn_param['n_neighbors'] = num_neigbhor_valid
    knn_param['weights'] = calculate_distance
    knn_param['n_jobs'] = -1
    knn_param['metric'] = 'manhattan'
    # knn_param['metric'] = mydist
    # knn_param['algorithm'] = 'brute'

    # # validation training
    # knn = KNeighborsClassifier(**knn_param)

    # # knn.fit(local_Xvalid_train, local_yvalid_train)
    # knn.fit(filtered_local_Xvalid_train, filtered_local_yvalid_train)

    # yprob_valid_knn = np.zeros((local_Xvalid_test.shape[0], len(local_place_id_unique) )) 
    
    # # start = time.time()
    # yprob_valid_knn[:, knn.classes_]= knn.predict_proba(local_Xvalid_test)
    # # end = time.time()

    # # print end - start

    # y_valid_sort = np.argsort(-yprob_valid_knn)

    # valid_score = map3(local_yvalid_test, y_valid_sort)

    # print "knn only: ", valid_score


    # training
    num_neigbhor = np.floor_divide(np.sqrt(filtered_local_ytrain.shape[0]), 5.3).astype(int)
    # num_neigbhor = 36

    knn_param['n_neighbors'] = num_neigbhor

    print "train: ", num_sample, " test: ", num_test, "neighbors:", num_neigbhor

    # training for test
    knn = KNeighborsClassifier(**knn_param)

    # knn.fit(local_Xtrain, local_ytrain)
    knn.fit(filtered_local_Xtrain, filtered_local_ytrain)


    yprob_knn = np.zeros((local_Xtest.shape[0], len(local_place_id_unique) )) 
    yprob_knn[:, knn.classes_]= knn.predict_proba(local_Xtest)

    # y_sort = np.argsort(-yprob_knn)


    # adjust knn probabilty using histogram counting of some feature
    outer_local_place_id_train_dict, outer_binning_table, outer_valid_binning_table = local_histogram

    # # validation training
    # local_Xtrain_accuracy = df_train_local['accuracy'].values

    # local_Xvalid_test_time = local_Xtrain_time[valid_test_rows]
    # local_Xvalid_test_accuracy = local_Xtrain_accuracy[valid_test_rows]

    # yprob_valid_hour = np.zeros((local_Xvalid_test.shape[0], len(local_place_id_unique) )) 
    # yprob_valid_four_hour = np.zeros((local_Xvalid_test.shape[0], len(local_place_id_unique) )) 
    # yprob_valid_weekday = np.zeros((local_Xvalid_test.shape[0], len(local_place_id_unique) )) 
    # yprob_valid_four_hour_shift = np.zeros((local_Xvalid_test.shape[0], len(local_place_id_unique) )) 
    # yprob_valid_weekday_shift = np.zeros((local_Xvalid_test.shape[0], len(local_place_id_unique) )) 
    # yprob_valid_accuracy = np.zeros((local_Xvalid_test.shape[0], len(local_place_id_unique) )) 

    # # generate histogram probability
    # fill_binning_prob(outer_valid_binning_table[0], local_Xvalid_test_time, get_hour, yprob_valid_hour, outer_local_place_id_train_dict, local_place_id_unique)
    # fill_binning_prob(outer_valid_binning_table[1], local_Xvalid_test_time, get_four_hour, yprob_valid_four_hour, outer_local_place_id_train_dict, local_place_id_unique)
    # fill_binning_prob(outer_valid_binning_table[2], local_Xvalid_test_time, get_weekday, yprob_valid_weekday, outer_local_place_id_train_dict, local_place_id_unique)
    # fill_binning_prob(outer_valid_binning_table[3], local_Xvalid_test_time, get_four_hour_shift, yprob_valid_four_hour_shift, outer_local_place_id_train_dict, local_place_id_unique)
    # fill_binning_prob(outer_valid_binning_table[4], local_Xvalid_test_time, get_weekday_shift, yprob_valid_weekday_shift, outer_local_place_id_train_dict, local_place_id_unique)
    # fill_binning_prob(outer_valid_binning_table[5], local_Xvalid_test_accuracy, get_accuracy, yprob_valid_accuracy, outer_local_place_id_train_dict, local_place_id_unique)

    # prob_th = 0.0001
    # yprob_valid_knn[yprob_valid_knn < prob_th] = prob_th

    # weights = [0.1, 0.1, 0.4]

    # yprob_valid = np.log10(yprob_valid_knn) * 1.0 \
    #             + np.log10(yprob_valid_four_hour) * weights[0] \
    #             + np.log10(yprob_valid_hour) * weights[1] \
    #             + np.log10(yprob_valid_weekday) * weights[2]

    # y_valid_sort = np.argsort(-yprob_valid)

    # valid_score = map3(local_yvalid_test, y_valid_sort)

    # print "adjusted:", valid_score

    # final_weights = np.array([0.1, 0.05, 0.2, 0.05, 0.2, 0.2])
    # final_weights = final_weights / np.sum(final_weights) * 0.7

    # yprob_valid = np.log10(yprob_valid_knn) * 1.0 \
    #             + np.log10(yprob_valid_four_hour) * final_weights[0] \
    #             + np.log10(yprob_valid_hour) * final_weights[1] \
    #             + np.log10(yprob_valid_weekday) * final_weights[2] \
    #             + np.log10(yprob_valid_four_hour_shift) * final_weights[3] \
    #             + np.log10(yprob_valid_weekday_shift) * final_weights[4] \
    #             + np.log10(yprob_valid_accuracy) * final_weights[5] 

    # yprob_valid = np.power(10, yprob_valid / 1.7)

    # print np.amax(yprob_valid), np.amin(yprob_valid)

    # y_valid_sort = np.argsort(-yprob_valid)

    # valid_score = map3(local_yvalid_test, y_valid_sort)

    # print "adjusted:", valid_score

    local_Xtest_accuracy = df_test_local['accuracy'].values

    yprob_hour = np.zeros((local_Xtest.shape[0], len(local_place_id_unique) )) 
    yprob_four_hour = np.zeros((local_Xtest.shape[0], len(local_place_id_unique) )) 
    yprob_weekday = np.zeros((local_Xtest.shape[0], len(local_place_id_unique) )) 
    yprob_four_hour_shift = np.zeros((local_Xtest.shape[0], len(local_place_id_unique) )) 
    yprob_weekday_shift = np.zeros((local_Xtest.shape[0], len(local_place_id_unique) )) 
    yprob_accuracy = np.zeros((local_Xtest.shape[0], len(local_place_id_unique) ))

    fill_binning_prob(outer_binning_table[0], local_Xtest_time, get_hour, yprob_hour, outer_local_place_id_train_dict, local_place_id_unique)
    fill_binning_prob(outer_binning_table[1], local_Xtest_time, get_four_hour, yprob_four_hour, outer_local_place_id_train_dict, local_place_id_unique)
    fill_binning_prob(outer_binning_table[2], local_Xtest_time, get_weekday, yprob_weekday, outer_local_place_id_train_dict, local_place_id_unique)
    fill_binning_prob(outer_binning_table[3], local_Xtest_time, get_four_hour_shift, yprob_four_hour_shift, outer_local_place_id_train_dict, local_place_id_unique)
    fill_binning_prob(outer_binning_table[4], local_Xtest_time, get_weekday_shift, yprob_weekday_shift, outer_local_place_id_train_dict, local_place_id_unique)
    fill_binning_prob(outer_binning_table[5], local_Xtest_accuracy, get_accuracy, yprob_accuracy, outer_local_place_id_train_dict, local_place_id_unique)

    prob_th = 0.001
    yprob_knn[yprob_knn < prob_th] = prob_th

    final_weights = np.array([0.1, 0.05, 0.2, 0.05, 0.2, 0.2])
    final_weights = final_weights / np.sum(final_weights) * 0.7

    yprob = np.log10(yprob_knn) * 1.0 \
                + np.log10(yprob_four_hour) * final_weights[0] \
                + np.log10(yprob_hour) * final_weights[1] \
                + np.log10(yprob_weekday) * final_weights[2] \
                + np.log10(yprob_four_hour_shift) * final_weights[3] \
                + np.log10(yprob_weekday_shift) * final_weights[4] \
                + np.log10(yprob_accuracy) * final_weights[5] 

    yprob = np.power(10, yprob / 1.7)

    y_sort = np.argsort(-yprob)

    # update_validation_log(cur_hash, valid_score)

    # update_validation_output(y_valid_sort, yprob_valid, local_row_id_train, valid_test_rows, local_place_id_invtrans_dict)

    update_test_output(y_sort, yprob, local_row_id_test, local_place_id_invtrans_dict)

    # update_submission(y_sort, local_row_id_test, local_place_id_invtrans_dict)

def fill_binning_prob(binning_table, local_X, trans_func, y_prob, hist_place_id_dict, local_place_id_unique):
    th = 0.001
    trans_local_X = trans_func(local_X)

    for i_col, local_place_id in enumerate(local_place_id_unique):
        ip = hist_place_id_dict[local_place_id]
        target_binning_func = np.vectorize(lambda x: binning_table[ip, x])

        y_prob[:, i_col] = target_binning_func(trans_local_X)

    y_prob[y_prob < th] = th


def run_local_histogram(df_train_local, hist_target_cols, hist_transform_func, hist_bins):
    valid_split = 655200

    local_place_id_train = df_train_local['place_id'].values

    local_place_id_train_unique = np.unique(local_place_id_train)
    num_local_place_id_train_unique = local_place_id_train_unique.shape[0]

    local_binning_tables = list()
    local_valid_binning_tables = list()
    for i, bins in enumerate(hist_bins):
        bin_size = len(bins)
        binning_table = np.zeros((num_local_place_id_train_unique, bin_size))
        valid_binning_table = np.zeros((num_local_place_id_train_unique, bin_size))

        local_binning_tables.append(binning_table)
        local_valid_binning_tables.append(valid_binning_table)

    for ip, place_id in enumerate(local_place_id_train_unique):
        target_df_train_local = df_train_local[ df_train_local['place_id'] == place_id ]
        target_df_valid_train_local = target_df_train_local[ target_df_train_local['time'] <= valid_split ]

        for i_trans, target_cols in enumerate(hist_target_cols):
            trans_func = hist_transform_func[i_trans]
            trans_bins = hist_bins[i_trans]

            binning_table = local_binning_tables[i_trans]
            valid_binning_table = local_valid_binning_tables[i_trans]

            target_Xtrain = target_df_train_local[target_cols].values
            trans_target_Xtrain = trans_func(target_Xtrain)

            target_Xvalid_train = target_df_valid_train_local[target_cols].values
            trans_target_Xvalid_train = trans_func(target_Xvalid_train)

            num_target_train = trans_target_Xtrain.shape[0]
            num_target_valid_train = trans_target_Xvalid_train.shape[0]

            trans_Xtrain_unique, trans_Xtrain_counts = np.unique(trans_target_Xtrain, return_counts = True)
            trans_Xtrain_counts = trans_Xtrain_counts.astype(float) / num_target_train

            trans_Xvalid_train_unique, trans_Xvalid_train_counts = np.unique(trans_target_Xvalid_train, return_counts = True)
            trans_Xvalid_train_counts = trans_Xvalid_train_counts.astype(float) / num_target_valid_train

            # here we assume bin starts from 0 
            for bin, bin_count in zip(trans_Xtrain_unique, trans_Xtrain_counts):
                binning_table[ip, bin] = bin_count

            for bin, bin_count in zip(trans_Xvalid_train_unique, trans_Xvalid_train_counts):
                valid_binning_table[ip, bin] = bin_count

    local_place_id_train_dict = dict(zip(local_place_id_train_unique, range(num_local_place_id_train_unique)))

    return local_place_id_train_dict, local_binning_tables, local_valid_binning_tables

def get_hour(X_time):
    return np.remainder(np.floor_divide(X_time, 60), 24)

def get_four_hour(X_time):
    return np.remainder(np.floor_divide(X_time, 60 * 4), 6)

def get_four_hour_shift(X_time):
    return np.remainder(np.floor_divide(X_time + 60 * 2, 60 * 4), 6)

def get_weekday(X_time):
    return np.remainder(np.floor_divide(X_time, 60 * 24), 7)

def get_weekday_shift(X_time):
    return np.remainder(np.floor_divide(X_time + 60 * 11, 60 * 24), 7)

def get_year(X_time):
    return np.floor_divide(X_time, 1440 * 365)

# def get_weekday_halfday(X_time):
#     return np.remainder(np.floor_divide(X_time + 60 * 6, 60 * 12), 14)  

def get_accuracy(X_accuracy):
    bins = np.array([0.0, 10.0, 40.0, 65.0, 125.0, 165.0])
    return np.digitize(X_accuracy, bins) - 1

def run_grid_knn():
    train_dtypes = dict() 
    train_dtypes['row_id'] = np.dtype(np.int32)
    train_dtypes['x'] = np.dtype(float)
    train_dtypes['y'] = np.dtype(float)
    train_dtypes['accuracy'] =  np.dtype(np.int16)
    train_dtypes['time'] = np.dtype(np.int32)
    train_dtypes['place_id'] = np.int64

    test_dtypes = dict(**train_dtypes)
    test_dtypes.pop('place_id')

    df_train = pd.read_csv(TRAIN_FILE, dtype = train_dtypes, index_col = 0)
    df_test = pd.read_csv(TEST_FILE, dtype = test_dtypes, index_col = 0)

    range_x = 10
    range_y = 20

    inner_range_x = 2
    inner_range_y = 2

    x_aug = np.around(0.6 / range_x, decimals = 4)
    y_aug = np.around(0.6 / range_y, decimals = 4)

    inner_x_aug = np.around(0.6 / (range_x * inner_range_x), decimals = 4)
    inner_y_aug = np.around(0.6 / (range_y * inner_range_y), decimals = 4)

    hist_target_cols = [['time'], ['time'], ['time'], ['time'], ['time'], ['accuracy']]
    hist_transform_func = [get_hour, get_four_hour, get_weekday, get_four_hour_shift, get_weekday_shift, get_accuracy]
    hist_bins = [range(24), range(6), range(7), range(6), range(7), range(6)]

    for ix, [x_min, x_max] in enumerate(fetch_grid_1d(range_x)):
        x_lower = x_min
        x_upper = x_max if x_max <= 10.0 else 10.0

        df_train_col = df_train[ (df_train['x'] >= x_min - x_aug) & (df_train['x'] <= x_max + x_aug) ]
        df_test_col = df_test[ (df_test['x'] >= x_min) & (df_test['x'] < x_max) ]

        for iy, [y_min, y_max] in enumerate(fetch_grid_1d(range_y)):
            y_lower = y_min
            y_upper = y_max if y_max <= 10.0 else 10.0

            df_train_local = df_train_col[ (df_train_col['y'] >= y_min - y_aug) & (df_train_col['y'] <= y_max + y_aug) ]
            df_test_local = df_test_col[ (df_test_col['y'] >= y_min) & (df_test_col['y'] < y_max) ]           

            print (ix, iy)

            local_histogram = run_local_histogram(df_train_local, hist_target_cols, hist_transform_func, hist_bins)

            for inner_ix, [inner_x_min, inner_x_max] in enumerate(fetch_grid_1d_inner(inner_range_x, x_lower, x_upper)):
                inner_df_train_col = df_train_local[ (df_train_local['x'] >= inner_x_min - inner_x_aug) & (df_train_local['x'] <= inner_x_max + inner_x_aug) ]
                inner_df_test_col = df_test_local[ (df_test_local['x'] >= inner_x_min) & (df_test_local['x'] < inner_x_max) ]

                for inner_iy, [inner_y_min, inner_y_max] in enumerate(fetch_grid_1d_inner(inner_range_y, y_lower, y_upper)):
                    inner_df_train_local = \
                        inner_df_train_col[ (inner_df_train_col['y'] >= inner_y_min - inner_y_aug) & (inner_df_train_col['y'] <= inner_y_max + inner_y_aug) ]
                    inner_df_test_local = inner_df_test_col[ (inner_df_test_col['y'] >= inner_y_min) & (inner_df_test_col['y'] < inner_y_max) ]

                    run_local_knn(inner_df_train_local, inner_df_test_local, [inner_x_min, inner_x_max, inner_y_min, inner_y_max], local_histogram)



if __name__ == "__main__":
    run_grid_knn()
