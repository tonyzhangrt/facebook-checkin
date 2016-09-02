import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import time
import multiprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

NUM_CORES = multiprocessing.cpu_count()/2

TRAIN_FILE = '../input/train.csv'
TEST_FILE = '../input/test.csv'

TRAIN_TEST_HASH_FILE = '../input/train_test_hash.csv'

timestamp =  datetime.datetime.now().strftime('%Y%m%d%H%M%S')

VALID_LOG_FILE = '../valid/valid_log_xgb_aug_nn_63ys_{0:s}.csv'.format(timestamp)
VALID_OUTPUT_FILE = '../valid/valid_score_xgb_aug_nn_63ys_{0:s}.csv'.format(timestamp)
TEST_OUTPUT_FILE = '../train/test_score_xgb_aug_nn_63ys_{0:s}.csv'.format(timestamp)
SUBMISSION_FILE = '../output/submission_xgb_aug_nn_63ys_{0:s}.csv'.format(timestamp)

# writing header
# with open(SUBMISSION_FILE, 'w') as f:
#     f.write("row_id,place_id\n")

with open(VALID_OUTPUT_FILE, 'w') as f:
    f.write("row_id,label,prob\n")

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
    dist = abs(x[0] - y[0]) * 462.0
    dist += abs(x[1] - y[1]) * 975.0
    dist += abs(x[2] - y[2]) * 10.0
    # dist += min(abs(x[3] - y[3]) % 24, 24 - abs(y[3] - x[3]) % 24) * 4.0
    dist += min(abs(x[3] - y[3]) % 288, 288 - abs(y[3] - x[3]) % 288) / 3.0
    dist += min(abs(x[4] - y[4]) % 7, 7 - abs(y[4] - x[4]) % 7) * 3.12
    dist += min(abs(x[5] - y[5]) % 12, 12 - abs(y[5] - x[5]) % 12) * 2.12
    dist += abs(x[6] - y[6]) * 10.0
    
    return dist

def fetch_grid_1d(range_x):
    for ix in range(range_x):
        x_min = np.around(float(ix) / range_x * 10, decimals = 4)
        x_max = np.around((float(ix) + 1) / range_x * 10, decimals = 4)

        if ix == range_x - 1:
            x_max += 0.0001

        yield [x_min, x_max]

def fetch_grid_1d_shift(range_x):
    for ix in range(range_x - 1):
        x_min = np.around((float(ix) + 0.5) / range_x * 10, decimals = 4)
        x_max = np.around((float(ix) + 1.5) / range_x * 10, decimals = 4)
        
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

def fill_dist(X_dist, neigh_dist, neigh_ind, ytrain, upper = 200.0):
    num_rows = X_dist.shape[0]
    for i in range(num_rows):
        for ind, dist  in zip(neigh_ind[i], neigh_dist[i]):
            if dist == 0:
                continue
            yid = ytrain[ind]
            X_dist[i, yid] += 1
            # invert_dist = upper / (dist + 1)
            # if X_dist[i, yid] < invert_dist:
            #     X_dist[i, yid] = invert_dist


def run_local_xgb(df_train_local, df_test_local, anchors):
    [x_min, x_max, y_min, y_max] = anchors

    num_sample = df_train_local.shape[0]
    num_test = df_test_local.shape[0]

    # generate validation set
    local_Xtrain_time = df_train_local['time'].values
    local_Xtrain_x = df_train_local['x'].values
    local_Xtrain_y = df_train_local['y'].values

    local_row_id_train = df_train_local.index.values
    local_row_id_test = df_test_local.index.values

    # generate local place id dict and ytrain
    local_place_id_train = df_train_local['place_id'].values

    local_place_id_unique = np.unique(local_place_id_train)
    num_place_id_unique = local_place_id_unique.shape[0]

    local_place_id_trans_dict = dict(zip(local_place_id_unique, range(num_place_id_unique)))
    local_place_id_invtrans_dict = dict(zip(range(num_place_id_unique), local_place_id_unique))

    local_place_id_encoder = np.vectorize(lambda x:local_place_id_trans_dict[x])
    local_place_id_decoder = np.vectorize(lambda x:local_place_id_invtrans_dict[x])

    local_ytrain = local_place_id_encoder(local_place_id_train)

    valid_split = 655200

    valid_train_rows = np.argwhere(local_Xtrain_time <= valid_split).flatten()
    valid_test_rows = np.argwhere( (local_Xtrain_time > valid_split) & \
        (local_Xtrain_x >= x_min) & (local_Xtrain_x < x_max) & (local_Xtrain_y >= y_min) & (local_Xtrain_y < y_max) ).flatten()

    num_valid_test = valid_test_rows.shape[0]

    local_yvalid_train = local_ytrain[valid_train_rows]
    local_yvalid_test = local_ytrain[valid_test_rows]


    # filter place id with low appearance
    thresh = 3
    local_yvalid_train_unique, local_yvalid_train_count = np.unique(local_yvalid_train, return_counts = True)
    local_yvalid_train_count_dict = dict(zip(local_yvalid_train_unique, local_yvalid_train_count))
    local_yvalid_train_count_func = np.vectorize(lambda x: local_yvalid_train_count_dict[x])
    local_yvalid_train_counts = local_yvalid_train_count_func(local_yvalid_train)
    filtered_valid_train_rows = np.argwhere(local_yvalid_train_counts >= thresh).flatten()
    num_filtered_valid_train = filtered_valid_train_rows.shape[0]


    local_ytrain_unique, local_ytrain_count = np.unique(local_ytrain, return_counts = True)
    local_ytrain_count_dict = dict(zip(local_ytrain_unique, local_ytrain_count))
    local_ytrain_count_func = np.vectorize(lambda x: local_ytrain_count_dict[x])
    local_ytrain_counts = local_ytrain_count_func(local_ytrain)
    filtered_train_rows = np.argwhere(local_ytrain_counts >= thresh).flatten()
    num_filtered_train = filtered_train_rows.shape[0]

    filtered_local_yvalid_train = local_yvalid_train[filtered_valid_train_rows]
    filtered_local_ytrain = local_ytrain[filtered_train_rows]

    # place_id appear in filtered valid train must be in filtered train
    filtered_ytrain_unique = np.unique(filtered_local_ytrain)
    num_filtered_ytrain_unique = filtered_ytrain_unique.shape[0]

    filtered_ytrain_trans_dict = dict(zip(filtered_ytrain_unique, range(num_filtered_ytrain_unique)))
    filtered_ytrain_invtrans_dict = dict(zip(range(num_filtered_ytrain_unique), local_place_id_unique))

    filtered_ytrain_encoder = np.vectorize(lambda x:filtered_ytrain_trans_dict[x])
    filtered_ytrain_decoder = np.vectorize(lambda x:filtered_ytrain_invtrans_dict[x])

    trans_filtered_local_yvalid_train = filtered_ytrain_encoder(filtered_local_yvalid_train)
    trans_filtered_local_ytrain = filtered_ytrain_encoder(filtered_local_ytrain)

   # generate nearest neighbor features using tmp matrix
    tmp_local_Xtrain = np.zeros((num_sample, 311))
    tmp_local_Xtrain[:, :3] =  df_train_local[['x', 'y', 'accuracy']].values
    tmp_local_Xtrain[:, 2] = np.log10(tmp_local_Xtrain[:, 2])

    local_Xtrain_time = df_train_local['time'].values
    local_Xtrain_day = np.floor_divide(local_Xtrain_time, 1440) # day
    # local_Xtrain_month = np.floor_divide(local_Xtrain_day * 12, 365) # month
    local_Xtrain_month = np.floor_divide(local_Xtrain_day, 30) # month

    tmp_local_Xtrain[:, 310] = np.floor_divide(local_Xtrain_day, 365) # year
    local_Xtrain_minute = np.floor_divide(np.remainder(local_Xtrain_time, 1440), 5) # 5 minute
    local_Xtrain_weekday = np.remainder(local_Xtrain_day, 7) 
    local_Xtrain_month = np.remainder(local_Xtrain_month, 12)
    period_encode(tmp_local_Xtrain, minute_encoder, local_Xtrain_minute, 3)
    period_encode(tmp_local_Xtrain, weekday_encoder, local_Xtrain_weekday, 291)
    period_encode(tmp_local_Xtrain, month_encoder, local_Xtrain_month, 298)


    tmp_local_Xtest = np.zeros((num_test, 311))
    tmp_local_Xtest[:, :3] =  df_test_local[['x', 'y', 'accuracy']].values
    tmp_local_Xtest[:, 2] = np.log10(tmp_local_Xtest[:, 2]) # accuracy

    local_Xtest_time = df_test_local['time'].values
    local_Xtest_day = np.floor_divide(local_Xtest_time, 1440) # day
    # local_Xtest_month = np.floor_divide(local_Xtest_day * 12, 365) # month
    local_Xtest_month = np.floor_divide(local_Xtest_day, 30) # month
    tmp_local_Xtest[:, 310] = np.floor_divide(local_Xtest_day, 365) # year
    local_Xtest_minute = np.floor_divide(np.remainder(local_Xtest_time, 1440), 5) # 5 minute
    local_Xtest_weekday = np.remainder(local_Xtest_day, 7) 
    local_Xtest_month = np.remainder(local_Xtest_month, 12)
    period_encode(tmp_local_Xtest, minute_encoder, local_Xtest_minute, 3)
    period_encode(tmp_local_Xtest, weekday_encoder, local_Xtest_weekday, 291)
    period_encode(tmp_local_Xtest, month_encoder, local_Xtest_month, 298)

    # give different weight
    fw = np.array([462.0, 975.0, 10.0, 1.0 / 6.0, 1.56, 1.06, 10.0])
    f_offset = np.array([0, 1, 2, 3, 291, 298, 310, 311])
    for i in range(7):
        for col in range(f_offset[i], f_offset[i+1]):
            tmp_local_Xtrain[:, col] *= fw[i]
            tmp_local_Xtest[:, col] *= fw[i]


    tmp_local_Xvalid_train = tmp_local_Xtrain[valid_train_rows, :]
    tmp_local_Xvalid_test = tmp_local_Xtrain[valid_test_rows, :]

    tmp_filtered_local_Xvalid_train = tmp_local_Xvalid_train[filtered_valid_train_rows]
    tmp_filtered_local_Xtrain = tmp_local_Xtrain[filtered_train_rows]

    # generate nearest neighbor features

    num_neigbhor = 33

    nn_param = dict()
    nn_param['n_neighbors'] = num_neigbhor
    nn_param['radius'] = 200
    nn_param['n_jobs'] = -1
    nn_param['metric'] = 'manhattan'
    # nn_param['metric'] = mydist

    nn = NearestNeighbors(**nn_param)
    nn.fit(tmp_filtered_local_Xvalid_train)

    filtered_valid_train_neigh_dist, filtered_valid_train_neigh_ind = nn.kneighbors(X = tmp_filtered_local_Xvalid_train, 
        n_neighbors = num_neigbhor+1, return_distance=True)
    valid_test_neigh_dist, valid_test_neigh_ind = nn.kneighbors(X = tmp_local_Xvalid_test, return_distance=True)
    
    filtered_Xvalid_train_dist = np.zeros((num_filtered_valid_train, num_filtered_ytrain_unique)) 
    Xvalid_test_dist = np.zeros((num_valid_test, num_filtered_ytrain_unique))

    fill_dist(filtered_Xvalid_train_dist, filtered_valid_train_neigh_dist, filtered_valid_train_neigh_ind, trans_filtered_local_yvalid_train)
    fill_dist(Xvalid_test_dist, valid_test_neigh_dist, valid_test_neigh_ind, trans_filtered_local_yvalid_train)

    nn = NearestNeighbors(**nn_param)
    nn.fit(tmp_filtered_local_Xtrain)

    filtered_train_neigh_dist, filtered_train_neigh_ind = nn.kneighbors(X = tmp_filtered_local_Xtrain, 
        n_neighbors = num_neigbhor+1, return_distance=True)
    test_neigh_dist, test_neigh_ind = nn.kneighbors(X = tmp_local_Xtest, return_distance=True)

    # # print filtered_train_neigh_dist[0]
    # # print filtered_train_neigh_ind[0]
    # # print trans_filtered_local_ytrain[filtered_train_neigh_ind[0]]
    # # print trans_filtered_local_ytrain[filtered_train_neigh_ind[0]+1]

    # print test_neigh_dist[0]
    # print test_neigh_ind[0] 
    # print trans_filtered_local_ytrain[test_neigh_ind[0]]

    # print test_neigh_dist[1]
    # print test_neigh_ind[1] 
    # print trans_filtered_local_ytrain[test_neigh_ind[1]]

    filtered_Xtrain_dist = np.zeros((num_filtered_train, num_filtered_ytrain_unique))
    Xtest_dist = np.zeros((num_test, num_filtered_ytrain_unique))

    fill_dist(filtered_Xtrain_dist, filtered_train_neigh_dist, filtered_train_neigh_ind, trans_filtered_local_ytrain)
    fill_dist(Xtest_dist, test_neigh_dist, test_neigh_ind, trans_filtered_local_ytrain)

    # combine distance feature with the original feature
    local_Xtrain = np.zeros((num_sample, 7))
    local_Xtrain[:, :3] = df_train_local[['x', 'y', 'accuracy']].values
    local_Xtrain[:, 2] = np.log10(local_Xtrain[:, 2])
    local_Xtrain[:, 6] = np.floor_divide(local_Xtrain_day, 365) # year
    local_Xtrain[:, 3] = np.remainder(local_Xtrain_time, 1440) # minute
    local_Xtrain[:, 4] = np.remainder(local_Xtrain_day, 7) 
    local_Xtrain[:, 5] = np.remainder(local_Xtrain_month, 12)    

    filtered_local_Xvalid_train = np.zeros((num_filtered_valid_train, 7 + num_filtered_ytrain_unique))
    filtered_local_Xvalid_train[:, :7] = local_Xtrain[valid_train_rows, :][filtered_valid_train_rows, :]
    filtered_local_Xvalid_train[:, 7:] = filtered_Xvalid_train_dist

    local_Xvalid_test = np.zeros((num_valid_test, 7 + num_filtered_ytrain_unique))
    local_Xvalid_test[:, :7] = local_Xtrain[valid_test_rows, :]
    local_Xvalid_test[:, 7:] = Xvalid_test_dist

    filtered_local_Xtrain = np.zeros((num_filtered_train, 7 + num_filtered_ytrain_unique))
    filtered_local_Xtrain[:, :7] = local_Xtrain[filtered_train_rows, :]
    filtered_local_Xtrain[:, 7:] = filtered_Xtrain_dist


    local_Xtest = np.zeros((num_test, 7 + num_filtered_ytrain_unique))
    local_Xtest[:, :3] =  df_test_local[['x', 'y', 'accuracy']].values
    local_Xtest[:, 2] = np.log10(local_Xtest[:, 2])
    local_Xtest[:, 6] = np.floor_divide(local_Xtest_day, 365) # year
    local_Xtest[:, 3] = np.remainder(local_Xtest_time, 1440) # minute
    local_Xtest[:, 4] = np.remainder(local_Xtest_day, 7) 
    local_Xtest[:, 5] = np.remainder(local_Xtest_month, 12)
    local_Xtest[:, 7:] = Xtest_dist


    # create validation set 
    # xg_valid_train = xgb.DMatrix(local_Xvalid_train, label = local_yvalid_train)
    # xg_valid_test= xgb.DMatrix(local_Xvalid_test, label = local_yvalid_test)

    # xg_train = xgb.DMatrix(local_Xtrain, label = local_ytrain)
    # xg_test= xgb.DMatrix(local_Xtest)

    print "train: ", num_filtered_train, "test:", num_test
    print "valid train: ", num_filtered_valid_train, "valid test:", num_valid_test

    print filtered_local_Xvalid_train.shape, local_Xvalid_test.shape
    print filtered_local_Xtrain.shape, local_Xtest.shape

    xg_valid_train = xgb.DMatrix(filtered_local_Xvalid_train, label = filtered_local_yvalid_train)
    xg_valid_test= xgb.DMatrix(local_Xvalid_test, label = local_yvalid_test)

    xg_train = xgb.DMatrix(filtered_local_Xtrain, label = filtered_local_ytrain)
    xg_test= xgb.DMatrix(local_Xtest)

    # setup parameters for xgboost
    param = dict()
    # use softmax multi-class classification
    param['objective'] = 'multi:softprob'
    # scale weight of positive examples
    param['eta'] = 0.02
    param['subsample'] = 0.9
    param['colsample_bytree'] = 0.8
    # param['colsample_bylevel'] = 0.9
    param['max_depth'] = 4
    param['silent'] = 1
    param['nthread'] = NUM_CORES
    param['num_class'] = num_place_id_unique

    num_round = 110
    min_round = 25


    # validation training
    watchlist = [ (xg_valid_train,'train'), (xg_valid_test, 'test') ]
    # bst_valid = xgb.train(param, xg_valid_train, num_round, watchlist, early_stopping_rounds = 30, verbose_eval = 0)
    bst_valid = xgb.train(param, xg_valid_train, num_round, watchlist, feval = map3eval, maximize = True, early_stopping_rounds = 30, verbose_eval = 0)

    final_num_round = min(num_round, max(min_round, bst_valid.best_iteration+16))

    yprob_valid = bst_valid.predict( xg_valid_test, ntree_limit=final_num_round).reshape( local_yvalid_test.shape[0], param['num_class'] )

    y_valid_sort = np.argsort(-yprob_valid)

    # valid_score = map3(local_yvalid_test, y_valid_sort)

    # print "best round: ", bst_valid.best_iteration, "validation score: ", valid_score

    # training for test
    bst = xgb.train(param, xg_train, final_num_round)
    yprob = bst.predict(xg_test).reshape(local_Xtest.shape[0], param['num_class'] )

    y_sort = np.argsort(-yprob)


    # update_validation_log(cur_hash, valid_score)

    update_validation_output(y_valid_sort, yprob_valid, local_row_id_train, valid_test_rows, local_place_id_invtrans_dict)

    update_test_output(y_sort, yprob, local_row_id_test, local_place_id_invtrans_dict)

    # update_submission(y_sort, local_row_id_test, local_place_id_invtrans_dict)

def run_grid_xgb():
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

    range_x = 63
    range_y = 159

    x_aug = np.around(0.5 / range_x, decimals = 4)
    y_aug = np.around(0.5 / range_y, decimals = 4)

    for ix, [x_min, x_max] in enumerate(fetch_grid_1d(range_x)):
        df_train_col = df_train[ (df_train['x'] >= x_min - x_aug) & (df_train['x'] <= x_max + x_aug) ]
        df_test_col = df_test[ (df_test['x'] >= x_min) & (df_test['x'] < x_max) ]

        for iy, [y_min, y_max] in enumerate(fetch_grid_1d_shift(range_y)):
            df_train_local = df_train_col[ (df_train_col['y'] >= y_min - y_aug) & (df_train_col['y'] <= y_max + y_aug) ]
            df_test_local = df_test_col[ (df_test_col['y'] >= y_min) & (df_test_col['y'] < y_max) ]           

            print (ix, iy)

            run_local_xgb(df_train_local, df_test_local, [x_min, x_max, y_min, y_max])



if __name__ == "__main__":
    run_grid_xgb()
