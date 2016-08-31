import numpy as np
import xgboost as xgb
import ml_metrics as metrics
import datetime
import multiprocessing

NUM_CORES = multiprocessing.cpu_count()

TRAIN_TEST_HASH_FILE = '../input/train_test_hash_63_159_ys.csv'

timestamp =  datetime.datetime.now().strftime('%Y%m%d%H%M%S')

VALID_LOG_FILE = '../valid/valid_log_63ys_{0:s}.csv'.format(timestamp)
VALID_OUTPUT_FILE = '../valid/valid_score_63ys_{0:s}.csv'.format(timestamp)
TEST_OUTPUT_FILE = '../train/test_score_63ys_{0:s}.csv'.format(timestamp)
SUBMISSION_FILE = '../output/submission_63ys_{0:s}.csv'.format(timestamp)

# writing header
with open(SUBMISSION_FILE, 'w') as f:
    f.write("row_id,place_id\n")

with open(VALID_LOG_FILE, 'w') as f:
    f.write("hash,score\n")

with open(VALID_OUTPUT_FILE, 'w') as f:
    f.write("row_id,label,prob\n")

with open(TEST_OUTPUT_FILE, 'w') as f:
    f.write("row_id,label,prob\n")


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

def run_local_xgb(local_row_id_train, local_Xtrain, local_place_id_train, local_row_id_test, local_Xtest, cur_hash):
    local_place_id_unique = list(np.unique(local_place_id_train))

    local_place_id_trans_dict = dict(zip(local_place_id_unique, range(len(local_place_id_unique))))
    local_place_id_invtrans_dict = dict(zip(range(len(local_place_id_unique)), local_place_id_unique))

    local_place_id_encoder = np.vectorize(lambda x:local_place_id_trans_dict[x])
    local_place_id_decoder = np.vectorize(lambda x:local_place_id_invtrans_dict[x])

    local_ytrain = local_place_id_encoder(local_place_id_train)

    valid_split = 655200

    valid_train_rows = np.squeeze(np.argwhere(local_Xtrain[:, 3] <= valid_split))
    valid_test_rows = np.squeeze(np.argwhere(local_Xtrain[:, 3] > valid_split))

    local_Xvalid_train = local_Xtrain[valid_train_rows, :]
    local_Xvalid_test = local_Xtrain[valid_test_rows, :]

    local_yvalid_train = local_ytrain[valid_train_rows]
    local_yvalid_test = local_ytrain[valid_test_rows]

    # create validation set 
    xg_valid_train = xgb.DMatrix(local_Xvalid_train, label = local_yvalid_train)
    xg_valid_test= xgb.DMatrix(local_Xvalid_test, label = local_yvalid_test)

    xg_train = xgb.DMatrix(local_Xtrain, label = local_ytrain)
    xg_test= xgb.DMatrix(local_Xtest)

    # setup parameters for xgboost
    param = dict()
    # use softmax multi-class classification
    param['objective'] = 'multi:softprob'
    # scale weight of positive examples
    param['eta'] = 0.03
    param['subsample'] = 0.9
    param['colsample_bytree'] = 0.9
    # param['colsample_bylevel'] = 0.9
    param['max_depth'] = 3
    param['silent'] = 1
    param['nthread'] = NUM_CORES
    param['num_class'] = len(local_place_id_unique)

    num_round = 100


    # validation training
    watchlist = [ (xg_valid_train,'train'), (xg_valid_test, 'test') ]
    # bst_valid = xgb.train(param, xg_valid_train, num_round, watchlist, early_stopping_rounds = 30, verbose_eval = 0)
    bst_valid = xgb.train(param, xg_valid_train, num_round, watchlist, feval = map3eval, maximize = True, early_stopping_rounds = 30, verbose_eval = 0)

    final_num_round = min(num_round, max(25, bst_valid.best_iteration+16))

    yprob_valid = bst_valid.predict( xg_valid_test, ntree_limit=final_num_round).reshape( local_yvalid_test.shape[0], param['num_class'] )

    y_valid_sort = np.argsort(-yprob_valid)

    valid_score = map3(local_yvalid_test, y_valid_sort)

    # local_ylabel_valid = np.expand_dims(local_yvalid_test, axis = 1)

    # valid_score = metrics.mapk(local_ylabel_valid, y_valid_sort, 3)

    # training for test
    # final_num_round = min(num_round, bst_valid.best_iteration + 20)

    bst = xgb.train(param, xg_train, final_num_round)
    yprob = bst.predict(xg_test).reshape(local_Xtest.shape[0], param['num_class'] )

    y_sort = np.argsort(-yprob)

    update_validation_log(cur_hash, valid_score)

    update_validation_output(y_valid_sort, yprob_valid, local_row_id_train, valid_test_rows, local_place_id_invtrans_dict)

    update_test_output(y_sort, yprob, local_row_id_test, local_place_id_invtrans_dict)

    update_submission(y_sort, local_row_id_test, local_place_id_invtrans_dict)

def run_grid_xgb():
    f_train_test_hash = open(TRAIN_TEST_HASH_FILE)
    f_train_test_hash.readline()

    read_col_names = ['hash', 'row_id', 'x', 'y', 'accuracy', 'time', 'place_id']
    read_col_dict = dict(zip(read_col_names, range(len(read_col_names))))
    print read_col_dict
    hash_col = read_col_dict['hash']
    row_id_col = read_col_dict['row_id']
    accuracy_col = read_col_dict['accuracy']
    time_col = read_col_dict['time']
    x_pos_col = read_col_dict['x']
    y_pos_col = read_col_dict['y']
    place_id_col = read_col_dict['place_id']

    write_Xcol_name = ['row_id', 'x', 'y', 'accuracy', 'time', 'timedayround', 'timeweekround']

    # target_hash = 231489536
    # counter = 0

    while 1:
        line =f_train_test_hash.readline()
        if not line:
            print "end of file"
            break
        if line[0] == '$':
            print '%d finished' %  cur_hash
            # counter += 1
            # if (counter == 2):
            #     break
            continue    
        if line[0] == '^':
            data = [word.strip() for word in line.split(',')]
            cur_hash = int(data[0][1:])
            num_sample = int(data[1])
            print "%d, train: %d" % (cur_hash, num_sample)
            
            # loading training data
            local_Xtrain = np.zeros((num_sample, 6))
            local_row_id_train = np.zeros(num_sample, int)
            local_place_id_train = np.zeros(num_sample, int)
            for irow in xrange(num_sample):
                line = f_train_test_hash.readline()
                data = [word.strip() for word in line.split(',')]

                local_Xtrain[irow, 0] = float(data[x_pos_col])
                local_Xtrain[irow, 1] = float(data[y_pos_col])
                local_Xtrain[irow, 2] = float(data[accuracy_col])
                local_Xtrain[irow, 3] = float(data[time_col])
                
                local_row_id_train[irow] = int(data[row_id_col])
                local_place_id_train[irow] = int(data[place_id_col])
            
            local_Xtrain[:, 4] = np.remainder(local_Xtrain[:, 3], 24 * 60)
            local_Xtrain[:, 5] = np.remainder(local_Xtrain[:, 3], 24 * 60 * 7)
            
            line = f_train_test_hash.readline()
            if line[0] == '?':
                print "training set generated"
            else:
                print "size error?"
                print line            
                break
            line = f_train_test_hash.readline()
            if line[0] != '!':
                print "test set missing?"
                print line
                break
            
            data = [word.strip() for word in line.split(',')]
            cur_hash = int(data[0][1:])
            num_test = int(data[1])
            
            print "%d, test: %d" % (cur_hash, num_test)
            
            # loading test data
            local_Xtest = np.zeros((num_test, 6))
            local_row_id_test = np.zeros(num_test, int)
            for irow in xrange(num_test):
                line = f_train_test_hash.readline()
                data = [word.strip() for word in line.split(',')]

                local_Xtest[irow, 0] = float(data[x_pos_col])
                local_Xtest[irow, 1] = float(data[y_pos_col])
                local_Xtest[irow, 2] = float(data[accuracy_col])
                local_Xtest[irow, 3] = float(data[time_col])
                
                local_row_id_test[irow] = int(data[row_id_col])
            
            local_Xtest[:, 4] = np.remainder(local_Xtest[:, 3], 24 * 60)
            local_Xtest[:, 5] = np.remainder(local_Xtest[:, 3], 24 * 60 * 7)
        
            run_local_xgb(local_row_id_train, local_Xtrain, local_place_id_train, local_row_id_test, local_Xtest, cur_hash)

if __name__ == "__main__":
    run_grid_xgb()
