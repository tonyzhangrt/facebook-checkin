#include "my_utils.h"
#include "grid_utils.h"
#include "grid_split.h"
#include "valid_test_sub.h"
#include "basic_count.h"
#include <iostream>

using namespace std;

void reorder_validation(){
    int max_rank = 8, max_sub = 3;
    size_t num_valid_train = 23318975, num_valid_test = 5799046;
    double mapk_score = 0, weight = 1;
    unsigned int limit = 50;

    /* reorder row_id */
    ifstream f_valid_score_63("../valid/valid_score_63.csv");
    ifstream f_valid_score_63xs("../valid/valid_score_63xs.csv");
    ifstream f_valid_score_63ys("../valid/valid_score_63ys.csv");
    ifstream f_valid_score_63xys("../valid/valid_score_63xys.csv");

    ofstream f_valid_score_63_reorder("../valid/reorder_valid_score_63.csv");
    ofstream f_valid_score_63xs_reorder("../valid/reorder_valid_score_63xs.csv");
    ofstream f_valid_score_63ys_reorder("../valid/reorder_valid_score_63ys.csv");
    ofstream f_valid_score_63xys_reorder("../valid/reorder_valid_score_63xys.csv");

    reorder_score(f_valid_score_63, f_valid_score_63_reorder);
    reorder_score(f_valid_score_63xs, f_valid_score_63xs_reorder);
    reorder_score(f_valid_score_63ys, f_valid_score_63ys_reorder);
    reorder_score(f_valid_score_63xys, f_valid_score_63xys_reorder);

    f_valid_score_63.close();
    f_valid_score_63xs.close();
    f_valid_score_63ys.close();
    f_valid_score_63xys.close();
    f_valid_score_63_reorder.close();
    f_valid_score_63xs_reorder.close();
    f_valid_score_63ys_reorder.close();
    f_valid_score_63xys_reorder.close();

    ifstream f_valid_score_xgb_new2_63("../valid/valid_score_xgb_new2_63.csv");
    ifstream f_valid_score_xgb_new2_63xs("../valid/valid_score_xgb_new2_63xs.csv");
    ifstream f_valid_score_xgb_new2_63ys("../valid/valid_score_xgb_new2_63ys.csv");
    ifstream f_valid_score_xgb_new2_63xys("../valid/valid_score_xgb_new2_63xys.csv");

    ofstream f_valid_score_xgb_new2_63_reorder("../valid/reorder_valid_score_xgb_new2_63.csv");
    ofstream f_valid_score_xgb_new2_63xs_reorder("../valid/reorder_valid_score_xgb_new2_63xs.csv");
    ofstream f_valid_score_xgb_new2_63ys_reorder("../valid/reorder_valid_score_xgb_new2_63ys.csv");
    ofstream f_valid_score_xgb_new2_63xys_reorder("../valid/reorder_valid_score_xgb_new2_63xys.csv");

    reorder_score(f_valid_score_xgb_new2_63, f_valid_score_xgb_new2_63_reorder);
    reorder_score(f_valid_score_xgb_new2_63xs, f_valid_score_xgb_new2_63xs_reorder);
    reorder_score(f_valid_score_xgb_new2_63ys, f_valid_score_xgb_new2_63ys_reorder);
    reorder_score(f_valid_score_xgb_new2_63xys, f_valid_score_xgb_new2_63xys_reorder);

    f_valid_score_xgb_new2_63.close();
    f_valid_score_xgb_new2_63xs.close();
    f_valid_score_xgb_new2_63ys.close();
    f_valid_score_xgb_new2_63xys.close();
    f_valid_score_xgb_new2_63_reorder.close();
    f_valid_score_xgb_new2_63xs_reorder.close();
    f_valid_score_xgb_new2_63ys_reorder.close();
    f_valid_score_xgb_new2_63xys_reorder.close();

    ifstream f_valid_score_xgb_aug_nn_63("../valid/valid_score_xgb_aug_nn_63.csv");
    ifstream f_valid_score_xgb_aug_nn_63xs("../valid/valid_score_xgb_aug_nn_63xs.csv");
    ifstream f_valid_score_xgb_aug_nn_63ys("../valid/valid_score_xgb_aug_nn_63ys.csv");
    ifstream f_valid_score_xgb_aug_nn_63xys("../valid/valid_score_xgb_aug_nn_63xys.csv");

    ofstream f_valid_score_xgb_aug_nn_63_reorder("../valid/reorder_valid_score_xgb_aug_nn_63.csv");
    ofstream f_valid_score_xgb_aug_nn_63xs_reorder("../valid/reorder_valid_score_xgb_aug_nn_63xs.csv");
    ofstream f_valid_score_xgb_aug_nn_63ys_reorder("../valid/reorder_valid_score_xgb_aug_nn_63ys.csv");
    ofstream f_valid_score_xgb_aug_nn_63xys_reorder("../valid/reorder_valid_score_xgb_aug_nn_63xys.csv");    

    reorder_score(f_valid_score_xgb_aug_nn_63, f_valid_score_xgb_aug_nn_63_reorder);
    reorder_score(f_valid_score_xgb_aug_nn_63xs, f_valid_score_xgb_aug_nn_63xs_reorder);
    reorder_score(f_valid_score_xgb_aug_nn_63ys, f_valid_score_xgb_aug_nn_63ys_reorder);
    reorder_score(f_valid_score_xgb_aug_nn_63xys, f_valid_score_xgb_aug_nn_63xys_reorder);

    f_valid_score_xgb_aug_nn_63.close();
    f_valid_score_xgb_aug_nn_63xs.close();
    f_valid_score_xgb_aug_nn_63ys.close();
    f_valid_score_xgb_aug_nn_63xys.close();
    f_valid_score_xgb_aug_nn_63_reorder.close();
    f_valid_score_xgb_aug_nn_63xs_reorder.close();
    f_valid_score_xgb_aug_nn_63ys_reorder.close();
    f_valid_score_xgb_aug_nn_63xys_reorder.close();

    ifstream f_valid_score_knn_count_20("../valid/valid_score_knn_count_20.csv");
    ofstream f_valid_score_knn_count_20_reorder("../valid/reorder_valid_score_knn_count_20.csv");

    reorder_score(f_valid_score_knn_count_20, f_valid_score_knn_count_20_reorder);
    f_valid_score_knn_count_20.close();
    f_valid_score_knn_count_20_reorder.close();
    /* end reorder row_id */
}

void check_validation(){
    int max_rank = 8, max_sub = 3;
    size_t num_valid_train = 23318975, num_valid_test = 5799046;
    double mapk_score = 0, weight = 1;
    unsigned int limit = 50;

    ifstream f_valid_test("../valid/valid_test.csv");

    ifstream reorder_f_valid_score_63("../valid/reorder_valid_score_63.csv");
    ifstream reorder_f_valid_score_63xs("../valid/reorder_valid_score_63xs.csv");
    ifstream reorder_f_valid_score_63ys("../valid/reorder_valid_score_63ys.csv");
    ifstream reorder_f_valid_score_63xys("../valid/reorder_valid_score_63xys.csv");

    ifstream reorder_f_valid_score_xgb_new2_63("../valid/reorder_valid_score_xgb_new2_63.csv");
    ifstream reorder_f_valid_score_xgb_new2_63xs("../valid/reorder_valid_score_xgb_new2_63xs.csv");
    ifstream reorder_f_valid_score_xgb_new2_63ys("../valid/reorder_valid_score_xgb_new2_63ys.csv");
    ifstream reorder_f_valid_score_xgb_new2_63xys("../valid/reorder_valid_score_xgb_new2_63xys.csv");    

    ifstream reorder_f_valid_score_xgb_aug_nn_63("../valid/reorder_valid_score_xgb_aug_nn_63.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_63xs("../valid/reorder_valid_score_xgb_aug_nn_63xs.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_63ys("../valid/reorder_valid_score_xgb_aug_nn_63ys.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_63xys("../valid/reorder_valid_score_xgb_aug_nn_63xys.csv");  

    ifstream reorder_f_valid_score_knn_count_20("../valid/reorder_valid_score_knn_count_20.csv");
    
    vector<double> weights_all(13, 1.0);
    vector<ifstream*> f_reorder_valid_scores_all = {
    &reorder_f_valid_score_xgb_new2_63, &reorder_f_valid_score_xgb_new2_63xs, &reorder_f_valid_score_xgb_new2_63ys, &reorder_f_valid_score_xgb_new2_63xys,
    &reorder_f_valid_score_63, &reorder_f_valid_score_63xs, &reorder_f_valid_score_63ys, &reorder_f_valid_score_63xys,
    &reorder_f_valid_score_xgb_aug_nn_63, &reorder_f_valid_score_xgb_aug_nn_63xs, &reorder_f_valid_score_xgb_aug_nn_63ys, &reorder_f_valid_score_xgb_aug_nn_63xys,
    &reorder_f_valid_score_knn_count_20 
    }; 

    weights_all[12] = 2.0;

    size_t cache_size = 2000000;

    // partial check
    num_valid_test = cache_size;    

    mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    cout << mapk_score << endl;

    /* testing all end */
    reorder_f_valid_score_63.close();
    reorder_f_valid_score_63xs.close();
    reorder_f_valid_score_63ys.close();
    reorder_f_valid_score_63xys.close();   

    reorder_f_valid_score_xgb_new2_63.close();
    reorder_f_valid_score_xgb_new2_63xs.close();
    reorder_f_valid_score_xgb_new2_63ys.close();
    reorder_f_valid_score_xgb_new2_63xys.close(); 

    reorder_f_valid_score_xgb_aug_nn_63.close();
    reorder_f_valid_score_xgb_aug_nn_63xs.close();
    reorder_f_valid_score_xgb_aug_nn_63ys.close();
    reorder_f_valid_score_xgb_aug_nn_63xys.close();

    reorder_f_valid_score_knn_count_20.close();

    f_valid_test.close();

}

void reorder_test(){
    int max_rank = 8, max_sub = 3;
    size_t num_train = 29118021, num_test = 8607230;
    double mapk_score = 0, weight = 1;

    cout << "reorder_test" << endl;

    /* reorder row_id */
    ifstream f_test_score_63("../train/test_score_63.csv");
    ifstream f_test_score_63xs("../train/test_score_63xs.csv");
    ifstream f_test_score_63ys("../train/test_score_63ys.csv");
    ifstream f_test_score_63xys("../train/test_score_63xys.csv");

    ofstream f_test_score_63_reorder("../train/reorder_test_score_63.csv");
    ofstream f_test_score_63xs_reorder("../train/reorder_test_score_63xs.csv");
    ofstream f_test_score_63ys_reorder("../train/reorder_test_score_63ys.csv");
    ofstream f_test_score_63xys_reorder("../train/reorder_test_score_63xys.csv");

    reorder_score(f_test_score_63, f_test_score_63_reorder);
    reorder_score(f_test_score_63xs, f_test_score_63xs_reorder);
    reorder_score(f_test_score_63ys, f_test_score_63ys_reorder);
    reorder_score(f_test_score_63xys, f_test_score_63xys_reorder);

    f_test_score_63.close();
    f_test_score_63xs.close();
    f_test_score_63ys.close();
    f_test_score_63xys.close();
    f_test_score_63_reorder.close();
    f_test_score_63xs_reorder.close();
    f_test_score_63ys_reorder.close();
    f_test_score_63xys_reorder.close();

    ifstream f_test_score_xgb_new2_63("../train/test_score_xgb_new2_63.csv");
    ifstream f_test_score_xgb_new2_63xs("../train/test_score_xgb_new2_63xs.csv");
    ifstream f_test_score_xgb_new2_63ys("../train/test_score_xgb_new2_63ys.csv");
    ifstream f_test_score_xgb_new2_63xys("../train/test_score_xgb_new2_63xys.csv");

    ofstream f_test_score_xgb_new2_63_reorder("../train/reorder_test_score_xgb_new2_63.csv");
    ofstream f_test_score_xgb_new2_63xs_reorder("../train/reorder_test_score_xgb_new2_63xs.csv");
    ofstream f_test_score_xgb_new2_63ys_reorder("../train/reorder_test_score_xgb_new2_63ys.csv");
    ofstream f_test_score_xgb_new2_63xys_reorder("../train/reorder_test_score_xgb_new2_63xys.csv");

    reorder_score(f_test_score_xgb_new2_63, f_test_score_xgb_new2_63_reorder);
    reorder_score(f_test_score_xgb_new2_63xs, f_test_score_xgb_new2_63xs_reorder);
    reorder_score(f_test_score_xgb_new2_63ys, f_test_score_xgb_new2_63ys_reorder);
    reorder_score(f_test_score_xgb_new2_63xys, f_test_score_xgb_new2_63xys_reorder);

    f_test_score_xgb_new2_63.close();
    f_test_score_xgb_new2_63xs.close();
    f_test_score_xgb_new2_63ys.close();
    f_test_score_xgb_new2_63xys.close();
    f_test_score_xgb_new2_63_reorder.close();
    f_test_score_xgb_new2_63xs_reorder.close();
    f_test_score_xgb_new2_63ys_reorder.close();
    f_test_score_xgb_new2_63xys_reorder.close();

    ifstream f_test_score_xgb_aug_nn_63("../train/test_score_xgb_aug_nn_63.csv");
    ifstream f_test_score_xgb_aug_nn_63xs("../train/test_score_xgb_aug_nn_63xs.csv");
    ifstream f_test_score_xgb_aug_nn_63ys("../train/test_score_xgb_aug_nn_63ys.csv");
    ifstream f_test_score_xgb_aug_nn_63xys("../train/test_score_xgb_aug_nn_63xys.csv");

    ofstream f_test_score_xgb_aug_nn_63_reorder("../train/reorder_test_score_xgb_aug_nn_63.csv");
    ofstream f_test_score_xgb_aug_nn_63xs_reorder("../train/reorder_test_score_xgb_aug_nn_63xs.csv");
    ofstream f_test_score_xgb_aug_nn_63ys_reorder("../train/reorder_test_score_xgb_aug_nn_63ys.csv");
    ofstream f_test_score_xgb_aug_nn_63xys_reorder("../train/reorder_test_score_xgb_aug_nn_63xys.csv");    

    reorder_score(f_test_score_xgb_aug_nn_63, f_test_score_xgb_aug_nn_63_reorder);
    reorder_score(f_test_score_xgb_aug_nn_63xs, f_test_score_xgb_aug_nn_63xs_reorder);
    reorder_score(f_test_score_xgb_aug_nn_63ys, f_test_score_xgb_aug_nn_63ys_reorder);
    reorder_score(f_test_score_xgb_aug_nn_63xys, f_test_score_xgb_aug_nn_63xys_reorder);

    f_test_score_xgb_aug_nn_63.close();
    f_test_score_xgb_aug_nn_63xs.close();
    f_test_score_xgb_aug_nn_63ys.close();
    f_test_score_xgb_aug_nn_63xys.close();
    f_test_score_xgb_aug_nn_63_reorder.close();
    f_test_score_xgb_aug_nn_63xs_reorder.close();
    f_test_score_xgb_aug_nn_63ys_reorder.close();
    f_test_score_xgb_aug_nn_63xys_reorder.close();

    ifstream f_test_score_knn_count_20("../train/test_score_knn_count_20.csv");
    ofstream f_test_score_knn_count_20_reorder("../train/reorder_test_score_knn_count_20.csv");

    reorder_score(f_test_score_knn_count_20, f_test_score_knn_count_20_reorder);
    f_test_score_knn_count_20.close();
    f_test_score_knn_count_20_reorder.close();

}

void run_merge(){
    int max_rank = 8, max_sub = 3;
    size_t num_train = 29118021, num_test = 8607230;
    double mapk_score = 0, weight = 1;

    cout << "run_merge" << endl;

    ifstream f_train ("../input/train.csv");
    ifstream f_test("../input/test.csv");

    /* generate score from file */
    ofstream f_test_sub("../output/grid_all_sub.csv");

    ifstream reorder_f_test_score_63("../train/reorder_test_score_63.csv");
    ifstream reorder_f_test_score_63xs("../train/reorder_test_score_63xs.csv");
    ifstream reorder_f_test_score_63ys("../train/reorder_test_score_63ys.csv");
    ifstream reorder_f_test_score_63xys("../train/reorder_test_score_63xys.csv");

    ifstream reorder_f_test_score_xgb_new2_63("../train/reorder_test_score_xgb_new2_63.csv");
    ifstream reorder_f_test_score_xgb_new2_63xs("../train/reorder_test_score_xgb_new2_63xs.csv");
    ifstream reorder_f_test_score_xgb_new2_63ys("../train/reorder_test_score_xgb_new2_63ys.csv");
    ifstream reorder_f_test_score_xgb_new2_63xys("../train/reorder_test_score_xgb_new2_63xys.csv");    

    ifstream reorder_f_test_score_xgb_aug_nn_63("../train/reorder_test_score_xgb_aug_nn_63.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_63xs("../train/reorder_test_score_xgb_aug_nn_63xs.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_63ys("../train/reorder_test_score_xgb_aug_nn_63ys.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_63xys("../train/reorder_test_score_xgb_aug_nn_63xys.csv");  

    ifstream reorder_f_test_score_knn_count_20("../train/reorder_test_score_knn_count_20.csv");

    vector<double> weights_all(13, 1.0);
    vector<ifstream*> f_reorder_test_scores_all = {
    &reorder_f_test_score_xgb_new2_63, &reorder_f_test_score_xgb_new2_63xs, &reorder_f_test_score_xgb_new2_63ys, &reorder_f_test_score_xgb_new2_63xys,
    &reorder_f_test_score_63, &reorder_f_test_score_63xs, &reorder_f_test_score_63ys, &reorder_f_test_score_63xys,
    &reorder_f_test_score_xgb_aug_nn_63, &reorder_f_test_score_xgb_aug_nn_63xs, &reorder_f_test_score_xgb_aug_nn_63ys, &reorder_f_test_score_xgb_aug_nn_63xys,
    &reorder_f_test_score_knn_count_20 
    };    

    size_t cache_size = 1000000;

    weights_all[12] = 2.0;

    file_cache_generate_sub(f_test_sub, f_reorder_test_scores_all, weights_all,
            max_sub, num_test, cache_size);


    f_test_sub.close();

    reorder_f_test_score_63.close();
    reorder_f_test_score_63xs.close();
    reorder_f_test_score_63ys.close();
    reorder_f_test_score_63xys.close();   

    reorder_f_test_score_xgb_new2_63.close();
    reorder_f_test_score_xgb_new2_63xs.close();
    reorder_f_test_score_xgb_new2_63ys.close();
    reorder_f_test_score_xgb_new2_63xys.close(); 

    reorder_f_test_score_xgb_aug_nn_63.close();
    reorder_f_test_score_xgb_aug_nn_63xs.close();
    reorder_f_test_score_xgb_aug_nn_63ys.close();
    reorder_f_test_score_xgb_aug_nn_63xys.close();

    reorder_f_test_score_knn_count_20.close();

    /* new version end */
}

int main(){
    int max_rank = 8, max_sub = 3, weight = 1;
    size_t num_train = 29118021, num_test = 8607230;
    
    string timestamp = get_timestamp();

    ifstream f_train ("../input/train.csv");
    ifstream f_test("../input/test.csv");

    /* generate validation data set begin */

    ofstream f_valid_train("../valid/valid_train.csv");
    ofstream f_valid_test("../valid/valid_test.csv");

    time_split_generate_validate(655200, f_train, f_valid_train, f_valid_test);

    f_valid_train.close();
    f_valid_test.close();

    /* generate validation dat set end */

    /* generate train test hash separated file begin*/
    ofstream f_train_test_hash("../input/train_test_hash_63_159.csv");
    ofstream f_train_test_hash_xs("../input/train_test_hash_63_159_xs.csv");
    ofstream f_train_test_hash_ys("../input/train_test_hash_63_159_ys.csv");
    ofstream f_train_test_hash_xys("../input/train_test_hash_63_159_xys.csv");

    train_test_split_by_grid(63, 159, f_train, f_test, f_train_test_hash, nulltime, prep_xy);
    train_test_split_by_grid(63, 159, f_train, f_test, f_train_test_hash_xs, nulltime, prep_xy_xshift);
    train_test_split_by_grid(63, 159, f_train, f_test, f_train_test_hash_ys, nulltime, prep_xy_yshift);
    train_test_split_by_grid(63, 159, f_train, f_test, f_train_test_hash_xys, nulltime, prep_xy_xyshift);

    /* generate hash separated file end*/

    /* check validation score from existing results */
    reorder_validation();
    check_validation();
    /* check validation score from existing results end */

    /* merge result using score*/
    reorder_test();
    run_merge();
    /* merge result using score*/
}