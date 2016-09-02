#include "grid_split.h"
#include "my_utils.h"
#include "grid_utils.h"
#include <unordered_map>
#include <string>
#include <iostream>

using namespace std;


void split_by_grid(const unsigned int x_range, const unsigned int y_range, ifstream &f_input, ofstream &f_output, unsigned int (*tfunc_ptr) (unsigned int),
    pair<unsigned int, unsigned int> (*xyfunc_ptr)(const double, const double, const unsigned int, const unsigned int)){
    unordered_map<unsigned int, vector<string> > grid_lines;
    string line, header;
    getline(f_input, header);
    while ( getline(f_input, line) ){
        vector<string> tokens;
        split_line(line, ',', tokens);

        unsigned int row_id, cur_time, accuracy, hash, ix, iy;
        unsigned long place_id;
        double x , y;

        row_id = stoi(tokens[0]);
        accuracy = stoi(tokens[3]);
        cur_time = stoi(tokens[4]);
        place_id = stoul(tokens[5]);
        x = stod(tokens[1]);
        y = stod(tokens[2]);

        
        pair<unsigned int, unsigned int> ix_iy = xyfunc_ptr(x, y, x_range, y_range);
        ix = ix_iy.first;
        iy = ix_iy.second;

        unsigned int conv_time;
        conv_time = (*tfunc_ptr)(cur_time);

        if (x_range <= 512 && y_range <=1024){
            hash = iy << 22 | ix << 13 | conv_time;
        }else{
            hash = iy << 21 | ix << 12 | conv_time;
        }

        if (grid_lines.count(hash)){
            grid_lines[hash].push_back(line);
        }else{
            grid_lines[hash] = vector<string>(1, line);
        }
    }
    f_input.clear();
    f_input.seekg(0, ios_base::beg);


    cout << "total grid:" << grid_lines.size() << endl;
    f_output << "hash," << header << endl;
    for (auto it = grid_lines.begin(); it != grid_lines.end(); it++){
        int local_hash = (*it).first;
        vector<string> &local_grid_lines = (*it).second;
        int num_local_lines = local_grid_lines.size();
        f_output << '^' << local_hash << ',' << num_local_lines << endl;
        for (int j = 0; j < num_local_lines; j++){
            f_output << local_hash << ',' << local_grid_lines[j] << endl;
        }
        f_output << '$' << local_hash << endl;
        cout << local_hash << '\t' << num_local_lines << endl;
    }

}

void train_test_split_by_grid(const unsigned int x_range, const unsigned int y_range, ifstream &ftrain_input, ifstream &ftest_input, ofstream &f_output, 
    unsigned int (*tfunc_ptr) (unsigned int), pair<unsigned int, unsigned int> (*xyfunc_ptr)(const double, const double, const unsigned int, const unsigned int)){
    unordered_map<unsigned int, vector<string> > train_grid_lines;
    unordered_map<unsigned int, vector<string> > test_grid_lines;
    string line, header;

    /* ftrain input */
    getline(ftrain_input, header);
    while ( getline(ftrain_input, line) ){
        vector<string> tokens;
        split_line(line, ',', tokens);

        unsigned int row_id, cur_time, accuracy, hash, ix, iy;
        unsigned long place_id;
        double x , y;

        row_id = stoi(tokens[0]);
        accuracy = stoi(tokens[3]);
        cur_time = stoi(tokens[4]);
        place_id = stoul(tokens[5]);
        x = stod(tokens[1]);
        y = stod(tokens[2]);

        
        pair<unsigned int, unsigned int> ix_iy = xyfunc_ptr(x, y, x_range, y_range);
        ix = ix_iy.first;
        iy = ix_iy.second;

        unsigned int conv_time;
        conv_time = (*tfunc_ptr)(cur_time);

        if (x_range <= 512 && y_range <= 1024){
            hash = iy << 22 | ix << 13 | conv_time;
        }else{
            hash = iy << 21 | ix << 12 | conv_time;
        }

        if (train_grid_lines.count(hash)){
            train_grid_lines[hash].push_back(line);
        }else{
            train_grid_lines[hash] = vector<string>(1, line);
        }
    }
    ftrain_input.clear();
    ftrain_input.seekg(0, ios_base::beg);

    /* ftest input */
    getline(ftest_input, line);
    while ( getline(ftest_input, line) ){
        vector<string> tokens;
        split_line(line, ',', tokens);

        unsigned int row_id, cur_time, accuracy, hash, ix, iy;
        unsigned long place_id;
        double x , y;

        row_id = stoi(tokens[0]);
        accuracy = stoi(tokens[3]);
        cur_time = stoi(tokens[4]);
        x = stod(tokens[1]);
        y = stod(tokens[2]);

        
        pair<unsigned int, unsigned int> ix_iy = xyfunc_ptr(x, y, x_range, y_range);
        ix = ix_iy.first;
        iy = ix_iy.second;

        unsigned int conv_time;
        conv_time = (*tfunc_ptr)(cur_time);

        if (x_range <= 512 && y_range <= 1024){
            hash = iy << 22 | ix << 13 | conv_time;
        }else{
            hash = iy << 21 | ix << 12 | conv_time;
        }

        if (test_grid_lines.count(hash)){
            test_grid_lines[hash].push_back(line);
        }else{
            test_grid_lines[hash] = vector<string>(1, line);
        }
    }
    ftest_input.clear();
    ftest_input.seekg(0, ios_base::beg);

    /* here we only make sure train grid are all included, there is no prediction can be made if train grid missing  */
    cout << "total train grid:" << train_grid_lines.size() << endl;
    cout << "total test grid:" << test_grid_lines.size() << endl;
    f_output << "hash," << header << endl;
    for (auto it = train_grid_lines.begin(); it != train_grid_lines.end(); it++){
        int local_hash = (*it).first;
        vector<string> &local_train_grid_lines = (*it).second;
        int num_local_train_lines = local_train_grid_lines.size();
        f_output << '^' << local_hash << ',' << num_local_train_lines << endl;
        for (int j = 0; j < num_local_train_lines; j++){
            f_output << local_hash << ',' << local_train_grid_lines[j] << endl;
        }
        f_output << '?' << local_hash << endl;
        cout << local_hash << ",train:" << num_local_train_lines << endl;

        int num_local_test_lines = 0;
        if (test_grid_lines.count(local_hash)){
            vector<string> &local_test_grid_lines = test_grid_lines[local_hash];
            num_local_test_lines = local_test_grid_lines.size();
            f_output << '!' << local_hash << ',' << num_local_test_lines << endl;
            for (int j = 0; j < num_local_test_lines; j++){
                f_output << local_hash << ',' << local_test_grid_lines[j]<< endl;
            }
            f_output << '$' << local_hash << endl;
        }else{
            f_output << '!' << local_hash << ',' << num_local_test_lines << endl;
            f_output << '$' << local_hash << endl;
        }
        cout << local_hash << ",test:" << num_local_test_lines << endl;
    }

}

// assign weight of sample using different weights according to accuracy represented as a rectangle with ratio
void train_test_split_by_grid_weight_by_acc(const unsigned int x_range, const unsigned int y_range, double acc_ratio, double aug_ratio, 
    ifstream &ftrain_input, ifstream &ftest_input, ofstream &f_output, 
    unsigned int (*tfunc_ptr) (unsigned int), pair<unsigned int, unsigned int> (*xyfunc_ptr)(const double, const double, const unsigned int, const unsigned int),
    vector<double> (*anchor_func)(const unsigned int, const unsigned int, const unsigned int, const unsigned int)){
    unordered_map<unsigned int, vector<string> > train_grid_lines;
    unordered_map<unsigned int, vector<string> > test_grid_lines;
    string line, header;

    /* ftrain input */
    getline(ftrain_input, header);
    while ( getline(ftrain_input, line) ){
        vector<string> tokens;
        split_line(line, ',', tokens);

        unsigned int row_id, cur_time, accuracy, hash;
        unsigned long place_id;
        double x , y;

        row_id = stoi(tokens[0]);
        accuracy = stoi(tokens[3]);
        cur_time = stoi(tokens[4]);
        place_id = stoul(tokens[5]);
        x = stod(tokens[1]);
        y = stod(tokens[2]);

        unsigned int orig_ix, orig_iy;
        pair<unsigned int, unsigned int> orig_ix_iy = xyfunc_ptr(x, y, x_range, y_range);
        orig_ix = orig_ix_iy.first;
        orig_iy = orig_ix_iy.second;

        double x_min, x_max, y_min, y_max, x_span, y_span;
        x_span = double(accuracy) / 1000;
        y_span = x_span / acc_ratio;
        x_min = x - x_span;
        x_max = x + x_span;
        y_min = y - y_span;
        y_max = y + y_span;

        // make sure that the boundary within the grid
        x_min = min(max(x_min, 0.0), 10.0);
        x_max = min(max(x_max, 0.0), 10.0);
        y_min = min(max(y_min, 0.0), 10.0);
        y_max = min(max(y_max, 0.0), 10.0);
        
        // augment the the training grid
        double target_x_min, target_x_max, target_y_min, target_y_max;
        unsigned int ix_min, ix_max, iy_min, iy_max;
        target_x_min = x - 10.0 / x_range * aug_ratio;
        target_x_max = x + 10.0 / x_range * aug_ratio;
        target_y_min = y - 10.0 / y_range * aug_ratio;
        target_y_max = y + 10.0 / y_range * aug_ratio;

        target_x_min = min(max(max(target_x_min, x_min), 0.0), 10.0);
        target_x_max = min(max(min(target_x_max, x_max), 0.0), 10.0);
        target_y_min = min(max(max(target_y_min, y_min), 0.0), 10.0);
        target_y_max = min(max(min(target_y_max, y_max), 0.0), 10.0);

        pair<unsigned int, unsigned int> ix_iy_min = xyfunc_ptr(target_x_min, target_y_min, x_range, y_range);
        pair<unsigned int, unsigned int> ix_iy_max = xyfunc_ptr(target_x_max, target_y_max, x_range, y_range);
        ix_min = ix_iy_min.first;
        iy_min = ix_iy_min.second;
        ix_max = ix_iy_max.first;
        iy_max = ix_iy_max.second;

        double global_area =  rect_intersect_global(x_min, x_max, y_min, y_max);

        unsigned int conv_time;
        conv_time = (*tfunc_ptr)(cur_time);

        unsigned int ix, iy;
        for (ix = ix_min; ix <= ix_max; ix++){
            for (iy = iy_min; iy <= iy_max; iy++){
                double local_area, local_weight;
                local_area = rect_intersect_local(x_min, x_max, y_min, y_max, ix, iy, x_range, y_range, anchor_func);
                local_weight = local_area / global_area;
                if (local_weight < 1e-6) continue;

                if (x_range <= 512 && y_range <= 1024){
                    hash = iy << 22 | ix << 13 | conv_time;
                }else{
                    hash = iy << 21 | ix << 12 | conv_time;
                }

                unsigned int inside = 0;
                if (ix == orig_ix && iy == orig_iy){
                    inside = 1;
                }

                string new_line;
                new_line = line + "," + to_string(local_weight) + "," + to_string(inside);

                if (train_grid_lines.count(hash)){
                    train_grid_lines[hash].push_back(new_line);
                }else{
                    train_grid_lines[hash] = vector<string>(1, new_line);
                }
            }
        }

    }
    ftrain_input.clear();
    ftrain_input.seekg(0, ios_base::beg);

    /* ftest input */
    getline(ftest_input, line);
    while ( getline(ftest_input, line) ){
        vector<string> tokens;
        split_line(line, ',', tokens);

        unsigned int row_id, cur_time, accuracy, hash, ix, iy;
        unsigned long place_id;
        double x , y;

        row_id = stoi(tokens[0]);
        accuracy = stoi(tokens[3]);
        cur_time = stoi(tokens[4]);
        x = stod(tokens[1]);
        y = stod(tokens[2]);

        
        pair<unsigned int, unsigned int> ix_iy = xyfunc_ptr(x, y, x_range, y_range);
        ix = ix_iy.first;
        iy = ix_iy.second;

        unsigned int conv_time;
        conv_time = (*tfunc_ptr)(cur_time);

        if (x_range <= 512 && y_range <= 1024){
            hash = iy << 22 | ix << 13 | conv_time;
        }else{
            hash = iy << 21 | ix << 12 | conv_time;
        }

        if (test_grid_lines.count(hash)){
            test_grid_lines[hash].push_back(line);
        }else{
            test_grid_lines[hash] = vector<string>(1, line);
        }
    }
    ftest_input.clear();
    ftest_input.seekg(0, ios_base::beg);

    /* here we only make sure train grid are all included, there is no prediction can be made if train grid missing  */
    cout << "total train grid:" << train_grid_lines.size() << endl;
    cout << "total test grid:" << test_grid_lines.size() << endl;
    f_output << "hash," << header << ",weight,inside" << endl;
    for (auto it = train_grid_lines.begin(); it != train_grid_lines.end(); it++){
        int local_hash = (*it).first;
        vector<string> &local_train_grid_lines = (*it).second;
        int num_local_train_lines = local_train_grid_lines.size();
        f_output << '^' << local_hash << ',' << num_local_train_lines << endl;
        for (int j = 0; j < num_local_train_lines; j++){
            f_output << local_train_grid_lines[j] << endl;
        }
        f_output << '?' << local_hash << endl;
        cout << local_hash << ",train:" << num_local_train_lines << endl;

        int num_local_test_lines = 0;
        if (test_grid_lines.count(local_hash)){
            vector<string> &local_test_grid_lines = test_grid_lines[local_hash];
            num_local_test_lines = local_test_grid_lines.size();
            f_output << '!' << local_hash << ',' << num_local_test_lines << endl;
            for (int j = 0; j < num_local_test_lines; j++){
                f_output << local_test_grid_lines[j]<< endl;
            }
            f_output << '$' << local_hash << endl;
        }else{
            f_output << '!' << local_hash << ',' << num_local_test_lines << endl;
            f_output << '$' << local_hash << endl;
        }
        cout << local_hash << ",test:" << num_local_test_lines << endl;
    }

}

// suppose accuracy is horizontal accuaracy and use gaussian distribution
void train_test_split_by_grid_weight_gaussian(const unsigned int x_range, const unsigned int y_range, double acc_ratio, double aug_ratio,
    ifstream &ftrain_input, ifstream &ftest_input, ofstream &f_output, 
    unsigned int (*tfunc_ptr) (unsigned int), pair<unsigned int, unsigned int> (*xyfunc_ptr)(const double, const double, const unsigned int, const unsigned int),
    vector<double> (*anchor_func)(const unsigned int, const unsigned int, const unsigned int, const unsigned int)){
    unordered_map<unsigned int, vector<string> > train_grid_lines;
    unordered_map<unsigned int, vector<string> > test_grid_lines;
    string line, header;

    /* ftrain input */
    getline(ftrain_input, header);
    while ( getline(ftrain_input, line) ){
        vector<string> tokens;
        split_line(line, ',', tokens);

        unsigned int row_id, cur_time, accuracy, hash;
        unsigned long place_id;
        double x , y;

        row_id = stoi(tokens[0]);
        accuracy = stoi(tokens[3]);
        cur_time = stoi(tokens[4]);
        place_id = stoul(tokens[5]);
        x = stod(tokens[1]);
        y = stod(tokens[2]);

        unsigned int orig_ix, orig_iy;
        pair<unsigned int, unsigned int> orig_ix_iy = xyfunc_ptr(x, y, x_range, y_range);
        orig_ix = orig_ix_iy.first;
        orig_iy = orig_ix_iy.second;

        double x_span, y_span;
        x_span = double(accuracy) / 1000;
        y_span = x_span / acc_ratio;
        
        // augment the the training grid and find the grid would be using the point
        double target_x_min, target_x_max, target_y_min, target_y_max;
        unsigned int ix_min, ix_max, iy_min, iy_max;
        target_x_min = x - 10.0 / x_range * aug_ratio;
        target_x_max = x + 10.0 / x_range * aug_ratio;
        target_y_min = y - 10.0 / y_range * aug_ratio;
        target_y_max = y + 10.0 / y_range * aug_ratio;

        target_x_min = min(max(target_x_min, 0.0), 10.0);
        target_x_max = min(max(target_x_max, 0.0), 10.0);
        target_y_min = min(max(target_y_min, 0.0), 10.0);
        target_y_max = min(max(target_y_max, 0.0), 10.0);

        pair<unsigned int, unsigned int> ix_iy_min = xyfunc_ptr(target_x_min, target_y_min, x_range, y_range);
        pair<unsigned int, unsigned int> ix_iy_max = xyfunc_ptr(target_x_max, target_y_max, x_range, y_range);
        ix_min = ix_iy_min.first;
        iy_min = ix_iy_min.second;
        ix_max = ix_iy_max.first;
        iy_max = ix_iy_max.second;

        unsigned int conv_time;
        conv_time = (*tfunc_ptr)(cur_time);

        unsigned int ix, iy;
        for (ix = ix_min; ix <= ix_max; ix++){
            for (iy = iy_min; iy <= iy_max; iy++){
                double local_weight;
                local_weight = gaussain_prob_local(x, x_span, y, y_span, ix, iy, x_range, y_range, anchor_func);
                if (local_weight < 1e-6) continue;

                if (x_range <= 512 && y_range <= 1024){
                    hash = iy << 22 | ix << 13 | conv_time;
                }else{
                    hash = iy << 21 | ix << 12 | conv_time;
                }

                unsigned int inside = 0;
                if (ix == orig_ix && iy == orig_iy){
                    inside = 1;
                }

                string new_line;
                new_line = line + "," + to_string(local_weight) + "," + to_string(inside);

                if (train_grid_lines.count(hash)){
                    train_grid_lines[hash].push_back(new_line);
                }else{
                    train_grid_lines[hash] = vector<string>(1, new_line);
                }
            }
        }

    }
    ftrain_input.clear();
    ftrain_input.seekg(0, ios_base::beg);

    /* ftest input */
    getline(ftest_input, line);
    while ( getline(ftest_input, line) ){
        vector<string> tokens;
        split_line(line, ',', tokens);

        unsigned int row_id, cur_time, accuracy, hash, ix, iy;
        unsigned long place_id;
        double x , y;

        row_id = stoi(tokens[0]);
        accuracy = stoi(tokens[3]);
        cur_time = stoi(tokens[4]);
        x = stod(tokens[1]);
        y = stod(tokens[2]);

        
        pair<unsigned int, unsigned int> ix_iy = xyfunc_ptr(x, y, x_range, y_range);
        ix = ix_iy.first;
        iy = ix_iy.second;

        unsigned int conv_time;
        conv_time = (*tfunc_ptr)(cur_time);

        if (x_range <= 512 && y_range <= 1024){
            hash = iy << 22 | ix << 13 | conv_time;
        }else{
            hash = iy << 21 | ix << 12 | conv_time;
        }

        if (test_grid_lines.count(hash)){
            test_grid_lines[hash].push_back(line);
        }else{
            test_grid_lines[hash] = vector<string>(1, line);
        }
    }
    ftest_input.clear();
    ftest_input.seekg(0, ios_base::beg);

    /* here we only make sure train grid are all included, there is no prediction can be made if train grid missing  */
    cout << "total train grid:" << train_grid_lines.size() << endl;
    cout << "total test grid:" << test_grid_lines.size() << endl;
    f_output << "hash," << header << ",weight,inside" << endl;

    for (unsigned int ix = 0; ix < x_range; ix++){
        for (unsigned int iy = 0; iy < y_range; iy++){
            unsigned int local_hash;

            if (x_range <= 512 && y_range <= 1024){
                local_hash = iy << 22 | ix << 13 | 0;
            }else{
                local_hash = iy << 21 | ix << 12 | 0;
            }
            if (!train_grid_lines.count(local_hash)) continue;
            vector<string> &local_train_grid_lines = train_grid_lines[local_hash];
            int num_local_train_lines = local_train_grid_lines.size();
            f_output << '^' << local_hash << ',' << num_local_train_lines << endl;
            for (int j = 0; j < num_local_train_lines; j++){
                f_output << local_train_grid_lines[j] << endl;
            }
            f_output << '?' << local_hash << endl;
            cout << local_hash << ",train:" << num_local_train_lines << endl;

            int num_local_test_lines = 0;
            if (test_grid_lines.count(local_hash)){
                vector<string> &local_test_grid_lines = test_grid_lines[local_hash];
                num_local_test_lines = local_test_grid_lines.size();
                f_output << '!' << local_hash << ',' << num_local_test_lines << endl;
                for (int j = 0; j < num_local_test_lines; j++){
                    f_output << local_test_grid_lines[j]<< endl;
                }
                f_output << '$' << local_hash << endl;
            }else{
                f_output << '!' << local_hash << ',' << num_local_test_lines << endl;
                f_output << '$' << local_hash << endl;
            }
            cout << local_hash << ",test:" << num_local_test_lines << endl;
        }
    }
}