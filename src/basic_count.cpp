#include "basic_count.h"
#include "my_utils.h"
#include "grid_utils.h"
#include <iostream>
#include <queue>
#include <algorithm>

using namespace std;

void generate_grid(const unsigned int x_range, const unsigned int y_range, ifstream &f_train, unsigned int (*tfunc_ptr) (unsigned int),
        unordered_map<unsigned int, unordered_map<unsigned long, unsigned int> > &grid,
        pair<unsigned int, unsigned int> (*xyfunc_ptr)(const double, const double, const unsigned int, const unsigned int)){
    string line;
    getline(f_train, line);
    while ( getline(f_train, line) ){
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

        
        pair<unsigned int, unsigned int> ix_iy = (*xyfunc_ptr)(x, y, x_range, y_range);
        ix = ix_iy.first;
        iy = ix_iy.second;

        unsigned int conv_time;
        conv_time = (*tfunc_ptr)(cur_time);

        if (x_range <= 512 && y_range <= 1024){
            hash = iy << 22 | ix << 13 | conv_time;
        }else{
            hash = iy << 21 | ix << 12 | conv_time;
        }

        if (grid.count(hash)){
            unordered_map<unsigned long, unsigned int> &local_grid = grid[hash];
            if (local_grid.count(place_id)){
                local_grid[place_id]++;
            }else{
                local_grid[place_id] = 1;
            }
        }else{
            grid[hash] = unordered_map<unsigned long, unsigned int>( {{place_id, 1}} );
        }
    }
    f_train.clear();
    f_train.seekg(0, ios_base::beg);
}

void generate_top(int max_rank, unordered_map<unsigned int, vector<unsigned long> > &grid_top,
        unordered_map<unsigned int, unordered_map<unsigned long, unsigned int> > &grid){
    // clear grid_top
    grid_top.clear();

    for (auto it = grid.begin(); it != grid.end(); it++){
        unsigned int hash = (*it).first;
        unordered_map<unsigned long, unsigned int> &local_grid = (*it).second;

        priority_queue<pair<unsigned int, unsigned long> > local_pq;
        for (auto local_it = local_grid.begin(); local_it != local_grid.end(); local_it++){
            local_pq.push(pair<unsigned int, unsigned long>( (*local_it).second, (*local_it).first ));
        }

        grid_top[hash] = vector<unsigned long>();
        int rank = 0;
        while (!local_pq.empty() && rank < max_rank){
            const pair<unsigned int, unsigned long> &top_item = local_pq.top();
            grid_top[hash].push_back(top_item.second);
            local_pq.pop();
        }
    }

    grid.clear();
}

void generate_top_ratio(int max_rank, unordered_map<unsigned int, vector<pair<unsigned long, double> > > &grid_top_ratio,
        unordered_map<unsigned int, unordered_map<unsigned long, unsigned int> > &grid){
    // clear grid_top
    grid_top_ratio.clear();

    for (auto it = grid.begin(); it != grid.end(); it++){
        unsigned int hash = (*it).first;
        unordered_map<unsigned long, unsigned int> &local_grid = (*it).second;

        priority_queue<pair<unsigned int, unsigned long> > local_pq;
        unsigned int count_total = 0;
        for (auto local_it = local_grid.begin(); local_it != local_grid.end(); local_it++){
            count_total += (*local_it).second;
            local_pq.push(pair<unsigned int, unsigned long>( (*local_it).second, (*local_it).first ));
        }

        grid_top_ratio[hash] = vector<pair<unsigned long, double> >();
        int rank = 0;
        while (!local_pq.empty() && rank < max_rank){
            const pair<unsigned int, unsigned long> &top_item = local_pq.top();
            double ratio = double(top_item.first) / count_total;
            grid_top_ratio[hash].push_back(pair<unsigned long, double>(top_item.second, ratio));
            local_pq.pop();
        }
    }

    grid.clear();
}


void update_score(const unsigned int x_range, const unsigned int y_range, ifstream &f_test, int max_rank, double weight, unsigned int (*tfunc_ptr) (unsigned int),
    vector<unordered_map<unsigned long, double> > &score, unordered_map<unsigned int, vector<unsigned long> > &grid_top){
    string line;
    getline(f_test, line);
    int i_line = 0;
    int empty_count = 0;
    int all_miss_count = 0;
    while ( getline(f_test, line) ){
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

        pair<unsigned int, unsigned int> ix_iy = prep_xy(x, y, x_range, y_range);
        ix = ix_iy.first;
        iy = ix_iy.second;

        unsigned int conv_time;
        conv_time = (*tfunc_ptr)(cur_time);


        if (x_range <= 512 && y_range <= 1024){
            hash = iy << 22 | ix << 13 | conv_time;
        }else{
            hash = iy << 21 | ix << 12 | conv_time;
        }

        unordered_map<unsigned long, double> &line_score = score[i_line];
        if (grid_top.count(hash)){
            vector<unsigned long> &top_places = grid_top[hash]; 
            int num_top = top_places.size();
            for (int i = 0; i < num_top && i < max_rank; i++ ){
                unsigned long top_place = top_places[i];
                if (line_score.count(top_place)){
                    line_score[top_place] += weight * (max_rank - i);
                }else{
                    line_score[top_place] = weight * (max_rank - i);
                }
            }
        }else{
            empty_count++;
        }
        if (line_score.empty())
            all_miss_count++;

        i_line++;
    }
    cout << empty_count++ << '\t' << all_miss_count << endl;
    f_test.clear();
    f_test.seekg(0, ios_base::beg);
}

void update_score_ratio(const unsigned int x_range, const unsigned int y_range, ifstream &f_test, int max_rank, double weight, unsigned int (*tfunc_ptr) (unsigned int),
    vector<unordered_map<unsigned long, double> > &score, unordered_map<unsigned int, vector<pair<unsigned long, double> > > &grid_top_ratio){
    string line;
    getline(f_test, line);
    int i_line = 0;
    int empty_count = 0;
    int all_miss_count = 0;
    while ( getline(f_test, line) ){
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

        pair<unsigned int, unsigned int> ix_iy = prep_xy(x, y, x_range, y_range);
        ix = ix_iy.first;
        iy = ix_iy.second;

        unsigned int conv_time;
        conv_time = (*tfunc_ptr)(cur_time);

        if (x_range <= 512 && y_range <= 1024){
            hash = iy << 22 | ix << 13 | conv_time;
        }else{
            hash = iy << 21 | ix << 12 | conv_time;
        }

        unordered_map<unsigned long, double> &line_score = score[i_line];
        if (grid_top_ratio.count(hash)){
            vector<pair<unsigned long, double> > &top_places = grid_top_ratio[hash];
            int num_top = top_places.size();
            for (int i = 0; i < num_top && i < max_rank; i++ ){
                unsigned long top_place = top_places[i].first;
                double top_place_ratio = top_places[i].second;
                if (line_score.count(top_place)){
                    line_score[top_place] += weight * top_place_ratio;
                }else{
                    line_score[top_place] = weight * top_place_ratio;
                }
            }
        }else{
            empty_count++;
        }
        if (line_score.empty())
            all_miss_count++;

        i_line++;
    }
    cout << empty_count++ << '\t' << all_miss_count << endl;
    f_test.clear();
    f_test.seekg(0, ios_base::beg);
}