#include "valid_test_sub.h"
#include "my_utils.h"
#include <iostream>
#include <queue>
#include <algorithm>
#include <string>

using namespace std;

double apk(unordered_set<unsigned long> &actual, vector<unsigned long> &predicted, const int k){
    int num_pred = predicted.size();
    num_pred = num_pred < k ? num_pred : k;

    int num_hit = 0;
    double score = 0;
    for (int i = 0; i < num_pred; i++){
        if (actual.count(predicted[i])){
            num_hit++;
            score += double(num_hit) / (i+1);
        }
    }
    if (num_hit != 0)
        score /= actual.size();

    return score;
}

void time_split_generate_validate(unsigned int split, ifstream &f_train, ofstream &f_valid_train, ofstream &f_valid_test){
    string line;

    getline(f_train, line);
    f_valid_train << line << endl;
    f_valid_test << line << endl;

    while (getline(f_train, line)){
        vector<string> tokens;
        split_line(line, ',', tokens);

        unsigned int cur_time;
        cur_time = stoi(tokens[4]);

        if (cur_time <= split){
            f_valid_train << line << endl;
        }else{
            f_valid_test << line << endl;
        }
    }

    f_train.clear();
    f_train.seekg(0, ios_base::beg);
}

void generate_valid_score(ofstream &f_sub, int max_sub, vector<unordered_map<unsigned long, double> > &score){
    size_t num_lines = score.size();

    f_sub << "row_id,place_id" << endl;

    for (int i_line = 0; i_line < num_lines; i_line++){
        unordered_map<unsigned long, double> &line_score = score[i_line];
        priority_queue<pair<double, unsigned long> > line_pq;
        for (auto it = line_score.begin(); it != line_score.end(); it++){
            line_pq.push( pair<double, unsigned long>( (*it).second, (*it).first) );
        }
        f_sub << i_line << ',';
        int i = 0;
        while (i < max_sub && !line_pq.empty()){
            f_sub << ',';
            f_sub << line_pq.top().second;
            f_sub << ',';
            f_sub << line_pq.top().first;
            line_pq.pop();
            i++;
        }
        f_sub << endl;
    }
}

void generate_sub(ofstream &f_sub, int max_sub, vector<unordered_map<unsigned long, double> > &score){
    size_t num_lines = score.size();

    f_sub << "row_id,place_id" << endl;

    for (int i_line = 0; i_line < num_lines; i_line++){
        unordered_map<unsigned long, double> &line_score = score[i_line];
        priority_queue<pair<double, unsigned long> > line_pq;
        for (auto it = line_score.begin(); it != line_score.end(); it++){
            line_pq.push( pair<double, unsigned long>( (*it).second, (*it).first) );
        }
        f_sub << i_line << ',';
        int i = 0;
        while (i < max_sub && !line_pq.empty()){
            f_sub << ' ';
            f_sub << line_pq.top().second;
            line_pq.pop();
            i++;
        }
        f_sub << endl;
    }
}

double validate_apk(ifstream &f_valid_test, int max_sub, vector<unordered_map<unsigned long, double>> &score){
    size_t num_lines = score.size();
    string line;
    // maybe write a header to the f_valid
    getline(f_valid_test, line);
    double mapk_score = 0;

    for (int i_line = 0; i_line < num_lines; i_line++){
        if ( !getline(f_valid_test, line) ){
            cout << "f_valid not matching with score!" << endl;
            return 0;
        }
        vector<string> tokens;
        split_line(line, ',', tokens);

        unsigned long place_id;
        unsigned int row_id;

        place_id = stoul(tokens[5]);
        row_id = stoi(tokens[0]);

        unordered_set<unsigned long> line_act({place_id});

        unordered_map<unsigned long, double> &line_score = score[i_line];
        priority_queue<pair<double, unsigned long> > line_pq;
        for (auto it = line_score.begin(); it != line_score.end(); it++){
            line_pq.push( pair<double, unsigned long>( (*it).second, (*it).first) );
        }
        vector<unsigned long> line_pred;
        int i = 0;
        while (i < max_sub && !line_pq.empty()){
            line_pred.push_back(line_pq.top().second);
            line_pq.pop();
            i++;
        }
        double cur_score = apk(line_act, line_pred, max_sub);

        // if (cur_score < 0.3)
        //     cout << line << endl;

        mapk_score += cur_score;
    }
    mapk_score /= num_lines;
    f_valid_test.clear();
    f_valid_test.seekg(0, ios_base::beg);

    return mapk_score;
}

void reorder_score(ifstream &f_input, ofstream &f_output){
    string header, line;

    getline(f_input, header);
    unordered_map<unsigned int, string> line_map;
    vector<unsigned int> row_ids;

    while ( getline(f_input, line) ){
        vector<string> tokens;
        split_line(line, ',', tokens);

        unsigned int row_id;

        row_id = stoi(tokens[0]);

        row_ids.push_back(row_id);
        line_map[row_id] = line;
    }
    f_input.clear();
    f_input.seekg(0, ios_base::beg);

    sort(row_ids.begin(), row_ids.end());

    unsigned int num_lines = line_map.size();
    cout << "reorder_score: " << num_lines << endl;

    f_output << header << endl;
    for (int i = 0; i < num_lines; i++){
        unsigned int row_id = row_ids[i];
        f_output << line_map[row_id] << endl;
    }
}

void generate_row_id_map(ifstream &f_valid_test, unordered_map<unsigned int, unsigned int> &row_id_map){
    string line;

    getline(f_valid_test, line);

    unsigned int line_counter = 0;
    while (getline(f_valid_test, line)){
        vector<string> tokens;
        split_line(line, ',', tokens);

        unsigned int row_id;
        row_id = stoi(tokens[0]);

        row_id_map[row_id] = line_counter;
        line_counter++;
    }

    f_valid_test.clear();
    f_valid_test.seekg(0, ios_base::beg);
}


// here we assume that the file is unordered
void file_update_score(ifstream &f_input, double weight, 
        vector<unordered_map<unsigned long, double> > &score,  unordered_map<unsigned int, unsigned int> &row_id_map){
    size_t num_lines = score.size();
    string line;
    // maybe write a header to the f_valid
    getline(f_input, line);

    for (int i_line = 0; i_line < num_lines; i_line++){
        if ( !getline(f_input, line) ){
            cout << "file not matching with score!" << endl;
            // return 0;
        }
        vector<string> tokens;
        split_line(line, ',', tokens);

        int num_tokens, num_pairs;
        num_tokens = tokens.size();
        num_pairs = (tokens.size()-1)/2;

        unsigned int row_id, line_id;

        row_id = stoi(tokens[0]);
        line_id = row_id_map[row_id];

        unordered_map<unsigned long, double> &line_score = score[line_id];
        for (int i = 0; i < num_pairs; i++){
            unsigned long place_id;
            double score;

            place_id = stoul(tokens[i * 2 + 1]);
            score = stod(tokens[i * 2 + 2]);

            if (line_score.count(place_id)){
                line_score[place_id] += weight * score;
            }else{
                line_score[place_id] = weight * score;
            }
        }
    }

    f_input.clear();
    f_input.seekg(0, ios_base::beg);
}

// file_update_score only keep first several
void file_update_score_limited(ifstream &f_input, double weight, unsigned int limit,
        vector<unordered_map<unsigned long, double> > &score,  unordered_map<unsigned int, unsigned int> &row_id_map){
    size_t num_lines = score.size();
    string line;
    // maybe write a header to the f_valid
    getline(f_input, line);

    for (int i_line = 0; i_line < num_lines; i_line++){
        if ( !getline(f_input, line) ){
            cout << "file not matching with score!" << endl;
            // return 0;
        }
        vector<string> tokens;
        split_line(line, ',', tokens);

        int num_tokens, num_pairs;
        num_tokens = tokens.size();
        num_pairs = (tokens.size()-1)/2;

        unsigned int row_id, line_id;

        row_id = stoi(tokens[0]);
        line_id = row_id_map[row_id];

        unordered_map<unsigned long, double> &line_score = score[line_id];
        for (int i = 0; i < num_pairs; i++){
            unsigned long place_id;
            double score;

            place_id = stoul(tokens[i * 2 + 1]);
            score = stod(tokens[i * 2 + 2]);

            if (line_score.count(place_id)){
                line_score[place_id] += weight * score;
            }else{
                line_score[place_id] = weight * score;
            }
        }


        if (limit > 0 && limit < line_score.size()){
            vector<pair<double, unsigned long> > cur_line_scores;
            for (auto it = line_score.begin(); it != line_score.end(); it++){
                cur_line_scores.push_back( pair<double, unsigned long>((*it).second, (*it).first) );
            }
            for (int i = limit; i < cur_line_scores.size(); i++){
                line_score.erase(cur_line_scores[i].second);
            }
        }

    }

    f_input.clear();
    f_input.seekg(0, ios_base::beg);

}


// generate score and submission
void file_generate_sub(ofstream &f_sub, vector<ifstream*> &f_scores, vector<double> &weights,
        int max_sub, size_t num_lines){

    for (int i = 0; i < weights.size(); i++){
        cout << weights[i] << "\t";
    }
    cout << endl;

    int num_files = f_scores.size();
    string line;

    for (int i = 0; i < num_files; i++){
        getline(*(f_scores[i]), line);
    }

    f_sub << "row_id,place_id" << endl;
        
    for (int i_line = 0; i_line < num_lines; i_line++){
        // generate the candidate place_ids from the f_scores
        unsigned int row_id = -1;
        unordered_map<unsigned long, double> line_score;

        for (int i_file = 0; i_file < num_files; i_file++){
            if ( !getline(*(f_scores[i_file]), line) ){
                cout << "file not matching with score!" << endl;
                // return 0;
            }
            vector<string> score_tokens;
            split_line(line, ',', score_tokens);

            unsigned int score_row_id;
            int num_tokens, num_pairs;
            num_tokens = score_tokens.size();
            num_pairs = (score_tokens.size()-1)/2;

            score_row_id = stoi(score_tokens[0]);

            if (row_id == -1){
                row_id = score_row_id;
            }

            if (score_row_id != row_id){
                cout << "score row id not matching:" << row_id << ":" << score_row_id << endl;
            }

            double weight = weights[i_file];
            for (int i = 0; i < num_pairs; i++){
                unsigned long place_id;
                double score;

                place_id = stoul(score_tokens[i * 2 + 1]);
                score = stod(score_tokens[i * 2 + 2]);

                if (line_score.count(place_id)){
                    line_score[place_id] += weight * score;
                }else{
                    line_score[place_id] = weight * score;
                }
            }       
        }
        
        priority_queue<pair<double, unsigned long> > line_pq;
        for (auto it = line_score.begin(); it != line_score.end(); it++){
            line_pq.push( pair<double, unsigned long>( (*it).second, (*it).first) );
        }

        f_sub << row_id << ',';
        int i = 0;
        while (i < max_sub && !line_pq.empty()){
            f_sub << ' ';
            f_sub << line_pq.top().second;
            line_pq.pop();
            i++;
        }
        f_sub << endl;

    }

    for (int i = 0; i < num_files; i++){
        (*(f_scores[i])).clear();
        (*(f_scores[i])).seekg(0, ios_base::beg);
    }    

}


// generate score and submission
void file_cache_generate_sub(ofstream &f_sub, vector<ifstream*> &f_scores, vector<double> &weights,
        int max_sub, size_t num_lines, size_t cache_size){

    for (int i = 0; i < weights.size(); i++){
        cout << weights[i] << "\t";
    }
    cout << endl;

    int num_files = f_scores.size();
    string line;

    for (int i = 0; i < num_files; i++){
        getline(*(f_scores[i]), line);
    }

    f_sub << "row_id,place_id" << endl;
    

    int i_line = 0;
    while (i_line < num_lines){
        int next_line = i_line + cache_size < num_lines ? i_line + cache_size : num_lines;
        int cur_size = next_line - i_line;

        vector<unordered_map<unsigned long, double> > cache_line_score(cur_size);
        vector<unsigned int> cache_row_id(cur_size, -1);

        // generate the candidate place_ids from the f_scores
        
        for (int i_file = 0; i_file < num_files; i_file++){
            double weight = weights[i_file];
            for (int j_line = 0; j_line < cur_size; j_line++){
                if ( !getline(*(f_scores[i_file]), line) ){
                    cout << "file not matching with score!" << endl;
                    // return 0;
                }
                unordered_map<unsigned long, double> &line_score = cache_line_score[j_line];
                unsigned int &row_id = cache_row_id[j_line];

                vector<string> score_tokens;
                split_line(line, ',', score_tokens);

                unsigned int score_row_id;
                int num_tokens, num_pairs;
                num_tokens = score_tokens.size();
                num_pairs = (score_tokens.size()-1)/2;

                score_row_id = stoi(score_tokens[0]);

                if (row_id == -1){
                    row_id = score_row_id;
                }

                if (score_row_id != row_id){
                    cout << "score row id not matching:" << row_id << ":" << score_row_id << endl;
                }

                for (int i = 0; i < num_pairs; i++){
                    unsigned long place_id;
                    double score;

                    place_id = stoul(score_tokens[i * 2 + 1]);
                    score = stod(score_tokens[i * 2 + 2]);

                    if (line_score.count(place_id)){
                        line_score[place_id] += weight * score;
                    }else{
                        line_score[place_id] = weight * score;
                    }
                }
            }
        }
        
        for (int j_line = 0; j_line < cur_size; j_line++){
            unordered_map<unsigned long, double> &line_score = cache_line_score[j_line];
            unsigned int &row_id = cache_row_id[j_line];

            priority_queue<pair<double, unsigned long> > line_pq;
            for (auto it = line_score.begin(); it != line_score.end(); it++){
                line_pq.push( pair<double, unsigned long>( (*it).second, (*it).first) );
            }

            f_sub << row_id << ',';
            int i = 0;
            while (i < max_sub && !line_pq.empty()){
                f_sub << ' ';
                f_sub << line_pq.top().second;
                line_pq.pop();
                i++;
            }
            f_sub << endl;
        }        
        i_line = next_line;

        cout << next_line << " lines written" << endl;
    }

    for (int i = 0; i < num_files; i++){
        (*(f_scores[i])).clear();
        (*(f_scores[i])).seekg(0, ios_base::beg);
    }    

}


// generate score and compute apk line by line
double file_generate_validate_apk(ifstream &f_valid_test, vector<ifstream*> &f_scores, vector<double> &weights,
        int max_sub, size_t num_lines){

    int num_files = f_scores.size();
    string line;

    getline(f_valid_test, line);
    for (int i = 0; i < num_files; i++){
        getline(*(f_scores[i]), line);
    }
        
    double mapk_score = 0;
    for (int i_line = 0; i_line < num_lines; i_line++){
        // get the true place_id from f_valid_test
        if ( !getline(f_valid_test, line) ){
            cout << "f_valid not matching with score!" << endl;
            return 0;
        }
        vector<string> test_tokens;
        split_line(line, ',', test_tokens);

        unsigned long test_place_id;
        unsigned int test_row_id;

        test_place_id = stoul(test_tokens[5]);
        test_row_id = stoi(test_tokens[0]);

        unordered_set<unsigned long> line_act({test_place_id});

        // generate the candidate place_ids from the f_scores
        unordered_map<unsigned long, double> line_score;

        for (int i_file = 0; i_file < num_files; i_file++){
            if ( !getline(*(f_scores[i_file]), line) ){
                cout << "file not matching with score!" << endl;
                // return 0;
            }
            vector<string> score_tokens;
            split_line(line, ',', score_tokens);

            unsigned int score_row_id;
            int num_tokens, num_pairs;
            num_tokens = score_tokens.size();
            num_pairs = (score_tokens.size()-1)/2;

            score_row_id = stoi(score_tokens[0]);

            if (score_row_id != test_row_id){
                cout << "test and score row id not matching:" << test_row_id << ":" << score_row_id << endl;
            }

            double weight = weights[i_file];
            for (int i = 0; i < num_pairs; i++){
                unsigned long place_id;
                double score;

                place_id = stoul(score_tokens[i * 2 + 1]);
                score = stod(score_tokens[i * 2 + 2]);

                if (line_score.count(place_id)){
                    line_score[place_id] += weight * score;
                }else{
                    line_score[place_id] = weight * score;
                }
            }       
        }
        
        priority_queue<pair<double, unsigned long> > line_pq;
        for (auto it = line_score.begin(); it != line_score.end(); it++){
            line_pq.push( pair<double, unsigned long>( (*it).second, (*it).first) );
        }
        vector<unsigned long> line_pred;
        int i = 0;
        while (i < max_sub && !line_pq.empty()){
            line_pred.push_back(line_pq.top().second);
            line_pq.pop();
            i++;
        }
        double cur_score = apk(line_act, line_pred, max_sub);

        // if (cur_score < 0.3)
        //     cout << line << endl;

        mapk_score += cur_score;
    }
    mapk_score /= num_lines;
    
    f_valid_test.clear();
    f_valid_test.seekg(0, ios_base::beg);

    for (int i = 0; i < num_files; i++){
        (*(f_scores[i])).clear();
        (*(f_scores[i])).seekg(0, ios_base::beg);
    }    

    return mapk_score;
}

// generate score and compute apk line by line into a cache
double file_cache_generate_validate_apk(ifstream &f_valid_test, vector<ifstream*> &f_scores, vector<double> &weights,
        int max_sub, size_t num_lines, size_t cache_size){

    int num_files = f_scores.size();
    string line;

    getline(f_valid_test, line);
    for (int i = 0; i < num_files; i++){
        getline(*(f_scores[i]), line);
    }
        
    double mapk_score = 0;

    int i_line = 0;
    while (i_line < num_lines){
        int next_line = i_line + cache_size < num_lines ? i_line + cache_size : num_lines;
        int cur_size = next_line - i_line;

        vector<unordered_set<unsigned long> > cache_line_act(cur_size);
        vector<unordered_map<unsigned long, double> > cache_line_score(cur_size);
        vector<unsigned int> cache_row_id(cur_size);

        for (int j_line = 0; j_line < cur_size; j_line++){
            // get the true place_id from f_valid_test
            if ( !getline(f_valid_test, line) ){
                cout << "f_valid not matching with score!" << endl;
                return 0;
            }
            vector<string> test_tokens;
            split_line(line, ',', test_tokens);

            unsigned long test_place_id;
            unsigned int test_row_id;

            test_place_id = stoul(test_tokens[5]);
            test_row_id = stoi(test_tokens[0]);

            cache_line_act[j_line] = unordered_set<unsigned long>({test_place_id});
            cache_row_id[j_line] = test_row_id;
        }

        // generate the candidate place_ids from the f_scores
        for (int i_file = 0; i_file < num_files; i_file++){
            for (int j_line = 0; j_line < cur_size; j_line++){
                if ( !getline(*(f_scores[i_file]), line) ){
                    cout << "file not matching with score!" << endl;
                    // return 0;
                }
                vector<string> score_tokens;
                split_line(line, ',', score_tokens);

                unsigned int score_row_id;
                int num_tokens, num_pairs;
                num_tokens = score_tokens.size();
                num_pairs = (score_tokens.size()-1)/2;

                score_row_id = stoi(score_tokens[0]);

                unsigned int test_row_id = cache_row_id[j_line];
                if (score_row_id != test_row_id){
                    cout << "test and score row id not matching:" << test_row_id << ":" << score_row_id << endl;
                }

                unordered_map<unsigned long, double> &line_score = cache_line_score[j_line];
                double weight = weights[i_file];
                for (int i = 0; i < num_pairs; i++){
                    unsigned long place_id;
                    double score;

                    place_id = stoul(score_tokens[i * 2 + 1]);
                    score = stod(score_tokens[i * 2 + 2]);

                    if (line_score.count(place_id)){
                        line_score[place_id] += weight * score;
                    }else{
                        line_score[place_id] = weight * score;
                    }
                }
            }   
        }

        for (int j_line = 0; j_line < cur_size; j_line++){  
            unordered_map<unsigned long, double> &line_score = cache_line_score[j_line];
            unordered_set<unsigned long> &line_act = cache_line_act[j_line];

            priority_queue<pair<double, unsigned long> > line_pq;
            for (auto it = line_score.begin(); it != line_score.end(); it++){
                line_pq.push( pair<double, unsigned long>( (*it).second, (*it).first) );
            }
            vector<unsigned long> line_pred;
            int i = 0;
            while (i < max_sub && !line_pq.empty()){
                line_pred.push_back(line_pq.top().second);
                line_pq.pop();
                i++;
            }
            double cur_score = apk(line_act, line_pred, max_sub);

            // if (cur_score < 0.3)
            //     cout << line << endl;

            mapk_score += cur_score;
        }

        i_line = next_line;

        // print partial result
        cout << next_line << " lines validated: " << (mapk_score/next_line) << endl;

    }
    mapk_score /= num_lines;
    
    f_valid_test.clear();
    f_valid_test.seekg(0, ios_base::beg);

    for (int i = 0; i < num_files; i++){
        (*(f_scores[i])).clear();
        (*(f_scores[i])).seekg(0, ios_base::beg);
    }    

    return mapk_score;
}


// generate score and compute apk line by line into a cache with log add last one contains all the target
double file_cache_generate_validate_apk_log(ifstream &f_valid_test, vector<ifstream*> &f_scores, vector<double> &weights,
        int max_sub, size_t num_lines, size_t cache_size){

    int num_files = f_scores.size();
    int num_add_files = f_scores.size() - 1;
    double add_weights_sum = 0.0;

    for (int i = 0; i < num_add_files; i++){
        add_weights_sum += weights[i];
    }

    string line;

    getline(f_valid_test, line);
    for (int i = 0; i < num_files; i++){
        getline(*(f_scores[i]), line);
    }
        
    double mapk_score = 0;

    int i_line = 0;
    while (i_line < num_lines){
        int next_line = i_line + cache_size < num_lines ? i_line + cache_size : num_lines;
        int cur_size = next_line - i_line;

        vector<unordered_set<unsigned long> > cache_line_act(cur_size);
        vector<unordered_map<unsigned long, double> > cache_line_score(cur_size);
        vector<unordered_map<unsigned long, double> > cache_line_score_final(cur_size);

        vector<unsigned int> cache_row_id(cur_size);

        for (int j_line = 0; j_line < cur_size; j_line++){
            // get the true place_id from f_valid_test
            if ( !getline(f_valid_test, line) ){
                cout << "f_valid not matching with score!" << endl;
                return 0;
            }
            vector<string> test_tokens;
            split_line(line, ',', test_tokens);

            unsigned long test_place_id;
            unsigned int test_row_id;

            test_place_id = stoul(test_tokens[5]);
            test_row_id = stoi(test_tokens[0]);

            cache_line_act[j_line] = unordered_set<unsigned long>({test_place_id});
            cache_row_id[j_line] = test_row_id;
        }

        // generate the candidate place_ids from the f_scores
        for (int i_file = 0; i_file < num_add_files; i_file++){
            for (int j_line = 0; j_line < cur_size; j_line++){
                if ( !getline(*(f_scores[i_file]), line) ){
                    cout << "file not matching with score!" << endl;
                    // return 0;
                }
                vector<string> score_tokens;
                split_line(line, ',', score_tokens);

                unsigned int score_row_id;
                int num_tokens, num_pairs;
                num_tokens = score_tokens.size();
                num_pairs = (score_tokens.size()-1)/2;

                score_row_id = stoi(score_tokens[0]);

                unsigned int test_row_id = cache_row_id[j_line];
                if (score_row_id != test_row_id){
                    cout << "test and score row id not matching:" << test_row_id << ":" << score_row_id << endl;
                }

                unordered_map<unsigned long, double> &line_score = cache_line_score[j_line];
                double weight = weights[i_file];
                for (int i = 0; i < num_pairs; i++){
                    unsigned long place_id;
                    double score;

                    place_id = stoul(score_tokens[i * 2 + 1]);
                    score = stod(score_tokens[i * 2 + 2]);

                    if (line_score.count(place_id)){
                        line_score[place_id] += weight * score;
                    }else{
                        line_score[place_id] = weight * score;
                    }
                }
            }   
        }

        // last item using multiply
        int last_i_file = num_add_files;

        for (int j_line = 0; j_line < cur_size; j_line++){
            if ( !getline(*(f_scores[last_i_file]), line) ){
                cout << "file not matching with score!" << endl;
                // return 0;
            }
            vector<string> score_tokens;
            split_line(line, ',', score_tokens);

            unsigned int score_row_id;
            int num_tokens, num_pairs;
            num_tokens = score_tokens.size();
            num_pairs = (score_tokens.size()-1)/2;

            score_row_id = stoi(score_tokens[0]);

            unsigned int test_row_id = cache_row_id[j_line];
            if (score_row_id != test_row_id){
                cout << "test and score row id not matching:" << test_row_id << ":" << score_row_id << endl;
            }

            unordered_map<unsigned long, double> &line_score = cache_line_score[j_line];
            unordered_map<unsigned long, double> &line_score_final = cache_line_score_final[j_line];
            double weight = weights[last_i_file];
            for (int i = 0; i < num_pairs; i++){
                unsigned long place_id;
                double score;

                place_id = stoul(score_tokens[i * 2 + 1]);
                score = stod(score_tokens[i * 2 + 2]);

                if (line_score.count(place_id)){
                    line_score_final[place_id] = weight * log(score) + log(line_score[place_id] / add_weights_sum);
                }else{
                    line_score_final[place_id] = weight * log(score) + log(0.001);
                }
            }
        }   

        for (int j_line = 0; j_line < cur_size; j_line++){  
            unordered_map<unsigned long, double> &line_score_final = cache_line_score_final[j_line];
            unordered_set<unsigned long> &line_act = cache_line_act[j_line];

            priority_queue<pair<double, unsigned long> > line_pq;
            for (auto it = line_score_final.begin(); it != line_score_final.end(); it++){
                line_pq.push( pair<double, unsigned long>( (*it).second, (*it).first) );
            }
            vector<unsigned long> line_pred;
            int i = 0;
            while (i < max_sub && !line_pq.empty()){
                line_pred.push_back(line_pq.top().second);
                line_pq.pop();
                i++;
            }
            double cur_score = apk(line_act, line_pred, max_sub);

            // if (cur_score < 0.3)
            //     cout << line << endl;

            mapk_score += cur_score;
        }

        i_line = next_line;

        // print partial result
        cout << next_line << " lines validated: " << (mapk_score/next_line) << endl;

    }
    mapk_score /= num_lines;
    
    f_valid_test.clear();
    f_valid_test.seekg(0, ios_base::beg);

    for (int i = 0; i < num_files; i++){
        (*(f_scores[i])).clear();
        (*(f_scores[i])).seekg(0, ios_base::beg);
    }    

    return mapk_score;
}


