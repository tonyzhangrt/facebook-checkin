#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <bitset>
#include <ctime>
#include <queue>
#include <istream>
#include <algorithm>
using namespace std;

string get_timestamp(){
    char buff[15];
    time_t rawtime;
    struct tm * timeinfo;

    time (&rawtime);
    timeinfo = localtime (&rawtime);

    sprintf(buff, "%4d%02d%02d%02d%02d%02d", 1900 + timeinfo->tm_year, 1+timeinfo->tm_mon, 
        timeinfo->tm_mday, timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);

    string timestamp(buff);
    return timestamp;
}


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

    // cout << "act:" << *(actual.begin()) << '\t';
    // for (int i = 0; i < num_pred; i++){
    //     cout << predicted[i] << '\t';
    // }
    // cout << score << endl;

    return score;
}

double rect_intersect_global(double x_min, double x_max, double y_min, double y_max){
    x_min = x_min > 0 ? x_min : 0;
    x_max = x_max < 10 ? x_max : 10;
    y_min = y_min > 0 ? y_min : 0;
    y_max = y_max < 10 ? y_max : 10;

    return (x_max - x_min) * (y_max - y_min);
}

// 
double rect_intersect_local(const double x_min, const double x_max, const double y_min, const double y_max, 
    const unsigned int ix, const unsigned int iy, const unsigned int range_x, const unsigned int range_y,
    vector<double> (*anchor_func)(const unsigned int, const unsigned int, const unsigned int, const unsigned int)){
    double local_x_min, local_x_max, local_y_min, local_y_max;
    vector<double> anchor = anchor_func(ix, iy, range_x, range_y);

    local_x_min = anchor[0];
    local_x_max = anchor[1];
    local_y_min = anchor[2];
    local_y_max = anchor[3];

    double target_x_min, target_x_max, target_y_min, target_y_max;
    target_x_min = x_min > local_x_min ? x_min : local_x_min;
    target_x_max = x_max < local_x_max ? x_max : local_x_max;
    target_y_min = y_min > local_y_min ? y_min : local_y_min;
    target_y_max = y_max < local_y_max ? y_max : local_y_max;

    double x_span, y_span;
    x_span = target_x_max > target_x_min ? target_x_max - target_x_min : 0;
    y_span = target_y_max > target_y_min ? target_y_max - target_y_min : 0;

    return x_span * y_span;
}

// assuming rms accuracy
double gaussain_prob_local(const double x, const double x_span, const double y, const double y_span,  
    const unsigned int ix, const unsigned int iy, const unsigned range_x, const unsigned range_y,
    vector<double> (*anchor_func)(const unsigned int, const unsigned int, const unsigned int, const unsigned int)){
    double local_x_min, local_x_max, local_y_min, local_y_max;
    vector<double> anchor = anchor_func(ix, iy, range_x, range_y);

    local_x_min = anchor[0];
    local_x_max = anchor[1];
    local_y_min = anchor[2];
    local_y_max = anchor[3];

    double x_lower, x_upper, y_lower, y_upper;

    // suppose rms R = sqrt(2) * sigma
    x_lower = (local_x_min - x) / x_span;
    x_upper = (local_x_max - x) / x_span;
    y_lower = (local_y_min - y) / y_span;
    y_upper = (local_y_max - y) / y_span;

    return 0.25 * (erf(x_upper) - erf(x_lower)) * (erf(y_upper) - erf(y_lower));
}


vector<double> anchor_xy(const unsigned int ix, const unsigned int iy, const unsigned int range_x, const unsigned int range_y){
    double x_min, x_max, y_min, y_max;

    x_min = double(ix) / range_x * 10;
    x_max = (double(ix) + 1) / range_x * 10;
    y_min = double(iy) / range_y * 10;
    y_max = (double(iy) + 1) / range_y * 10;

    return vector<double>({x_min, x_max, y_min, y_max});
}

vector<double> anchor_xy_xshift(const unsigned int ix, const unsigned int iy, const unsigned int range_x, const unsigned int range_y){
    double x_min, x_max, y_min, y_max;

    x_min = (double(ix) + 0.5)/ range_x * 10;
    x_max = (double(ix) + 1.5) / range_x * 10;

    x_min = ix > 0 ? x_min : 0;
    x_max = ix < range_x - 2 ? x_max : 10;
    
    y_min = double(iy) / range_y * 10;
    y_max = (double(iy) + 1) / range_y * 10;

    return vector<double>({x_min, x_max, y_min, y_max});
}

vector<double> anchor_xy_yshift(const unsigned int ix, const unsigned int iy, const unsigned int range_x, const unsigned int range_y){
    double x_min, x_max, y_min, y_max;

    x_min = double(ix)/ range_x * 10;
    x_max = (double(ix) + 1) / range_x * 10;

    y_min = (double(iy) + 0.5) / range_y * 10;
    y_max = (double(iy) + 1.5) / range_y * 10;

    y_min = iy > 0 ? y_min : 0;
    y_max = iy < range_y - 2 ? y_max : 10;

    return vector<double>({x_min, x_max, y_min, y_max});
}

vector<double> anchor_xy_xyshift(const unsigned int ix, const unsigned int iy, const unsigned int range_x, const unsigned int range_y){
    double x_min, x_max, y_min, y_max;

    x_min = (double(ix) + 0.5)/ range_x * 10;
    x_max = (double(ix) + 1.5) / range_x * 10;

    x_min = ix > 0 ? x_min : 0;
    x_max = ix < range_x - 2 ? x_max : 10;

    y_min = (double(iy) + 0.5) / range_y * 10;
    y_max = (double(iy) + 1.5) / range_y * 10;

    y_min = iy > 0 ? y_min : 0;
    y_max = iy < range_y - 2 ? y_max : 10;

    return vector<double>({x_min, x_max, y_min, y_max});
}


pair<unsigned int, unsigned int> prep_xy(const double x, const double y, const unsigned int range_x, const unsigned int range_y){
    unsigned int ix, iy;

    ix = int (range_x * x / 10);
    iy = int (range_y * y / 10);

    ix = ix >= 0 ? ix : 0;
    ix = ix < range_x ? ix : range_x - 1;
    iy = iy >= 0 ? iy : 0;
    iy = iy < range_y ? iy : range_y - 1;

    return pair<unsigned int, unsigned int>(ix, iy);
}

pair<unsigned int, unsigned int> prep_xy_xshift(const double x, const double y, const unsigned int range_x, const unsigned int range_y){
    unsigned int ix, iy;

    ix = int ((range_x * x / 10) - 0.5) ;
    iy = int (range_y * y / 10);

    ix = ix >= 0 ? ix : 0;
    ix = ix < range_x - 1 ? ix : range_x - 2;
    iy = iy >= 0 ? iy : 0;
    iy = iy < range_y ? iy : range_y - 1;

    return pair<unsigned int, unsigned int>(ix, iy);
}

pair<unsigned int, unsigned int> prep_xy_yshift(const double x, const double y, const unsigned int range_x, const unsigned int range_y){
    unsigned int ix, iy;

    ix = int ((range_x * x / 10)) ;
    iy = int ((range_y * y / 10) - 0.5);

    ix = ix >= 0 ? ix : 0;
    ix = ix < range_x ? ix : range_x - 1;
    iy = iy >= 0 ? iy : 0;
    iy = iy < range_y - 1 ? iy : range_y - 2;

    return pair<unsigned int, unsigned int>(ix, iy);
}

pair<unsigned int, unsigned int> prep_xy_xyshift(const double x, const double y, const unsigned int range_x, const unsigned int range_y){
    unsigned int ix, iy;

    ix = int ((range_x * x / 10) - 0.5) ;
    iy = int ((range_y * y / 10) - 0.5);

    ix = ix >= 0 ? ix : 0;
    ix = ix < range_x - 1 ? ix : range_x - 2;
    iy = iy >= 0 ? iy : 0;
    iy = iy < range_y - 1 ? iy : range_y - 2;

    return pair<unsigned int, unsigned int>(ix, iy);
}

void split_line(const string &line, char delim, vector<string> &result){
    istringstream iss(line);
    string word;
    while (getline(iss, word, delim)){
        result.push_back(word);
    }
}

unsigned int nulltime(unsigned int cur_time){
    return 0;
}

unsigned int itime(unsigned int cur_time){
    return ((cur_time +120) % (24 * 60)) / 360;
}

unsigned int halfdayround0(unsigned int cur_time){
    return (cur_time % (12 * 60)) / 30;
}

unsigned int halfdayround(unsigned int cur_time){
    return (cur_time % (12 * 60)) / 60;
}

unsigned int halfdayround2(unsigned int cur_time){
    return (cur_time % (12 * 60)) / 120;
}

unsigned int dayround0(unsigned int cur_time){
    return (cur_time % (24 * 60 )) / 30;
}

unsigned int dayround(unsigned int cur_time){
    return (cur_time % (24 * 60 )) / 60;
}

unsigned int dayround2(unsigned int cur_time){
    return (cur_time % (24 * 60 )) / 120;
}

unsigned int dayround3(unsigned int cur_time){
    return (cur_time % (24 * 60 )) / 180;
}

unsigned int dayround4(unsigned int cur_time){
    return (cur_time % (24 * 60 )) / 240;
}

unsigned int dayround8(unsigned int cur_time){
    return (cur_time % (24 * 60 )) / 480;
}

unsigned int dayroundhalfday(unsigned int cur_time){
    return (cur_time % (24 * 60 )) / 720;
}

unsigned int weekround0(unsigned int cur_time){
    return (cur_time % (24 * 60 * 7)) / 60;
}

unsigned int weekround(unsigned int cur_time){
    return (cur_time % (24 * 60 * 7)) / 120;
}

unsigned int weekround1(unsigned int cur_time){
    return (cur_time % (24 * 60 * 7)) / 180;
}

unsigned int weekround2(unsigned int cur_time){
    return (cur_time % (24 * 60 * 7)) / 240;
}

unsigned int weekround4(unsigned int cur_time){
    return (cur_time % (24 * 60 * 7)) / 480;
}

unsigned int weekquaterday(unsigned int cur_time){
    return (cur_time % (24 * 60 * 7)) / 360;
}

unsigned int weekhalfday(unsigned int cur_time){
    return (cur_time % (24 * 60 * 7)) / 720;
}

unsigned int weekday(unsigned int cur_time){
    return (cur_time % (24 * 60 * 7)) / 1440;
}

unsigned int yearroundday(unsigned int cur_time){
    return (cur_time % (24 * 60 * 7 * 52)) / 1440;
}

unsigned int yearroundweek(unsigned int cur_time){
    return (cur_time % (24 * 60 * 7 * 52)) / (1440*7);
}

unsigned int weekdayshift(unsigned int cur_time){
    return ((cur_time+720) % (24 * 60 * 7)) / 1440;
}

unsigned int weekcount(unsigned int cur_time){
    return cur_time / (24 * 60 * 7);
}

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

// void reorder_score(ifstream &f_input, ofstream &f_output){
//     string header, line;

//     getline(f_input, header);
//     unordered_map<unsigned int, string> line_map;
//     vector<unsigned int> row_ids;

//     while ( getline(f_input, line) ){
//         vector<string> tokens;
//         split_line(line, ',', tokens);

//         unsigned int row_id;

//         row_id = stoi(tokens[0]);

//         row_ids.push_back(row_id);
//         line_map[row_id] = line;
//     }
//     f_input.clear();
//     f_input.seekg(0, ios_base::beg);

//     sort(row_ids.begin(), row_ids.end());

//     unsigned int num_lines = line_map.size();
//     cout << "reorder_score: " << num_lines << endl;

//     f_output << header << endl;
//     for (int i = 0; i < num_lines; i++){
//         unsigned int row_id = row_ids[i];
//         f_output << line_map[row_id] << endl;
//     }
// }


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
void train_test_split_by_grid_weight_by_acc(const unsigned int x_range, const unsigned int y_range, double acc_ratio, double aug_ratio, ifstream &ftrain_input, ifstream &ftest_input, ofstream &f_output, 
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
void train_test_split_by_grid_weight_gaussian(const unsigned int x_range, const unsigned int y_range, double acc_ratio, double aug_ratio, ifstream &ftrain_input, ifstream &ftest_input, ofstream &f_output, 
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


void run_validation(){
    int max_rank = 8, max_sub = 3;
    double weight = 1;
    size_t num_valid_train = 23318975, num_valid_test = 5799046;
    double mapk_score = 0;

    ifstream f_valid_train("../valid/valid_train.csv");
    ifstream f_valid_test("../valid/valid_test.csv");

    ofstream f_valid_score("../valid/basic_valid_score.csv");

    if (!f_valid_train.is_open() || !f_valid_test.is_open() ) return;

    unordered_map<unsigned int, unordered_map<unsigned long, unsigned int> > grid;
    unordered_map<unsigned int, vector<unsigned long> > grid_top;
    vector<unordered_map<unsigned long, double> > score(num_valid_test);

    const clock_t begin_time = clock();
    
    generate_grid(500, 1000, f_valid_train, itime, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(500, 1000, f_valid_test, max_rank, 2 * weight, itime, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "itime done" << endl;

    generate_grid(150, 300, f_valid_train, dayround0, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(150, 300, f_valid_test, max_rank, weight, dayround0, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "dayround0 done" << endl;

    generate_grid(251, 501, f_valid_train, dayround, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(251, 501, f_valid_test, max_rank, weight, dayround, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "dayround done" << endl;

    generate_grid(400, 800, f_valid_train, dayround2, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(400, 800, f_valid_test, max_rank, weight, dayround2, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "dayround2 done" << endl;

    generate_grid(425, 849, f_valid_train, dayround3, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(425, 849, f_valid_test, max_rank, weight, dayround3, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "dayround3 done" << endl;

    generate_grid(456, 912, f_valid_train, dayround4, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(456, 912, f_valid_test, max_rank, weight, dayround4, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "dayround4 done" << endl;

    generate_grid(476, 951, f_valid_train, dayround8, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(476, 951, f_valid_test, max_rank, weight, dayround8, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "dayround8 done" << endl;

    generate_grid(491, 980, f_valid_train, dayroundhalfday, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(491, 980, f_valid_test, max_rank, weight, dayroundhalfday, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "dayroundhalfday done" << endl;

    generate_grid(101, 201, f_valid_train, weekround0, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(101, 201, f_valid_test, max_rank, weight, weekround0, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "weekround0 done" << endl; 

    generate_grid(150, 300, f_valid_train, weekround, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(150, 300, f_valid_test, max_rank, weight, weekround, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "weekround done" << endl; 

    generate_grid(187, 374, f_valid_train, weekround1, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(187, 374, f_valid_test, max_rank, weight, weekround1, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "weekround1 done" << endl;    

    generate_grid(201, 401, f_valid_train, weekround2, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(201, 401, f_valid_test, max_rank, weight, weekround2, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "weekround2 done" << endl;

    generate_grid(250, 500, f_valid_train, weekround4, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(250, 500, f_valid_test, max_rank, weight, weekround4, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "weekround4 done" << endl;

    generate_grid(225, 451, f_valid_train, weekquaterday, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(225, 451, f_valid_test, max_rank, weight, weekquaterday, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "weekquaterday done" << endl;

    generate_grid(301, 602, f_valid_train, weekhalfday, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(301, 602, f_valid_test, max_rank, weight, weekhalfday, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "weekhalfday done" << endl;    

    generate_grid(450, 900, f_valid_train, weekday, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(450, 900, f_valid_test, max_rank, weight, weekday, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "weekday done" << endl;

    generate_grid(501, 1001, f_valid_train, nulltime, grid, prep_xy);

    generate_top(max_rank, grid_top, grid);

    update_score(501, 1001, f_valid_test, max_rank, weight, nulltime, score, grid_top);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "nulltime done" << endl; 

    /* 1 : 2.5 version */

    // generate_grid(447, 1118, f_valid_train, itime, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(447, 1118, f_valid_test, max_rank, 2 * weight, itime, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "itime done" << endl;

    // generate_grid(134, 335, f_valid_train, dayround0, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(134, 335, f_valid_test, max_rank, weight, dayround0, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "dayround0 done" << endl;

    // generate_grid(225, 559, f_valid_train, dayround, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(225, 559, f_valid_test, max_rank, weight, dayround, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "dayround done" << endl;

    // generate_grid(357, 894, f_valid_train, dayround2, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(357, 894, f_valid_test, max_rank, weight, dayround2, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "dayround2 done" << endl;

    // generate_grid(380, 950, f_valid_train, dayround3, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(380, 950, f_valid_test, max_rank, weight, dayround3, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "dayround3 done" << endl;

    // generate_grid(407, 1019, f_valid_train, dayround4, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(407, 1019, f_valid_test, max_rank, weight, dayround4, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "dayround4 done" << endl;

    // generate_grid(425, 1062, f_valid_train, dayround8, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(425, 1062, f_valid_test, max_rank, weight, dayround8, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "dayround8 done" << endl;

    // generate_grid(438, 1095, f_valid_train, dayroundhalfday, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(438, 1095, f_valid_test, max_rank, weight, dayroundhalfday, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "dayroundhalfday done" << endl;

    // generate_grid(90, 223, f_valid_train, weekround0, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(90, 223, f_valid_test, max_rank, weight, weekround0, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekround0 done" << endl; 

    // generate_grid(133, 336, f_valid_train, weekround, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(133, 336, f_valid_test, max_rank, weight, weekround, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekround done" << endl; 

    // generate_grid(168, 419, f_valid_train, weekround1, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(168, 419, f_valid_test, max_rank, weight, weekround1, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekround1 done" << endl; 

    // generate_grid(178, 447, f_valid_train, weekround2, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(178, 447, f_valid_test, max_rank, weight, weekround2, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekround2 done" << endl;

    // generate_grid(223, 560, f_valid_train, weekround4, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(223, 560, f_valid_test, max_rank, weight, weekround4, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekround4 done" << endl;

    // generate_grid(201, 503, f_valid_train, weekquaterday, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(201, 503, f_valid_test, max_rank, weight, weekquaterday, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekquaterday done" << endl;

    // generate_grid(268, 671, f_valid_train, weekhalfday, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(268, 671, f_valid_test, max_rank, weight, weekhalfday, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekhalfday done" << endl;    

    // generate_grid(403, 1006, f_valid_train, weekday, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(403, 1006, f_valid_test, max_rank, weight, weekday, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekday done" << endl; 

    // generate_grid(448, 1119, f_valid_train, nulltime, grid, prep_xy);

    // generate_top(max_rank, grid_top, grid);

    // update_score(448, 1119, f_valid_test, max_rank, weight, nulltime, score, grid_top);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "nulltime done" << endl; 


    // generate_valid_score(f_valid_score, 20, score);

    // cout << "validation sub generated" << endl;

    cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

    f_valid_score.close();
    f_valid_train.close();
    f_valid_test.close();
}

void run_validation_ratio(){
    int max_rank = 15, max_sub = 3;
    double weight = 1;
    size_t num_valid_train = 23318975, num_valid_test = 5799046;
    double mapk_score = 0;

    ifstream f_valid_train("../valid/valid_train.csv");
    ifstream f_valid_test("../valid/valid_test.csv");

    ofstream f_valid_score("../valid/basic_valid_score.csv");

    if (!f_valid_train.is_open() || !f_valid_test.is_open() ) return;

    unordered_map<unsigned int, unordered_map<unsigned long, unsigned int> > grid;
    unordered_map<unsigned int, vector<pair<unsigned long, double> > > grid_top_ratio;
    vector<unordered_map<unsigned long, double> > score(num_valid_test);

    const clock_t begin_time = clock();
    
    generate_grid(500, 1000, f_valid_train, itime, grid, prep_xy);

    generate_top_ratio(max_rank, grid_top_ratio, grid);

    update_score_ratio(500, 1000, f_valid_test, max_rank, 2 * weight, itime, score, grid_top_ratio);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "itime done" << endl;

    // generate_grid(500, 1000, f_valid_train, itime, grid, prep_xy_xshift);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(500, 1000, f_valid_test, max_rank, 2 * weight, itime, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "itime xshift done" << endl;

    // generate_grid(500, 1000, f_valid_train, itime, grid, prep_xy_yshift);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(500, 1000, f_valid_test, max_rank, 2 * weight, itime, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "itime yshift done" << endl;

    generate_grid(500, 1000, f_valid_train, itime, grid, prep_xy_xyshift);

    generate_top_ratio(max_rank, grid_top_ratio, grid);

    update_score_ratio(500, 1000, f_valid_test, max_rank, 2 * weight, itime, score, grid_top_ratio);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "itime xyshift done" << endl;

    generate_grid(151, 299, f_valid_train, dayround0, grid, prep_xy);

    generate_top_ratio(max_rank, grid_top_ratio, grid);

    update_score_ratio(151, 299, f_valid_test, max_rank, weight, dayround0, score, grid_top_ratio);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "dayround0 done" << endl;

    // generate_grid(151, 299, f_valid_train, dayround0, grid, prep_xy_xshift);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(151, 299, f_valid_test, max_rank, weight, dayround0, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "dayround0 xshift done" << endl;

    // generate_grid(151, 299, f_valid_train, dayround0, grid, prep_xy_yshift);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(151, 299, f_valid_test, max_rank, weight, dayround0, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "dayround0 yshift done" << endl;

    generate_grid(151, 299, f_valid_train, dayround0, grid, prep_xy_xyshift);

    generate_top_ratio(max_rank, grid_top_ratio, grid);

    update_score_ratio(151, 299, f_valid_test, max_rank, weight, dayround0, score, grid_top_ratio);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    cout << "dayround0 xyshift done" << endl;

    // generate_grid(250, 500, f_valid_train, dayround, grid);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(250, 500, f_valid_test, max_rank, weight, dayround, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "dayround done" << endl;

    // generate_grid(400, 800, f_valid_train, dayround2, grid);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(400, 800, f_valid_test, max_rank, weight, dayround2, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "dayround2 done" << endl;

    // generate_grid(100, 200, f_valid_train, weekround0, grid);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(100, 200, f_valid_test, max_rank, weight, weekround0, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekround0 done" << endl; 

    // generate_grid(150, 300, f_valid_train, weekround, grid);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(150, 300, f_valid_test, max_rank, weight, weekround, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekround done" << endl; 

    // generate_grid(200, 400, f_valid_train, weekround2, grid);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(200, 400, f_valid_test, max_rank, weight, weekround2, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekround2 done" << endl;

    // generate_grid(250, 500, f_valid_train, weekquaterday, grid);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(250, 500, f_valid_test, max_rank, weight, weekquaterday, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekquaterday done" << endl;

    // generate_grid(300, 600, f_valid_train, weekhalfday, grid);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(300, 600, f_valid_test, max_rank, weight, weekhalfday, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekhalfday done" << endl;    

    // generate_grid(450, 900, f_valid_train, weekday, grid);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(450, 900, f_valid_test, max_rank, weight, weekday, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "weekday done" << endl; 

    // generate_grid(500, 1000, f_valid_train, nulltime, grid);

    // generate_top_ratio(max_rank, grid_top_ratio, grid);

    // update_score_ratio(500, 1000, f_valid_test, max_rank, weight, nulltime, score, grid_top_ratio);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << "nulltimedone" << endl; 

    // generate_valid_score(f_valid_score, 20, score);

    // cout << "validation sub generated" << endl;

    cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

    f_valid_score.close();
    f_valid_train.close();
    f_valid_test.close();
}

void reorder_validation(){
    int max_rank = 8, max_sub = 3;
    size_t num_valid_train = 23318975, num_valid_test = 5799046;
    double mapk_score = 0, weight = 1;
    unsigned int limit = 50;

    // ifstream f_valid_train("../valid/valid_train.csv");
    // ifstream f_valid_test("../valid/valid_test.csv");

    // ifstream f_valid_score_71("../valid/valid_score_71_20160518180135.csv");
    // ifstream f_valid_score_100("../valid/valid_score_20160518025233.csv");
    // ifstream f_valid_score_100xs("../valid/valid_score_100xs_test.csv");
    // ifstream f_valid_score_100ys("../valid/valid_score_100ys_20160519162412.csv");
    // ifstream f_valid_score_100xys("../valid/valid_score_100xys_20160519162933.csv");

    /* reorder row_id */
    // ifstream f_valid_score_100("../valid/valid_score_100_20160605000504.csv");
    // ifstream f_valid_score_100xs("../valid/valid_score_100xs_20160606172637.csv");
    // ifstream f_valid_score_100ys("../valid/valid_score_100ys_20160606172711.csv");
    // ifstream f_valid_score_100xys("../valid/valid_score_100xys_20160607130744.csv");

    // ofstream f_valid_score_100_reorder("../valid/reorder_valid_score_100_20160605000504.csv");
    // ofstream f_valid_score_100xs_reorder("../valid/reorder_valid_score_100xs_20160606172637.csv");
    // ofstream f_valid_score_100ys_reorder("../valid/reorder_valid_score_100ys_20160606172711.csv");
    // ofstream f_valid_score_100xys_reorder("../valid/reorder_valid_score_100xys_20160607130744.csv");

    // reorder_score(f_valid_score_100, f_valid_score_100_reorder);
    // reorder_score(f_valid_score_100xs, f_valid_score_100xs_reorder);
    // reorder_score(f_valid_score_100ys, f_valid_score_100ys_reorder);
    // reorder_score(f_valid_score_100xys, f_valid_score_100xys_reorder);

    // f_valid_score_100.close();
    // f_valid_score_100xs.close();
    // f_valid_score_100ys.close();
    // f_valid_score_100xys.close();
    // f_valid_score_100_reorder.close();
    // f_valid_score_100xs_reorder.close();
    // f_valid_score_100ys_reorder.close();
    // f_valid_score_100xys_reorder.close();

    // ifstream f_valid_score_71("../valid/valid_score_71_20160605004036.csv");
    // ifstream f_valid_score_71xs("../valid/valid_score_71xs_20160607234123.csv");
    // ifstream f_valid_score_71ys("../valid/valid_score_71ys_20160607234206.csv");
    // ifstream f_valid_score_71xys("../valid/valid_score_71xys_20160607234405.csv");

    // ofstream f_valid_score_71_reorder("../valid/reorder_valid_score_71_20160605004036.csv");
    // ofstream f_valid_score_71xs_reorder("../valid/reorder_valid_score_71xs_20160607234123.csv");
    // ofstream f_valid_score_71ys_reorder("../valid/reorder_valid_score_71ys_20160607234206.csv");
    // ofstream f_valid_score_71xys_reorder("../valid/reorder_valid_score_71xys_20160607234405.csv");

    // reorder_score(f_valid_score_71, f_valid_score_71_reorder);
    // reorder_score(f_valid_score_71xs, f_valid_score_71xs_reorder);
    // reorder_score(f_valid_score_71ys, f_valid_score_71ys_reorder);
    // reorder_score(f_valid_score_71xys, f_valid_score_71xys_reorder);

    // f_valid_score_71.close();
    // f_valid_score_71xs.close();
    // f_valid_score_71ys.close();
    // f_valid_score_71xys.close();
    // f_valid_score_71_reorder.close();
    // f_valid_score_71xs_reorder.close();
    // f_valid_score_71ys_reorder.close();
    // f_valid_score_71xys_reorder.close();

    // ifstream f_valid_score_90("../valid/valid_score_90_20160608142409.csv");
    // ifstream f_valid_score_90xs("../valid/valid_score_90xs_20160608143143.csv");
    // ifstream f_valid_score_90ys("../valid/valid_score_90ys_20160609202512.csv");
    // ifstream f_valid_score_90xys("../valid/valid_score_90xys_20160609202658.csv");

    // ofstream f_valid_score_90_reorder("../valid/reorder_valid_score_90_20160608142409.csv");
    // ofstream f_valid_score_90xs_reorder("../valid/reorder_valid_score_90xs_20160608143143.csv");
    // ofstream f_valid_score_90ys_reorder("../valid/reorder_valid_score_90ys_20160609202512.csv");
    // ofstream f_valid_score_90xys_reorder("../valid/reorder_valid_score_90xys_20160609202658.csv");

    // reorder_score(f_valid_score_90, f_valid_score_90_reorder);
    // reorder_score(f_valid_score_90xs, f_valid_score_90xs_reorder);
    // reorder_score(f_valid_score_90ys, f_valid_score_90ys_reorder);
    // reorder_score(f_valid_score_90xys, f_valid_score_90xys_reorder);

    // f_valid_score_90.close();
    // f_valid_score_90xs.close();
    // f_valid_score_90ys.close();
    // f_valid_score_90xys.close();
    // f_valid_score_90_reorder.close();
    // f_valid_score_90xs_reorder.close();
    // f_valid_score_90ys_reorder.close();
    // f_valid_score_90xys_reorder.close();

    // ifstream f_valid_score_63("../valid/valid_score_63_20160610165455.csv");
    // ifstream f_valid_score_63xs("../valid/valid_score_63xs_20160610165529.csv");
    // ifstream f_valid_score_63ys("../valid/valid_score_63ys_20160610165711.csv");
    // ifstream f_valid_score_63xys("../valid/valid_score_63xys_20160610171448.csv");

    // ofstream f_valid_score_63_reorder("../valid/reorder_valid_score_63_20160610165455.csv");
    // ofstream f_valid_score_63xs_reorder("../valid/reorder_valid_score_63xs_20160610165529.csv");
    // ofstream f_valid_score_63ys_reorder("../valid/reorder_valid_score_63ys_20160610165711.csv");
    // ofstream f_valid_score_63xys_reorder("../valid/reorder_valid_score_63xys_20160610171448.csv");

    // reorder_score(f_valid_score_63, f_valid_score_63_reorder);
    // reorder_score(f_valid_score_63xs, f_valid_score_63xs_reorder);
    // reorder_score(f_valid_score_63ys, f_valid_score_63ys_reorder);
    // reorder_score(f_valid_score_63xys, f_valid_score_63xys_reorder);

    // f_valid_score_63.close();
    // f_valid_score_63xs.close();
    // f_valid_score_63ys.close();
    // f_valid_score_63xys.close();
    // f_valid_score_63_reorder.close();
    // f_valid_score_63xs_reorder.close();
    // f_valid_score_63ys_reorder.close();
    // f_valid_score_63xys_reorder.close();

    // ifstream f_valid_score_83("../valid/valid_score_83_20160621012105.csv");
    // ifstream f_valid_score_83xs("../valid/valid_score_83xs_20160621012258.csv");
    // ifstream f_valid_score_83ys("../valid/valid_score_83ys_20160621012335.csv");
    // ifstream f_valid_score_83xys("../valid/valid_score_83xys_20160621012424.csv");

    // ofstream f_valid_score_83_reorder("../valid/reorder_valid_score_83_20160621012105.csv");
    // ofstream f_valid_score_83xs_reorder("../valid/reorder_valid_score_83xs_20160621012258.csv");
    // ofstream f_valid_score_83ys_reorder("../valid/reorder_valid_score_83ys_20160621012335.csv");
    // ofstream f_valid_score_83xys_reorder("../valid/reorder_valid_score_83xys_20160621012424.csv");

    // reorder_score(f_valid_score_83, f_valid_score_83_reorder);
    // reorder_score(f_valid_score_83xs, f_valid_score_83xs_reorder);
    // reorder_score(f_valid_score_83ys, f_valid_score_83ys_reorder);
    // reorder_score(f_valid_score_83xys, f_valid_score_83xys_reorder);

    // f_valid_score_83.close();
    // f_valid_score_83xs.close();
    // f_valid_score_83ys.close();
    // f_valid_score_83xys.close();
    // f_valid_score_83_reorder.close();
    // f_valid_score_83xs_reorder.close();
    // f_valid_score_83ys_reorder.close();
    // f_valid_score_83xys_reorder.close();


    // ifstream f_valid_score_xgb_new2_100("../valid/valid_score_xgb_new2_100_20160622121911.csv");
    // ifstream f_valid_score_xgb_new2_100xs("../valid/valid_score_xgb_new2_100xs_20160623205834.csv");
    // ifstream f_valid_score_xgb_new2_100ys("../valid/valid_score_xgb_new2_100ys_20160623205919.csv");
    // ifstream f_valid_score_xgb_new2_100xys("../valid/valid_score_xgb_new2_100xys_20160623210050.csv");

    // ofstream f_valid_score_xgb_new2_100_reorder("../valid/reorder_valid_score_xgb_new2_100_20160622121911.csv");
    // ofstream f_valid_score_xgb_new2_100xs_reorder("../valid/reorder_valid_score_xgb_new2_100xs_20160623205834.csv");
    // ofstream f_valid_score_xgb_new2_100ys_reorder("../valid/reorder_valid_score_xgb_new2_100ys_20160623205919.csv");
    // ofstream f_valid_score_xgb_new2_100xys_reorder("../valid/reorder_valid_score_xgb_new2_100xys_20160623210050.csv");

    // reorder_score(f_valid_score_xgb_new2_100, f_valid_score_xgb_new2_100_reorder);
    // reorder_score(f_valid_score_xgb_new2_100xs, f_valid_score_xgb_new2_100xs_reorder);
    // reorder_score(f_valid_score_xgb_new2_100ys, f_valid_score_xgb_new2_100ys_reorder);
    // reorder_score(f_valid_score_xgb_new2_100xys, f_valid_score_xgb_new2_100xys_reorder);

    // f_valid_score_xgb_new2_100.close();
    // f_valid_score_xgb_new2_100xs.close();
    // f_valid_score_xgb_new2_100ys.close();
    // f_valid_score_xgb_new2_100xys.close();
    // f_valid_score_xgb_new2_100_reorder.close();
    // f_valid_score_xgb_new2_100xs_reorder.close();
    // f_valid_score_xgb_new2_100ys_reorder.close();
    // f_valid_score_xgb_new2_100xys_reorder.close();


    // ifstream f_valid_score_xgb_new2_71("../valid/valid_score_xgb_new2_71_20160627122639.csv");
    // ifstream f_valid_score_xgb_new2_71xs("../valid/valid_score_xgb_new2_71xs_20160627122713.csv");
    // ifstream f_valid_score_xgb_new2_71ys("../valid/valid_score_xgb_new2_71ys_20160627122814.csv");
    // ifstream f_valid_score_xgb_new2_71xys("../valid/valid_score_xgb_new2_71xys_20160627122942.csv");

    // ofstream f_valid_score_xgb_new2_71_reorder("../valid/reorder_valid_score_xgb_new2_71_20160627122639.csv");
    // ofstream f_valid_score_xgb_new2_71xs_reorder("../valid/reorder_valid_score_xgb_new2_71xs_20160627122713.csv");
    // ofstream f_valid_score_xgb_new2_71ys_reorder("../valid/reorder_valid_score_xgb_new2_71ys_20160627122814.csv");
    // ofstream f_valid_score_xgb_new2_71xys_reorder("../valid/reorder_valid_score_xgb_new2_71xys_20160627122942.csv");

    // reorder_score(f_valid_score_xgb_new2_71, f_valid_score_xgb_new2_71_reorder);
    // reorder_score(f_valid_score_xgb_new2_71xs, f_valid_score_xgb_new2_71xs_reorder);
    // reorder_score(f_valid_score_xgb_new2_71ys, f_valid_score_xgb_new2_71ys_reorder);
    // reorder_score(f_valid_score_xgb_new2_71xys, f_valid_score_xgb_new2_71xys_reorder);

    // f_valid_score_xgb_new2_71.close();
    // f_valid_score_xgb_new2_71xs.close();
    // f_valid_score_xgb_new2_71ys.close();
    // f_valid_score_xgb_new2_71xys.close();
    // f_valid_score_xgb_new2_71_reorder.close();
    // f_valid_score_xgb_new2_71xs_reorder.close();
    // f_valid_score_xgb_new2_71ys_reorder.close();
    // f_valid_score_xgb_new2_71xys_reorder.close();


    // ifstream f_valid_score_xgb_new2_90("../valid/valid_score_xgb_new2_90_20160623210124.csv");
    // ifstream f_valid_score_xgb_new2_90xs("../valid/valid_score_xgb_new2_90xs_20160624102803.csv");
    // ifstream f_valid_score_xgb_new2_90ys("../valid/valid_score_xgb_new2_90ys_20160623210225.csv");
    // ifstream f_valid_score_xgb_new2_90xys("../valid/valid_score_xgb_new2_90xys_20160623210250.csv");

    // ofstream f_valid_score_xgb_new2_90_reorder("../valid/reorder_valid_score_xgb_new2_90_20160623210124.csv");
    // ofstream f_valid_score_xgb_new2_90xs_reorder("../valid/reorder_valid_score_xgb_new2_90xs_20160624102803.csv");
    // ofstream f_valid_score_xgb_new2_90ys_reorder("../valid/reorder_valid_score_xgb_new2_90ys_20160623210225.csv");
    // ofstream f_valid_score_xgb_new2_90xys_reorder("../valid/reorder_valid_score_xgb_new2_90xys_20160623210250.csv");

    // reorder_score(f_valid_score_xgb_new2_90, f_valid_score_xgb_new2_90_reorder);
    // reorder_score(f_valid_score_xgb_new2_90xs, f_valid_score_xgb_new2_90xs_reorder);
    // reorder_score(f_valid_score_xgb_new2_90ys, f_valid_score_xgb_new2_90ys_reorder);
    // reorder_score(f_valid_score_xgb_new2_90xys, f_valid_score_xgb_new2_90xys_reorder);

    // f_valid_score_xgb_new2_90.close();
    // f_valid_score_xgb_new2_90xs.close();
    // f_valid_score_xgb_new2_90ys.close();
    // f_valid_score_xgb_new2_90xys.close();
    // f_valid_score_xgb_new2_90_reorder.close();
    // f_valid_score_xgb_new2_90xs_reorder.close();
    // f_valid_score_xgb_new2_90ys_reorder.close();
    // f_valid_score_xgb_new2_90xys_reorder.close();


    // ifstream f_valid_score_xgb_new2_63("../valid/valid_score_xgb_new2_63_20160627123102.csv");
    // ifstream f_valid_score_xgb_new2_63xs("../valid/valid_score_xgb_new2_63xs_20160627123139.csv");
    // ifstream f_valid_score_xgb_new2_63ys("../valid/valid_score_xgb_new2_63ys_20160627123204.csv");
    // ifstream f_valid_score_xgb_new2_63xys("../valid/valid_score_xgb_new2_63xys_20160627123228.csv");

    // ofstream f_valid_score_xgb_new2_63_reorder("../valid/reorder_valid_score_xgb_new2_63_20160627123102.csv");
    // ofstream f_valid_score_xgb_new2_63xs_reorder("../valid/reorder_valid_score_xgb_new2_63xs_20160627123139.csv");
    // ofstream f_valid_score_xgb_new2_63ys_reorder("../valid/reorder_valid_score_xgb_new2_63ys_20160627123204.csv");
    // ofstream f_valid_score_xgb_new2_63xys_reorder("../valid/reorder_valid_score_xgb_new2_63xys_20160627123228.csv");

    // reorder_score(f_valid_score_xgb_new2_63, f_valid_score_xgb_new2_63_reorder);
    // reorder_score(f_valid_score_xgb_new2_63xs, f_valid_score_xgb_new2_63xs_reorder);
    // reorder_score(f_valid_score_xgb_new2_63ys, f_valid_score_xgb_new2_63ys_reorder);
    // reorder_score(f_valid_score_xgb_new2_63xys, f_valid_score_xgb_new2_63xys_reorder);

    // f_valid_score_xgb_new2_63.close();
    // f_valid_score_xgb_new2_63xs.close();
    // f_valid_score_xgb_new2_63ys.close();
    // f_valid_score_xgb_new2_63xys.close();
    // f_valid_score_xgb_new2_63_reorder.close();
    // f_valid_score_xgb_new2_63xs_reorder.close();
    // f_valid_score_xgb_new2_63ys_reorder.close();
    // f_valid_score_xgb_new2_63xys_reorder.close();


    // ifstream f_valid_score_xgb_aug_nn_100("../valid/valid_score_xgb_100_20160702160701.csv");
    // ifstream f_valid_score_xgb_aug_nn_100xs("../valid/valid_score_xgb_aug_nn_100xs_20160702161826.csv");
    // ifstream f_valid_score_xgb_aug_nn_100ys("../valid/valid_score_xgb_aug_nn_100ys_20160702162024.csv");
    // ifstream f_valid_score_xgb_aug_nn_100xys("../valid/valid_score_xgb_aug_nn_100xys_20160702162115.csv");

    // ofstream f_valid_score_xgb_aug_nn_100_reorder("../valid/reorder_valid_score_xgb_100_20160702160701.csv");
    // ofstream f_valid_score_xgb_aug_nn_100xs_reorder("../valid/reorder_valid_score_xgb_aug_nn_100xs_20160702161826.csv");
    // ofstream f_valid_score_xgb_aug_nn_100ys_reorder("../valid/reorder_valid_score_xgb_aug_nn_100ys_20160702162024.csv");
    // ofstream f_valid_score_xgb_aug_nn_100xys_reorder("../valid/reorder_valid_score_xgb_aug_nn_100xys_20160702162115.csv");    

    // reorder_score(f_valid_score_xgb_aug_nn_100, f_valid_score_xgb_aug_nn_100_reorder);
    // reorder_score(f_valid_score_xgb_aug_nn_100xs, f_valid_score_xgb_aug_nn_100xs_reorder);
    // reorder_score(f_valid_score_xgb_aug_nn_100ys, f_valid_score_xgb_aug_nn_100ys_reorder);
    // reorder_score(f_valid_score_xgb_aug_nn_100xys, f_valid_score_xgb_aug_nn_100xys_reorder);

    // f_valid_score_xgb_aug_nn_100.close();
    // f_valid_score_xgb_aug_nn_100xs.close();
    // f_valid_score_xgb_aug_nn_100ys.close();
    // f_valid_score_xgb_aug_nn_100xys.close();
    // f_valid_score_xgb_aug_nn_100_reorder.close();
    // f_valid_score_xgb_aug_nn_100xs_reorder.close();
    // f_valid_score_xgb_aug_nn_100ys_reorder.close();
    // f_valid_score_xgb_aug_nn_100xys_reorder.close();

    // ifstream f_valid_score_xgb_aug_nn_90("../valid/valid_score_xgb_aug_nn_90_20160703214350.csv");
    // ifstream f_valid_score_xgb_aug_nn_90xs("../valid/valid_score_xgb_aug_nn_90xs_20160703214507.csv");
    // ifstream f_valid_score_xgb_aug_nn_90ys("../valid/valid_score_xgb_aug_nn_90ys_20160703214623.csv");
    // ifstream f_valid_score_xgb_aug_nn_90xys("../valid/valid_score_xgb_aug_nn_90xys_20160703214813.csv");

    // ofstream f_valid_score_xgb_aug_nn_90_reorder("../valid/reorder_valid_score_xgb_aug_nn_90_20160703214350.csv");
    // ofstream f_valid_score_xgb_aug_nn_90xs_reorder("../valid/reorder_valid_score_xgb_aug_nn_90xs_20160703214507.csv");
    // ofstream f_valid_score_xgb_aug_nn_90ys_reorder("../valid/reorder_valid_score_xgb_aug_nn_90ys_20160703214623.csv");
    // ofstream f_valid_score_xgb_aug_nn_90xys_reorder("../valid/reorder_valid_score_xgb_aug_nn_90xys_20160703214813.csv");    

    // reorder_score(f_valid_score_xgb_aug_nn_90, f_valid_score_xgb_aug_nn_90_reorder);
    // reorder_score(f_valid_score_xgb_aug_nn_90xs, f_valid_score_xgb_aug_nn_90xs_reorder);
    // reorder_score(f_valid_score_xgb_aug_nn_90ys, f_valid_score_xgb_aug_nn_90ys_reorder);
    // reorder_score(f_valid_score_xgb_aug_nn_90xys, f_valid_score_xgb_aug_nn_90xys_reorder);

    // f_valid_score_xgb_aug_nn_90.close();
    // f_valid_score_xgb_aug_nn_90xs.close();
    // f_valid_score_xgb_aug_nn_90ys.close();
    // f_valid_score_xgb_aug_nn_90xys.close();
    // f_valid_score_xgb_aug_nn_90_reorder.close();
    // f_valid_score_xgb_aug_nn_90xs_reorder.close();
    // f_valid_score_xgb_aug_nn_90ys_reorder.close();
    // f_valid_score_xgb_aug_nn_90xys_reorder.close();

    // ifstream f_valid_score_xgb_aug_nn_71("../valid/valid_score_xgb_aug_nn_71_20160702165843.csv");
    // ifstream f_valid_score_xgb_aug_nn_71xs("../valid/valid_score_xgb_aug_nn_71xs_20160702165921.csv.cp");
    // ifstream f_valid_score_xgb_aug_nn_71ys("../valid/valid_score_xgb_aug_nn_71ys_20160702170115.csv");
    // ifstream f_valid_score_xgb_aug_nn_71xys("../valid/valid_score_xgb_aug_nn_71xys_20160702170420.csv");

    // ofstream f_valid_score_xgb_aug_nn_71_reorder("../valid/reorder_valid_score_xgb_aug_nn_71_20160702165843.csv");
    // ofstream f_valid_score_xgb_aug_nn_71xs_reorder("../valid/reorder_valid_score_xgb_aug_nn_71xs_20160702165921.csv");
    // ofstream f_valid_score_xgb_aug_nn_71ys_reorder("../valid/reorder_valid_score_xgb_aug_nn_71ys_20160702170115.csv");
    // ofstream f_valid_score_xgb_aug_nn_71xys_reorder("../valid/reorder_valid_score_xgb_aug_nn_71xys_20160702170420.csv");    

    // reorder_score(f_valid_score_xgb_aug_nn_71, f_valid_score_xgb_aug_nn_71_reorder);
    // reorder_score(f_valid_score_xgb_aug_nn_71xs, f_valid_score_xgb_aug_nn_71xs_reorder);
    // reorder_score(f_valid_score_xgb_aug_nn_71ys, f_valid_score_xgb_aug_nn_71ys_reorder);
    // reorder_score(f_valid_score_xgb_aug_nn_71xys, f_valid_score_xgb_aug_nn_71xys_reorder);

    // f_valid_score_xgb_aug_nn_71.close();
    // f_valid_score_xgb_aug_nn_71xs.close();
    // f_valid_score_xgb_aug_nn_71ys.close();
    // f_valid_score_xgb_aug_nn_71xys.close();
    // f_valid_score_xgb_aug_nn_71_reorder.close();
    // f_valid_score_xgb_aug_nn_71xs_reorder.close();
    // f_valid_score_xgb_aug_nn_71ys_reorder.close();
    // f_valid_score_xgb_aug_nn_71xys_reorder.close();

    // ifstream f_valid_score_xgb_aug_nn_63("../valid/valid_score_xgb_aug_nn_63_20160702172227.csv");
    // ifstream f_valid_score_xgb_aug_nn_63xs("../valid/valid_score_xgb_aug_nn_63xs_20160702172327.csv");
    // ifstream f_valid_score_xgb_aug_nn_63ys("../valid/valid_score_xgb_aug_nn_63ys_20160702172433.csv");
    // ifstream f_valid_score_xgb_aug_nn_63xys("../valid/valid_score_xgb_aug_nn_63xys_20160702172549.csv.cp");

    // ofstream f_valid_score_xgb_aug_nn_63_reorder("../valid/reorder_valid_score_xgb_aug_nn_63_20160702172227.csv");
    // ofstream f_valid_score_xgb_aug_nn_63xs_reorder("../valid/reorder_valid_score_xgb_aug_nn_63xs_20160702172327.csv");
    // ofstream f_valid_score_xgb_aug_nn_63ys_reorder("../valid/reorder_valid_score_xgb_aug_nn_63ys_20160702172433.csv");
    // ofstream f_valid_score_xgb_aug_nn_63xys_reorder("../valid/reorder_valid_score_xgb_aug_nn_63xys_20160702172549.csv");    

    // reorder_score(f_valid_score_xgb_aug_nn_63, f_valid_score_xgb_aug_nn_63_reorder);
    // reorder_score(f_valid_score_xgb_aug_nn_63xs, f_valid_score_xgb_aug_nn_63xs_reorder);
    // reorder_score(f_valid_score_xgb_aug_nn_63ys, f_valid_score_xgb_aug_nn_63ys_reorder);
    // reorder_score(f_valid_score_xgb_aug_nn_63xys, f_valid_score_xgb_aug_nn_63xys_reorder);

    // f_valid_score_xgb_aug_nn_63.close();
    // f_valid_score_xgb_aug_nn_63xs.close();
    // f_valid_score_xgb_aug_nn_63ys.close();
    // f_valid_score_xgb_aug_nn_63xys.close();
    // f_valid_score_xgb_aug_nn_63_reorder.close();
    // f_valid_score_xgb_aug_nn_63xs_reorder.close();
    // f_valid_score_xgb_aug_nn_63ys_reorder.close();
    // f_valid_score_xgb_aug_nn_63xys_reorder.close();

    ifstream f_valid_score_knn_count_20("../valid/valid_score_knn_count_20_20160706045201.csv");
    ofstream f_valid_score_knn_count_20_reorder("../valid/reorder_valid_score_knn_count_20_20160706045201.csv");

    reorder_score(f_valid_score_knn_count_20, f_valid_score_knn_count_20_reorder);
    f_valid_score_knn_count_20.close();
    f_valid_score_knn_count_20_reorder.close();

    // ifstream f_valid_score_knn_20("../valid/valid_score_knn_20_20160701024005.csv");
    // ofstream f_valid_score_knn_20_reorder("../valid/reorder_valid_score_knn_20_20160701024005.csv");

    // reorder_score(f_valid_score_knn_20, f_valid_score_knn_20_reorder);

    // f_valid_score_knn_20.close();
    // f_valid_score_knn_20_reorder.close();

    // ifstream f_valid_score_knn_new2_20("../valid/valid_score_knn_20_20160701170046.csv");
    // ofstream f_valid_score_knn_new2_20_reorder("../valid/reorder_valid_score_knn_20_20160701170046.csv");

    // reorder_score(f_valid_score_knn_new2_20, f_valid_score_knn_new2_20_reorder);

    // f_valid_score_knn_new2_20.close();
    // f_valid_score_knn_new2_20_reorder.close();
    /* end reorder row_id */

    /* generate score from file */
}

void check_validation(){
    int max_rank = 8, max_sub = 3;
    size_t num_valid_train = 23318975, num_valid_test = 5799046;
    double mapk_score = 0, weight = 1;
    unsigned int limit = 50;

    ifstream f_valid_test("../valid/valid_test.csv");
    
    ifstream reorder_f_valid_score_100("../valid/reorder_valid_score_100_20160605000504.csv");
    ifstream reorder_f_valid_score_100xs("../valid/reorder_valid_score_100xs_20160606172637.csv");
    ifstream reorder_f_valid_score_100ys("../valid/reorder_valid_score_100ys_20160606172711.csv");
    ifstream reorder_f_valid_score_100xys("../valid/reorder_valid_score_100xys_20160607130744.csv");

    ifstream reorder_f_valid_score_71("../valid/reorder_valid_score_71_20160605004036.csv");
    ifstream reorder_f_valid_score_71xs("../valid/reorder_valid_score_71xs_20160607234123.csv");
    ifstream reorder_f_valid_score_71ys("../valid/reorder_valid_score_71ys_20160607234206.csv");
    ifstream reorder_f_valid_score_71xys("../valid/reorder_valid_score_71xys_20160607234405.csv");

    ifstream reorder_f_valid_score_90("../valid/reorder_valid_score_90_20160608142409.csv");
    ifstream reorder_f_valid_score_90xs("../valid/reorder_valid_score_90xs_20160608143143.csv");
    ifstream reorder_f_valid_score_90ys("../valid/reorder_valid_score_90ys_20160609202512.csv");
    ifstream reorder_f_valid_score_90xys("../valid/reorder_valid_score_90xys_20160609202658.csv");

    ifstream reorder_f_valid_score_63("../valid/reorder_valid_score_63_20160610165455.csv");
    ifstream reorder_f_valid_score_63xs("../valid/reorder_valid_score_63xs_20160610165529.csv");
    ifstream reorder_f_valid_score_63ys("../valid/reorder_valid_score_63ys_20160610165711.csv");
    ifstream reorder_f_valid_score_63xys("../valid/reorder_valid_score_63xys_20160610171448.csv");

    ifstream reorder_f_valid_score_83("../valid/reorder_valid_score_83_20160621012105.csv");
    ifstream reorder_f_valid_score_83xs("../valid/reorder_valid_score_83xs_20160621012258.csv");
    ifstream reorder_f_valid_score_83ys("../valid/reorder_valid_score_83ys_20160621012335.csv");
    ifstream reorder_f_valid_score_83xys("../valid/reorder_valid_score_83xys_20160621012424.csv");  

    // ifstream reorder_f_valid_score_knn_20("../valid/reorder_valid_score_knn_20_20160701024005.csv");
    // using knn below !

    ifstream reorder_f_valid_score_xgb_new2_100("../valid/reorder_valid_score_xgb_new2_100_20160622121911.csv");
    ifstream reorder_f_valid_score_xgb_new2_100xs("../valid/reorder_valid_score_xgb_new2_100xs_20160623205834.csv");
    ifstream reorder_f_valid_score_xgb_new2_100ys("../valid/reorder_valid_score_xgb_new2_100ys_20160623205919.csv");
    ifstream reorder_f_valid_score_xgb_new2_100xys("../valid/reorder_valid_score_xgb_new2_100xys_20160623210050.csv");

    ifstream reorder_f_valid_score_xgb_new2_71("../valid/reorder_valid_score_xgb_new2_71_20160627122639.csv");
    ifstream reorder_f_valid_score_xgb_new2_71xs("../valid/reorder_valid_score_xgb_new2_71xs_20160627122713.csv");
    ifstream reorder_f_valid_score_xgb_new2_71ys("../valid/reorder_valid_score_xgb_new2_71ys_20160627122814.csv");
    ifstream reorder_f_valid_score_xgb_new2_71xys("../valid/reorder_valid_score_xgb_new2_71xys_20160627122942.csv");

    ifstream reorder_f_valid_score_xgb_new2_90("../valid/reorder_valid_score_xgb_new2_90_20160623210124.csv");
    ifstream reorder_f_valid_score_xgb_new2_90xs("../valid/reorder_valid_score_xgb_new2_90xs_20160624102803.csv");
    ifstream reorder_f_valid_score_xgb_new2_90ys("../valid/reorder_valid_score_xgb_new2_90ys_20160623210225.csv");
    ifstream reorder_f_valid_score_xgb_new2_90xys("../valid/reorder_valid_score_xgb_new2_90xys_20160623210250.csv");

    ifstream reorder_f_valid_score_xgb_new2_63("../valid/reorder_valid_score_xgb_new2_63_20160627123102.csv");
    ifstream reorder_f_valid_score_xgb_new2_63xs("../valid/reorder_valid_score_xgb_new2_63xs_20160627123139.csv");
    ifstream reorder_f_valid_score_xgb_new2_63ys("../valid/reorder_valid_score_xgb_new2_63ys_20160627123204.csv");
    ifstream reorder_f_valid_score_xgb_new2_63xys("../valid/reorder_valid_score_xgb_new2_63xys_20160627123228.csv");    

    // aug nn
    ifstream reorder_f_valid_score_xgb_aug_nn_100("../valid/reorder_valid_score_xgb_100_20160702160701.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_100xs("../valid/reorder_valid_score_xgb_aug_nn_100xs_20160702161826.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_100ys("../valid/reorder_valid_score_xgb_aug_nn_100ys_20160702162024.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_100xys("../valid/reorder_valid_score_xgb_aug_nn_100xys_20160702162115.csv");

    ifstream reorder_f_valid_score_xgb_aug_nn_71("../valid/reorder_valid_score_xgb_aug_nn_71_20160702165843.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_71xs("../valid/reorder_valid_score_xgb_aug_nn_71xs_20160702165921.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_71ys("../valid/reorder_valid_score_xgb_aug_nn_71ys_20160702170115.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_71xys("../valid/reorder_valid_score_xgb_aug_nn_71xys_20160702170420.csv");  

    ifstream reorder_f_valid_score_xgb_aug_nn_90("../valid/reorder_valid_score_xgb_aug_nn_90_20160703214350.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_90xs("../valid/reorder_valid_score_xgb_aug_nn_90xs_20160703214507.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_90ys("../valid/reorder_valid_score_xgb_aug_nn_90ys_20160703214623.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_90xys("../valid/reorder_valid_score_xgb_aug_nn_90xys_20160703214813.csv");  

    ifstream reorder_f_valid_score_xgb_aug_nn_63("../valid/reorder_valid_score_xgb_aug_nn_63_20160702172227.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_63xs("../valid/reorder_valid_score_xgb_aug_nn_63xs_20160702172327.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_63ys("../valid/reorder_valid_score_xgb_aug_nn_63ys_20160702172433.csv");
    ifstream reorder_f_valid_score_xgb_aug_nn_63xys("../valid/reorder_valid_score_xgb_aug_nn_63xys_20160702172549.csv");  

    ifstream reorder_f_valid_score_knn_new2_20("../valid/reorder_valid_score_knn_20_20160701170046.csv");  
    ifstream reorder_f_valid_score_knn_count_20("../valid/reorder_valid_score_knn_count_20_20160706045201.csv");

    // ifstream reorder_f_valid_score_83("../valid/reorder_valid_score_83.csv");
    // ifstream reorder_f_valid_score_83xs("../valid/reorder_valid_score_83xs.csv");
    // ifstream reorder_f_valid_score_83ys("../valid/reorder_valid_score_83ys.csv");
    // ifstream reorder_f_valid_score_83xys("../valid/reorder_valid_score_83xys.csv");

    // // // ofstream f_valid_test_sub("../valid/valid_test_sub.csv");

    // // // vector<double> all_weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0};   
    // // vector<double> all_weights(8, 1.0);   
    // // vector<ifstream*> f_reorder_valid_scores(8);
    // // copy(f_reorder_valid_scores_100.begin(), f_reorder_valid_scores_100.end(), f_reorder_valid_scores.begin());
    // // copy(f_reorder_valid_scores_71.begin(), f_reorder_valid_scores_71.end(), f_reorder_valid_scores.begin()+4);
    // // // copy(f_reorder_valid_scores_83.begin(), f_reorder_valid_scores_83.end(), f_reorder_valid_scores.begin()+8);
    
    // vector<double> weights(16, 1.0);
    // vector<ifstream*> f_reorder_valid_scores = {
    // &reorder_f_valid_score_xgb_new2_100, &reorder_f_valid_score_xgb_new2_100xs, &reorder_f_valid_score_xgb_new2_100ys, &reorder_f_valid_score_xgb_new2_100xys,
    // &reorder_f_valid_score_xgb_new2_71, &reorder_f_valid_score_xgb_new2_71xs, &reorder_f_valid_score_xgb_new2_71ys, &reorder_f_valid_score_xgb_new2_71xys,
    // &reorder_f_valid_score_xgb_new2_90, &reorder_f_valid_score_xgb_new2_90xs, &reorder_f_valid_score_xgb_new2_90ys, &reorder_f_valid_score_xgb_new2_90xys,
    // &reorder_f_valid_score_xgb_new2_63, &reorder_f_valid_score_xgb_new2_63xs, &reorder_f_valid_score_xgb_new2_63ys, &reorder_f_valid_score_xgb_new2_63xys
    // };

    // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_valid_scores, weights, max_sub, num_valid_test);

    // cout << mapk_score << endl;

    // weights = vector<double> (12, 1.0);
    // f_reorder_valid_scores = vector<ifstream*>({
    // &reorder_f_valid_score_xgb_new2_100, &reorder_f_valid_score_xgb_new2_100xs, &reorder_f_valid_score_xgb_new2_100ys, &reorder_f_valid_score_xgb_new2_100xys,
    // &reorder_f_valid_score_xgb_new2_71, &reorder_f_valid_score_xgb_new2_71xs, &reorder_f_valid_score_xgb_new2_71ys, &reorder_f_valid_score_xgb_new2_71xys,
    // &reorder_f_valid_score_xgb_new2_90, &reorder_f_valid_score_xgb_new2_90xs, &reorder_f_valid_score_xgb_new2_90ys, &reorder_f_valid_score_xgb_new2_90xys
    // });

    // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_valid_scores, weights, max_sub, num_valid_test);

    // cout << mapk_score << endl;    

    // vector<double> weights_add(20, 1.0);
    // vector<ifstream*> f_reorder_valid_scores_add = {
    // &reorder_f_valid_score_100, &reorder_f_valid_score_100xs, &reorder_f_valid_score_100ys, &reorder_f_valid_score_100xys,
    // &reorder_f_valid_score_71, &reorder_f_valid_score_71xs, &reorder_f_valid_score_71ys, &reorder_f_valid_score_71xys,
    // &reorder_f_valid_score_90, &reorder_f_valid_score_90xs, &reorder_f_valid_score_90ys, &reorder_f_valid_score_90xys,
    // &reorder_f_valid_score_63, &reorder_f_valid_score_63xs, &reorder_f_valid_score_63ys, &reorder_f_valid_score_63xys,
    // &reorder_f_valid_score_83, &reorder_f_valid_score_83xs, &reorder_f_valid_score_83ys, &reorder_f_valid_score_83xys
    // };

    // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_valid_scores_add, weights_add, max_sub, num_valid_test);

    // cout << mapk_score << endl;

    // vector<double> weights(16, 1.0);
    // vector<ifstream*> f_reorder_valid_scores = {
    // &reorder_f_valid_score_xgb_new2_100, &reorder_f_valid_score_xgb_new2_100xs, &reorder_f_valid_score_xgb_new2_100ys, &reorder_f_valid_score_xgb_new2_100xys,
    // &reorder_f_valid_score_xgb_new2_71, &reorder_f_valid_score_xgb_new2_71xs, &reorder_f_valid_score_xgb_new2_71ys, &reorder_f_valid_score_xgb_new2_71xys,
    // &reorder_f_valid_score_xgb_new2_90, &reorder_f_valid_score_xgb_new2_90xs, &reorder_f_valid_score_xgb_new2_90ys, &reorder_f_valid_score_xgb_new2_90xys,
    // &reorder_f_valid_score_xgb_new2_63, &reorder_f_valid_score_xgb_new2_63xs, &reorder_f_valid_score_xgb_new2_63ys, &reorder_f_valid_score_xgb_new2_63xys
    // };

    // cout << f_reorder_valid_scores.size() << endl;

    // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_valid_scores, weights, max_sub, num_valid_test);

    // cout << mapk_score << endl;

    /* testing xgb */
    // vector<double> weights_all(12, 1.0);
    // vector<ifstream*> f_reorder_valid_scores_all = {
    // &reorder_f_valid_score_xgb_new2_100, &reorder_f_valid_score_xgb_new2_100xs, &reorder_f_valid_score_xgb_new2_100ys, &reorder_f_valid_score_xgb_new2_100xys,
    // &reorder_f_valid_score_100, &reorder_f_valid_score_100xs, &reorder_f_valid_score_100ys, &reorder_f_valid_score_100xys,
    // &reorder_f_valid_score_xgb_aug_nn_100, &reorder_f_valid_score_xgb_aug_nn_100xs, &reorder_f_valid_score_xgb_aug_nn_100ys, &reorder_f_valid_score_xgb_aug_nn_100xys
    // };

    // vector<double> weights_aug_nn(16, 1.0);
    // vector<ifstream*> f_reorder_valid_scores_aug_nn = {
    // &reorder_f_valid_score_xgb_aug_nn_100, &reorder_f_valid_score_xgb_aug_nn_100xs, &reorder_f_valid_score_xgb_aug_nn_100ys, &reorder_f_valid_score_xgb_aug_nn_100xys,
    // &reorder_f_valid_score_xgb_aug_nn_71, &reorder_f_valid_score_xgb_aug_nn_71xs, &reorder_f_valid_score_xgb_aug_nn_71ys, &reorder_f_valid_score_xgb_aug_nn_71xys,
    // &reorder_f_valid_score_xgb_aug_nn_90, &reorder_f_valid_score_xgb_aug_nn_90xs, &reorder_f_valid_score_xgb_aug_nn_90ys, &reorder_f_valid_score_xgb_aug_nn_90xys,
    // &reorder_f_valid_score_xgb_aug_nn_63, &reorder_f_valid_score_xgb_aug_nn_63xs, &reorder_f_valid_score_xgb_aug_nn_63ys, &reorder_f_valid_score_xgb_aug_nn_63xys      
    // };  

    // vector<double> weights_all(48, 1.0);
    // vector<ifstream*> f_reorder_valid_scores_all = {
    // &reorder_f_valid_score_xgb_new2_100, &reorder_f_valid_score_xgb_new2_100xs, &reorder_f_valid_score_xgb_new2_100ys, &reorder_f_valid_score_xgb_new2_100xys,
    // &reorder_f_valid_score_xgb_new2_71, &reorder_f_valid_score_xgb_new2_71xs, &reorder_f_valid_score_xgb_new2_71ys, &reorder_f_valid_score_xgb_new2_71xys,
    // &reorder_f_valid_score_xgb_new2_90, &reorder_f_valid_score_xgb_new2_90xs, &reorder_f_valid_score_xgb_new2_90ys, &reorder_f_valid_score_xgb_new2_90xys,
    // &reorder_f_valid_score_xgb_new2_63, &reorder_f_valid_score_xgb_new2_63xs, &reorder_f_valid_score_xgb_new2_63ys, &reorder_f_valid_score_xgb_new2_63xys,
    // &reorder_f_valid_score_100, &reorder_f_valid_score_100xs, &reorder_f_valid_score_100ys, &reorder_f_valid_score_100xys,
    // &reorder_f_valid_score_71, &reorder_f_valid_score_71xs, &reorder_f_valid_score_71ys, &reorder_f_valid_score_71xys,
    // &reorder_f_valid_score_90, &reorder_f_valid_score_90xs, &reorder_f_valid_score_90ys, &reorder_f_valid_score_90xys,
    // &reorder_f_valid_score_63, &reorder_f_valid_score_63xs, &reorder_f_valid_score_63ys, &reorder_f_valid_score_63xys,
    // &reorder_f_valid_score_xgb_aug_nn_100, &reorder_f_valid_score_xgb_aug_nn_100xs, &reorder_f_valid_score_xgb_aug_nn_100ys, &reorder_f_valid_score_xgb_aug_nn_100xys,
    // &reorder_f_valid_score_xgb_aug_nn_71, &reorder_f_valid_score_xgb_aug_nn_71xs, &reorder_f_valid_score_xgb_aug_nn_71ys, &reorder_f_valid_score_xgb_aug_nn_71xys,
    // &reorder_f_valid_score_xgb_aug_nn_90, &reorder_f_valid_score_xgb_aug_nn_90xs, &reorder_f_valid_score_xgb_aug_nn_90ys, &reorder_f_valid_score_xgb_aug_nn_90xys,
    // &reorder_f_valid_score_xgb_aug_nn_63, &reorder_f_valid_score_xgb_aug_nn_63xs, &reorder_f_valid_score_xgb_aug_nn_63ys, &reorder_f_valid_score_xgb_aug_nn_63xys      
    // };    

    // size_t cache_size = 2000000;

    // // partial check
    // num_valid_test = cache_size;    

    // cout << f_reorder_valid_scores_aug_nn.size() << endl;

    // mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_aug_nn, weights_aug_nn, max_sub, num_valid_test, cache_size);

    // cout << mapk_score << endl;

    // cout << f_reorder_valid_scores_all.size() << endl;
    
    // // mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);
    // // // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test);

    // // cout << mapk_score << endl;



    // for (int i = 32; i < 48; i++){
    //     weights_all[i] = 2.0;
    // }      

    // mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    // cout << mapk_score << endl;

    // for (int i = 32; i < 48; i++){
    //     weights_all[i] = 2.5;
    // }        

    // mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    // cout << mapk_score << endl;

    // for (int i = 32; i < 48; i++){
    //     weights_all[i] = 3.0;
    // }        

    // mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    // cout << mapk_score << endl;

    /* xgb test end */

    /* testing all */
    vector<double> weights_all(49, 1.0);
    vector<ifstream*> f_reorder_valid_scores_all = {
    &reorder_f_valid_score_xgb_new2_100, &reorder_f_valid_score_xgb_new2_100xs, &reorder_f_valid_score_xgb_new2_100ys, &reorder_f_valid_score_xgb_new2_100xys,
    &reorder_f_valid_score_xgb_new2_71, &reorder_f_valid_score_xgb_new2_71xs, &reorder_f_valid_score_xgb_new2_71ys, &reorder_f_valid_score_xgb_new2_71xys,
    &reorder_f_valid_score_xgb_new2_90, &reorder_f_valid_score_xgb_new2_90xs, &reorder_f_valid_score_xgb_new2_90ys, &reorder_f_valid_score_xgb_new2_90xys,
    &reorder_f_valid_score_xgb_new2_63, &reorder_f_valid_score_xgb_new2_63xs, &reorder_f_valid_score_xgb_new2_63ys, &reorder_f_valid_score_xgb_new2_63xys,
    &reorder_f_valid_score_100, &reorder_f_valid_score_100xs, &reorder_f_valid_score_100ys, &reorder_f_valid_score_100xys,
    &reorder_f_valid_score_71, &reorder_f_valid_score_71xs, &reorder_f_valid_score_71ys, &reorder_f_valid_score_71xys,
    &reorder_f_valid_score_90, &reorder_f_valid_score_90xs, &reorder_f_valid_score_90ys, &reorder_f_valid_score_90xys,
    &reorder_f_valid_score_63, &reorder_f_valid_score_63xs, &reorder_f_valid_score_63ys, &reorder_f_valid_score_63xys,
    &reorder_f_valid_score_xgb_aug_nn_100, &reorder_f_valid_score_xgb_aug_nn_100xs, &reorder_f_valid_score_xgb_aug_nn_100ys, &reorder_f_valid_score_xgb_aug_nn_100xys,
    &reorder_f_valid_score_xgb_aug_nn_71, &reorder_f_valid_score_xgb_aug_nn_71xs, &reorder_f_valid_score_xgb_aug_nn_71ys, &reorder_f_valid_score_xgb_aug_nn_71xys,
    &reorder_f_valid_score_xgb_aug_nn_90, &reorder_f_valid_score_xgb_aug_nn_90xs, &reorder_f_valid_score_xgb_aug_nn_90ys, &reorder_f_valid_score_xgb_aug_nn_90xys,
    &reorder_f_valid_score_xgb_aug_nn_63, &reorder_f_valid_score_xgb_aug_nn_63xs, &reorder_f_valid_score_xgb_aug_nn_63ys, &reorder_f_valid_score_xgb_aug_nn_63xys,
    &reorder_f_valid_score_knn_count_20 
    };    

    size_t cache_size = 500000;
    // partial check
    num_valid_test = cache_size; 

    for (int i = 32; i < 48; i++){
        weights_all[i] = 2.0;
    } 

    cout << f_reorder_valid_scores_all.size() << endl;
    
    // // mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);
    // // // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test);

    // // cout << mapk_score << endl;

    weights_all[48] = 8.0;

    cout << weights_all[48] << endl;

    mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    cout << mapk_score << endl;

    for (int i = 0; i < 16; i++){
        weights_all[i] = 0.8;
    }
    for (int i = 16; i < 32; i++){
        weights_all[i] = 1.2;
    }

    mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    cout << mapk_score << endl;

    for (int i = 0; i < 16; i++){
        weights_all[i] = 0.6;
    }
    for (int i = 16; i < 32; i++){
        weights_all[i] = 1.4;
    }

    mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    cout << mapk_score << endl;    


    // weights_all[48] = 0.8;

    // cout << weights_all[48] << endl;

    // mapk_score = file_cache_generate_validate_apk_log(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    // cout << mapk_score << endl;

    // weights_all[48] = 1.2;

    // cout << weights_all[48] << endl;

    // mapk_score = file_cache_generate_validate_apk_log(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    // cout << mapk_score << endl;

    // weights_all[48] = 1.8;

    // cout << weights_all[48] << endl;

    // mapk_score = file_cache_generate_validate_apk_log(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    // cout << mapk_score << endl;


    // weights_all[48] = 15.0;

    // cout << weights_all[48] << endl;

    // mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    // cout << mapk_score << endl;

    // weights_all[48] = 20.0;    

    // cout << weights_all[48] << endl;

    // mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    // cout << mapk_score << endl;

    // weights_all[48] = 25.0;    

    // cout << weights_all[48] << endl;

    // mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    // cout << mapk_score << endl;

    // weights_all[48] = 30.0;    

    // cout << weights_all[48] << endl;

    // mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    // cout << mapk_score << endl;     

    /* testing all end */

    // // vector<double> weights = {8.75, 8.75, 8.75, 8.75, 8.75, 8.75, 8.75, 8.75, 2.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5};
    // // vector<ifstream*> f_reorder_valid_scores = {
    // // &reorder_f_valid_score_100, &reorder_f_valid_score_100xs, &reorder_f_valid_score_100ys, &reorder_f_valid_score_100xys,
    // // &reorder_f_valid_score_71, &reorder_f_valid_score_71xs, &reorder_f_valid_score_71ys, &reorder_f_valid_score_71xys,
    // // &reorder_f_valid_score_rf_100, &reorder_f_valid_score_rf_100xs, &reorder_f_valid_score_rf_100ys, &reorder_f_valid_score_rf_100xys,
    // // &reorder_f_valid_score_rf_71, &reorder_f_valid_score_rf_71xs, &reorder_f_valid_score_rf_71ys, &reorder_f_valid_score_rf_71xys
    // // };

    // // cout << f_reorder_valid_scores.size() << endl;

    // // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_valid_scores, weights, max_sub, num_valid_test);

    // // cout << mapk_score << endl;

    // vector<double> weights = {8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1};
    // vector<ifstream*> f_reorder_valid_scores = {
    // &reorder_f_valid_score_100, &reorder_f_valid_score_100xs, &reorder_f_valid_score_100ys, &reorder_f_valid_score_100xys,
    // &reorder_f_valid_score_71, &reorder_f_valid_score_71xs, &reorder_f_valid_score_71ys, &reorder_f_valid_score_71xys,
    // &reorder_f_valid_score_rf_100, &reorder_f_valid_score_rf_100xs, &reorder_f_valid_score_rf_100ys, &reorder_f_valid_score_rf_100xys,
    // &reorder_f_valid_score_rf_71, &reorder_f_valid_score_rf_71xs, &reorder_f_valid_score_rf_71ys, &reorder_f_valid_score_rf_71xys
    // };

    // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_valid_scores, weights, max_sub, num_valid_test);

    // cout << mapk_score << endl;

    // // weights = vector<double>(8, 1.0);
    // // f_reorder_valid_scores = vector<ifstream*>({
    // // &reorder_f_valid_score_100, &reorder_f_valid_score_100xs, &reorder_f_valid_score_100ys, &reorder_f_valid_score_100xys,
    // // &reorder_f_valid_score_71, &reorder_f_valid_score_71xs, &reorder_f_valid_score_71ys, &reorder_f_valid_score_71xys
    // // });

    // // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_valid_scores, weights, max_sub, num_valid_test);

    // // cout << mapk_score << endl;

    // // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_valid_scores, weights, max_sub, num_valid_test);

    // // cout << mapk_score << endl;

    // // weights = vector<double>({3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1});

    // // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_valid_scores, weights, max_sub, num_valid_test);

    // // cout << mapk_score << endl;

    // // // mapk_score = file_generate_validate_apk(f_valid_test, f_valid_scores, weights, max_sub, num_valid_test);

    // // // cout << mapk_score << endl;

    // // // file_generate_sub(f_valid_test_sub, f_valid_scores, weights, max_sub, num_valid_test);


    // // f_reorder_valid_scores_100.clear();
    // // f_reorder_valid_scores_71.clear();
    // // f_reorder_valid_scores_83.clear();

    reorder_f_valid_score_100.close();
    reorder_f_valid_score_100xs.close();
    reorder_f_valid_score_100ys.close();
    reorder_f_valid_score_100xys.close();

    reorder_f_valid_score_71.close();
    reorder_f_valid_score_71xs.close();
    reorder_f_valid_score_71ys.close();
    reorder_f_valid_score_71xys.close();

    reorder_f_valid_score_90.close();
    reorder_f_valid_score_90xs.close();
    reorder_f_valid_score_90ys.close();
    reorder_f_valid_score_90xys.close();

    reorder_f_valid_score_63.close();
    reorder_f_valid_score_63xs.close();
    reorder_f_valid_score_63ys.close();
    reorder_f_valid_score_63xys.close();   

    // reorder_f_valid_score_83.close();
    // reorder_f_valid_score_83xs.close();
    // reorder_f_valid_score_83ys.close();
    // reorder_f_valid_score_83xys.close();  

    reorder_f_valid_score_xgb_new2_100.close();
    reorder_f_valid_score_xgb_new2_100xs.close();
    reorder_f_valid_score_xgb_new2_100ys.close();
    reorder_f_valid_score_xgb_new2_100xys.close();

    reorder_f_valid_score_xgb_new2_71.close();
    reorder_f_valid_score_xgb_new2_71xs.close();
    reorder_f_valid_score_xgb_new2_71ys.close();
    reorder_f_valid_score_xgb_new2_71xys.close();

    reorder_f_valid_score_xgb_new2_90.close();
    reorder_f_valid_score_xgb_new2_90xs.close();
    reorder_f_valid_score_xgb_new2_90ys.close();
    reorder_f_valid_score_xgb_new2_90xys.close();

    reorder_f_valid_score_xgb_new2_63.close();
    reorder_f_valid_score_xgb_new2_63xs.close();
    reorder_f_valid_score_xgb_new2_63ys.close();
    reorder_f_valid_score_xgb_new2_63xys.close(); 

    reorder_f_valid_score_xgb_aug_nn_100.close();
    reorder_f_valid_score_xgb_aug_nn_100xs.close();
    reorder_f_valid_score_xgb_aug_nn_100ys.close();
    reorder_f_valid_score_xgb_aug_nn_100xys.close();

    reorder_f_valid_score_xgb_aug_nn_71.close();
    reorder_f_valid_score_xgb_aug_nn_71xs.close();
    reorder_f_valid_score_xgb_aug_nn_71ys.close();
    reorder_f_valid_score_xgb_aug_nn_71xys.close();

    reorder_f_valid_score_xgb_aug_nn_90.close();
    reorder_f_valid_score_xgb_aug_nn_90xs.close();
    reorder_f_valid_score_xgb_aug_nn_90ys.close();
    reorder_f_valid_score_xgb_aug_nn_90xys.close();

    reorder_f_valid_score_xgb_aug_nn_63.close();
    reorder_f_valid_score_xgb_aug_nn_63xs.close();
    reorder_f_valid_score_xgb_aug_nn_63ys.close();
    reorder_f_valid_score_xgb_aug_nn_63xys.close();


    reorder_f_valid_score_knn_new2_20.close();
    reorder_f_valid_score_knn_count_20.close();
    // reorder_f_valid_score_knn_20.close();
    // reorder_f_valid_score_knn_new2_20.close();   

    // reorder_f_valid_score_rf_100.close();
    // reorder_f_valid_score_rf_100xs.close();
    // reorder_f_valid_score_rf_100ys.close();
    // reorder_f_valid_score_rf_100xys.close();
    
    // reorder_f_valid_score_rf_71.close();
    // reorder_f_valid_score_rf_71xs.close();
    // reorder_f_valid_score_rf_71ys.close();
    // reorder_f_valid_score_rf_71xys.close();

    f_valid_test.close();

    // // f_valid_test_sub.close();
}
    /* end generate score from file */


void direct_check_validation(){
    int max_rank = 8, max_sub = 3;
    size_t num_valid_train = 23318975, num_valid_test = 5799046;
    double mapk_score = 0, weight = 1;
    unsigned int limit = 50;

    /* generate score directly from unordered file */

    ifstream f_valid_train("../valid/valid_train.csv");
    ifstream f_valid_test("../valid/valid_test.csv");

    // ifstream f_valid_score_71("../valid/valid_score_71_20160518180135.csv");
    // ifstream f_valid_score_100("../valid/valid_score_100_20160525192911.csv");
    // ifstream f_valid_score_100xs("../valid/valid_score_100xs_test.csv");
    // ifstream f_valid_score_100ys("../valid/valid_score_100ys_20160519162412.csv");
    // ifstream f_valid_score_100xys("../valid/valid_score_100xys_20160519162933.csv");
    // ifstream f_valid_score_100_season("../valid/valid_score_100_season_20160527012525.csv");
    // ifstream f_valid_score_100_test("../valid/valid_score_100_20160531142958.csv");
    // ifstream f_valid_score_90_test("../valid/valid_score_90_20160603145638.csv");

    
    // ifstream f_valid_score_rf_71("../valid/valid_score_rf_71_20160614110347.csv");
    // ifstream f_valid_score_rf_71xs("../valid/valid_score_rf_71xs_20160616230057.csv");
    // ifstream f_valid_score_rf_71ys("../valid/valid_score_rf_71ys_20160616230134.csv");
    // ifstream f_valid_score_rf_71xys("../valid/valid_score_rf_71xys_20160616230321.csv");


    // ifstream f_valid_score_rf_90("../valid/valid_score_90_20160615092133.csv");

    // ifstream f_valid_score_rf_100_10("../valid/valid_score_rf_100_20160613153324.csv");
    // ifstream f_valid_score_rf_100_15("../valid/valid_score_rf_100_20160613153427.csv");
    // ifstream f_valid_score_rf_100_5("../valid/valid_score_rf_100_20160613153658.csv");
    // ifstream f_valid_score_rf_100_120("../valid/valid_score_rf_100_20160613160320.csv");
    // ifstream f_valid_score_rf_100_avg("../valid/valid_score_rf_100_20160613164832.csv");

    // ifstream f_valid_score_rf_100("../valid/valid_score_rf_100_20160613153658.csv");
    // ifstream f_valid_score_rf_100xs("../valid/valid_score_rf_100xs_20160614120135.csv");
    // ifstream f_valid_score_rf_100ys("../valid/valid_score_rf_100ys_20160614120241.csv");
    // ifstream f_valid_score_rf_100xys("../valid/valid_score_rf_100xys_20160614120416.csv");

    // ifstream f_valid_score_71("../valid/valid_score_71_20160605004036.csv");
    // ifstream f_valid_score_71xs("../valid/valid_score_71xs_20160607234123.csv");
    // ifstream f_valid_score_71ys("../valid/valid_score_71ys_20160607234206.csv");
    // ifstream f_valid_score_71xys("../valid/valid_score_71xys_20160607234405.csv");

    // ifstream f_valid_score_90("../valid/valid_score_90_20160608142409.csv");
    // ifstream f_valid_score_90xs("../valid/valid_score_90xs_20160608143143.csv");
    // ifstream f_valid_score_90ys("../valid/valid_score_90ys_20160609202512.csv");
    // ifstream f_valid_score_90xys("../valid/valid_score_90xys_20160609202658.csv");

    // // ifstream f_valid_score_test("../valid/valid_score_90_20160608142409.csv");

    // ifstream f_valid_score_avg_100("../valid/valid_score_xgb_avg_100_20160628181414.csv");

    // ifstream f_valid_score_100("../valid/valid_score_100_20160605000504.csv");
    // ifstream f_valid_score_100xs("../valid/valid_score_100xs_20160606172637.csv");
    // ifstream f_valid_score_100ys("../valid/valid_score_100ys_20160606172711.csv");
    // ifstream f_valid_score_100xys("../valid/valid_score_100xys_20160607130744.csv");
    
    // ifstream f_valid_score_knn_100("../valid/valid_score_knn_100_20160621112130.csv");

    // ifstream f_valid_score_knn_20("../valid/valid_score_knn_20_20160701024005.csv");
    // ifstream f_valid_score_knn_new_20("../valid/valid_score_knn_20_20160701141005.csv");
    // ifstream f_valid_score_knn_new2_20("../valid/valid_score_knn_20_20160701170046.csv");    
    ifstream f_valid_score_knn_vn_20("../valid/valid_score_knn_vn_20_20160701165732.csv");
    ifstream f_valid_score_knn_count_20("../valid/valid_score_knn_count_20_20160706045201.csv");

    // ifstream f_valid_score_xgb_new_100("../valid/valid_score_xgb_new_100_20160622102600.csv");

    // ifstream f_valid_score_xgb_aug_nn_100("../valid/valid_score_xgb_100_20160702160701.csv");
    // ifstream f_valid_score_xgb_aug_nn_100xs("../valid/valid_score_xgb_aug_nn_100xs_20160702161826.csv");
    // ifstream f_valid_score_xgb_aug_nn_100ys("../valid/valid_score_xgb_aug_nn_100ys_20160702162024.csv");
    // ifstream f_valid_score_xgb_aug_nn_100xys("../valid/valid_score_xgb_aug_nn_100xys_20160702162115.csv");


    // ifstream f_valid_score_xgb_new2_100("../valid/valid_score_xgb_new2_100_20160622121911.csv");
    // ifstream f_valid_score_xgb_new2_100xs("../valid/valid_score_xgb_new2_100xs_20160623205834.csv");
    // ifstream f_valid_score_xgb_new2_100ys("../valid/valid_score_xgb_new2_100ys_20160623205919.csv");
    // ifstream f_valid_score_xgb_new2_100xys("../valid/valid_score_xgb_new2_100xys_20160623210050.csv");

    // ifstream f_valid_score_xgb_new2_90("../valid/valid_score_xgb_new2_90_20160623210124.csv");
    // ifstream f_valid_score_xgb_new2_90xs("../valid/valid_score_xgb_new2_90xs_20160624102803.csv");
    // ifstream f_valid_score_xgb_new2_90ys("../valid/valid_score_xgb_new2_90ys_20160623210225.csv");
    // ifstream f_valid_score_xgb_new2_90xys("../valid/valid_score_xgb_new2_90xys_20160623210250.csv");

    // ifstream f_valid_score_xgb_new2_63("../valid/valid_score_xgb_new2_63_20160627123102.csv");
    // ifstream f_valid_score_xgb_new2_63xs("../valid/valid_score_xgb_new2_63xs_20160627123139.csv");
    // ifstream f_valid_score_xgb_new2_63ys("../valid/valid_score_xgb_new2_63ys_20160627123204.csv");
    // ifstream f_valid_score_xgb_new2_63xys("../valid/valid_score_xgb_new2_63xys_20160627123228.csv");

    // if (!f_valid_train.is_open() || !f_valid_test.is_open() || !f_valid_score_71.is_open() || !f_valid_score_100.is_open() ||
    //  !f_valid_score_100xs.is_open() || !f_valid_score_100ys.is_open() || !f_valid_score_100xys.is_open()) return;

    /* load the necessary */
    vector<unordered_map<unsigned long, double> > score(num_valid_test);
    unordered_map<unsigned int, unsigned int> row_id_map;
    const clock_t begin_time = clock();

    generate_row_id_map(f_valid_test, row_id_map);
    /* end load */ 

    file_update_score(f_valid_score_knn_count_20, weight, score, row_id_map);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    file_update_score(f_valid_score_knn_vn_20, weight, score, row_id_map);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    file_update_score(f_valid_score_knn_count_20, 0.5 * weight, score, row_id_map);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;

    file_update_score(f_valid_score_knn_count_20, 0.5 * weight, score, row_id_map);

    mapk_score = validate_apk(f_valid_test, max_sub, score);

    cout << mapk_score << endl;    


    f_valid_score_knn_count_20.close();
    f_valid_score_knn_vn_20.close();

    // file_update_score(f_valid_score_xgb_aug_nn_100, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_xgb_aug_nn_100xs, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_xgb_aug_nn_100ys, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_xgb_aug_nn_100xys, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;        


    // f_valid_score_xgb_aug_nn_100.close();
    // f_valid_score_xgb_aug_nn_100xs.close();
    // f_valid_score_xgb_aug_nn_100ys.close();
    // f_valid_score_xgb_aug_nn_100xys.close();


    // file_update_score(f_valid_score_knn_20, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_knn_20, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_knn_20, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // f_valid_score_xgb_new2_63.close();
    // f_valid_score_xgb_new2_63xs.close();
    // f_valid_score_xgb_new2_63ys.close();
    // f_valid_score_xgb_new2_63xys.close();
    // f_valid_score_knn_20.close();

    // file_update_score(f_valid_score_knn_20, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // f_valid_score_knn_20.close();

    // file_update_score(f_valid_score_100, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_xgb_new2_100, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_xgb_new_100, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;
    // score = vector<unordered_map<unsigned long, double> > (num_valid_test);

    // file_update_score(f_valid_score_avg_100, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_knn_100, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;   
    
    // f_valid_score_knn_100.close(); 

    // file_update_score(f_valid_score_rf_71, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // generate_row_id_map(f_valid_test, row_id_map);

    // file_update_score(f_valid_score_rf_90, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_100, 2.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_100xs, 2.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_100ys, 2.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_100xys, 2.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_71, 2.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_71xs, 2.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_71ys, 2.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_71xys, 2.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_71, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_71xs, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_71ys, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_71xys, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_71, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_71xs, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_71ys, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_rf_71xys, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_90, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_90xs, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_90ys, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_90xys, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;    

    // file_update_score(f_valid_score_90, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_90, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_90, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_90, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_90, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_100xs, 2 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_100xs, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_100xs, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_100xs, 0.5 * weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_100, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_100xs, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_100ys, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_100xys, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_71, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_71xs, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_71ys, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_71xys, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_100xs, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_100ys, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // file_update_score(f_valid_score_100xys, weight, score, row_id_map);

    // mapk_score = validate_apk(f_valid_test, max_sub, score);

    // cout << mapk_score << endl;

    // cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

    // f_valid_score_rf_71.close();
    // f_valid_score_rf_71xs.close();
    // f_valid_score_rf_71ys.close();
    // f_valid_score_rf_71xys.close();

    // f_valid_score_rf_100.close();
    // f_valid_score_rf_100xs.close();
    // f_valid_score_rf_100ys.close();
    // f_valid_score_rf_100xys.close();

    // f_valid_score_rf_90.close();
    // f_valid_score_rf_100.close();
    // f_valid_score_rf_100xs.close();

    // f_valid_score_71.close();
    // f_valid_score_71xs.close();
    // f_valid_score_71ys.close();
    // f_valid_score_71xys.close();
    // f_valid_score_90.close();
    // f_valid_score_90xs.close();
    // f_valid_score_90ys.close();
    // f_valid_score_90xys.close();
    // f_valid_score_100.close();
    // f_valid_score_100xs.close();
    // f_valid_score_100ys.close();
    // f_valid_score_100xys.close();

    // f_valid_score_100_test.close();
    // f_valid_score_90_test.close();
    // f_valid_score_test.close();
    // f_valid_score_test_xs.close();
    // f_valid_score_test_ys.close();
    // f_valid_score_test_xys.close();
    f_valid_train.close();
    f_valid_test.close();
}

void reorder_test(){
    int max_rank = 8, max_sub = 3;
    size_t num_train = 29118021, num_test = 8607230;
    double mapk_score = 0, weight = 1;

    cout << "reorder_test" << endl;

    /* reorder row_id */
    // ifstream f_test_score_100("../train/test_score_100_20160605000504.csv");
    // ifstream f_test_score_100xs("../train/test_score_100xs_20160606172637.csv");
    // ifstream f_test_score_100ys("../train/test_score_100ys_20160606172711.csv");
    // ifstream f_test_score_100xys("../train/test_score_100xys_20160607130744.csv");

    // ofstream f_test_score_100_reorder("../train/reorder_test_score_100_20160605000504.csv");
    // ofstream f_test_score_100xs_reorder("../train/reorder_test_score_100xs_20160606172637.csv");
    // ofstream f_test_score_100ys_reorder("../train/reorder_test_score_100ys_20160606172711.csv");
    // ofstream f_test_score_100xys_reorder("../train/reorder_test_score_100xys_20160607130744.csv");

    // reorder_score(f_test_score_100, f_test_score_100_reorder);
    // reorder_score(f_test_score_100xs, f_test_score_100xs_reorder);
    // reorder_score(f_test_score_100ys, f_test_score_100ys_reorder);
    // reorder_score(f_test_score_100xys, f_test_score_100xys_reorder);

    // f_test_score_100.close();
    // f_test_score_100xs.close();
    // f_test_score_100ys.close();
    // f_test_score_100xys.close();
    // f_test_score_100_reorder.close();
    // f_test_score_100xs_reorder.close();
    // f_test_score_100ys_reorder.close();
    // f_test_score_100xys_reorder.close();

    // ifstream f_test_score_71("../train/test_score_71_20160605004036.csv");
    // ifstream f_test_score_71xs("../train/test_score_71xs_20160607234123.csv");
    // ifstream f_test_score_71ys("../train/test_score_71ys_20160607234206.csv");
    // ifstream f_test_score_71xys("../train/test_score_71xys_20160607234405.csv");

    // ofstream f_test_score_71_reorder("../train/reorder_test_score_71_20160605004036.csv");
    // ofstream f_test_score_71xs_reorder("../train/reorder_test_score_71xs_20160607234123.csv");
    // ofstream f_test_score_71ys_reorder("../train/reorder_test_score_71ys_20160607234206.csv");
    // ofstream f_test_score_71xys_reorder("../train/reorder_test_score_71xys_20160607234405.csv");

    // reorder_score(f_test_score_71, f_test_score_71_reorder);
    // reorder_score(f_test_score_71xs, f_test_score_71xs_reorder);
    // reorder_score(f_test_score_71ys, f_test_score_71ys_reorder);
    // reorder_score(f_test_score_71xys, f_test_score_71xys_reorder);

    // f_test_score_71.close();
    // f_test_score_71xs.close();
    // f_test_score_71ys.close();
    // f_test_score_71xys.close();
    // f_test_score_71_reorder.close();
    // f_test_score_71xs_reorder.close();
    // f_test_score_71ys_reorder.close();
    // f_test_score_71xys_reorder.close();

    // ifstream f_test_score_90("../train/test_score_90_20160608142409.csv");
    // ifstream f_test_score_90xs("../train/test_score_90xs_20160608143143.csv");
    // ifstream f_test_score_90ys("../train/test_score_90ys_20160609202512.csv");
    // ifstream f_test_score_90xys("../train/test_score_90xys_20160609202658.csv");

    // ofstream f_test_score_90_reorder("../train/reorder_test_score_90_20160608142409.csv");
    // ofstream f_test_score_90xs_reorder("../train/reorder_test_score_90xs_20160608143143.csv");
    // ofstream f_test_score_90ys_reorder("../train/reorder_test_score_90ys_20160609202512.csv");
    // ofstream f_test_score_90xys_reorder("../train/reorder_test_score_90xys_20160609202658.csv");

    // reorder_score(f_test_score_90, f_test_score_90_reorder);
    // reorder_score(f_test_score_90xs, f_test_score_90xs_reorder);
    // reorder_score(f_test_score_90ys, f_test_score_90ys_reorder);
    // reorder_score(f_test_score_90xys, f_test_score_90xys_reorder);

    // f_test_score_90.close();
    // f_test_score_90xs.close();
    // f_test_score_90ys.close();
    // f_test_score_90xys.close();
    // f_test_score_90_reorder.close();
    // f_test_score_90xs_reorder.close();
    // f_test_score_90ys_reorder.close();
    // f_test_score_90xys_reorder.close();

    // ifstream f_test_score_63("../train/test_score_63_20160610165455.csv");
    // ifstream f_test_score_63xs("../train/test_score_63xs_20160610165529.csv");
    // ifstream f_test_score_63ys("../train/test_score_63ys_20160610165711.csv");
    // ifstream f_test_score_63xys("../train/test_score_63xys_20160610171448.csv");

    // ofstream f_test_score_63_reorder("../train/reorder_test_score_63_20160610165455.csv");
    // ofstream f_test_score_63xs_reorder("../train/reorder_test_score_63xs_20160610165529.csv");
    // ofstream f_test_score_63ys_reorder("../train/reorder_test_score_63ys_20160610165711.csv");
    // ofstream f_test_score_63xys_reorder("../train/reorder_test_score_63xys_20160610171448.csv");

    // reorder_score(f_test_score_63, f_test_score_63_reorder);
    // reorder_score(f_test_score_63xs, f_test_score_63xs_reorder);
    // reorder_score(f_test_score_63ys, f_test_score_63ys_reorder);
    // reorder_score(f_test_score_63xys, f_test_score_63xys_reorder);

    // f_test_score_63.close();
    // f_test_score_63xs.close();
    // f_test_score_63ys.close();
    // f_test_score_63xys.close();
    // f_test_score_63_reorder.close();
    // f_test_score_63xs_reorder.close();
    // f_test_score_63ys_reorder.close();
    // f_test_score_63xys_reorder.close();

    // // ifstream f_test_score_83("../train/test_score_83_20160621012105.csv");
    // // ifstream f_test_score_83xs("../train/test_score_83xs_20160621012258.csv");
    // // ifstream f_test_score_83ys("../train/test_score_83ys_20160621012335.csv");
    // // ifstream f_test_score_83xys("../train/test_score_83xys_20160621012424.csv");

    // // ofstream f_test_score_83_reorder("../train/reorder_test_score_83_20160621012105.csv");
    // // ofstream f_test_score_83xs_reorder("../train/reorder_test_score_83xs_20160621012258.csv");
    // // ofstream f_test_score_83ys_reorder("../train/reorder_test_score_83ys_20160621012335.csv");
    // // ofstream f_test_score_83xys_reorder("../train/reorder_test_score_83xys_20160621012424.csv");

    // // reorder_score(f_test_score_83, f_test_score_83_reorder);
    // // reorder_score(f_test_score_83xs, f_test_score_83xs_reorder);
    // // reorder_score(f_test_score_83ys, f_test_score_83ys_reorder);
    // // reorder_score(f_test_score_83xys, f_test_score_83xys_reorder);

    // // f_test_score_83.close();
    // // f_test_score_83xs.close();
    // // f_test_score_83ys.close();
    // // f_test_score_83xys.close();
    // // f_test_score_83_reorder.close();
    // // f_test_score_83xs_reorder.close();
    // // f_test_score_83ys_reorder.close();
    // // f_test_score_83xys_reorder.close();


    // ifstream f_test_score_xgb_new2_100("../train/test_score_xgb_new2_100_20160622121911.csv");
    // ifstream f_test_score_xgb_new2_100xs("../train/test_score_xgb_new2_100xs_20160623205834.csv");
    // ifstream f_test_score_xgb_new2_100ys("../train/test_score_xgb_new2_100ys_20160623205919.csv");
    // ifstream f_test_score_xgb_new2_100xys("../train/test_score_xgb_new2_100xys_20160623210050.csv");

    // ofstream f_test_score_xgb_new2_100_reorder("../train/reorder_test_score_xgb_new2_100_20160622121911.csv");
    // ofstream f_test_score_xgb_new2_100xs_reorder("../train/reorder_test_score_xgb_new2_100xs_20160623205834.csv");
    // ofstream f_test_score_xgb_new2_100ys_reorder("../train/reorder_test_score_xgb_new2_100ys_20160623205919.csv");
    // ofstream f_test_score_xgb_new2_100xys_reorder("../train/reorder_test_score_xgb_new2_100xys_20160623210050.csv");

    // reorder_score(f_test_score_xgb_new2_100, f_test_score_xgb_new2_100_reorder);
    // reorder_score(f_test_score_xgb_new2_100xs, f_test_score_xgb_new2_100xs_reorder);
    // reorder_score(f_test_score_xgb_new2_100ys, f_test_score_xgb_new2_100ys_reorder);
    // reorder_score(f_test_score_xgb_new2_100xys, f_test_score_xgb_new2_100xys_reorder);

    // f_test_score_xgb_new2_100.close();
    // f_test_score_xgb_new2_100xs.close();
    // f_test_score_xgb_new2_100ys.close();
    // f_test_score_xgb_new2_100xys.close();
    // f_test_score_xgb_new2_100_reorder.close();
    // f_test_score_xgb_new2_100xs_reorder.close();
    // f_test_score_xgb_new2_100ys_reorder.close();
    // f_test_score_xgb_new2_100xys_reorder.close();


    // ifstream f_test_score_xgb_new2_71("../train/test_score_xgb_new2_71_20160627122639.csv");
    // ifstream f_test_score_xgb_new2_71xs("../train/test_score_xgb_new2_71xs_20160627122713.csv");
    // ifstream f_test_score_xgb_new2_71ys("../train/test_score_xgb_new2_71ys_20160627122814.csv");
    // ifstream f_test_score_xgb_new2_71xys("../train/test_score_xgb_new2_71xys_20160627122942.csv");

    // ofstream f_test_score_xgb_new2_71_reorder("../train/reorder_test_score_xgb_new2_71_20160627122639.csv");
    // ofstream f_test_score_xgb_new2_71xs_reorder("../train/reorder_test_score_xgb_new2_71xs_20160627122713.csv");
    // ofstream f_test_score_xgb_new2_71ys_reorder("../train/reorder_test_score_xgb_new2_71ys_20160627122814.csv");
    // ofstream f_test_score_xgb_new2_71xys_reorder("../train/reorder_test_score_xgb_new2_71xys_20160627122942.csv");

    // reorder_score(f_test_score_xgb_new2_71, f_test_score_xgb_new2_71_reorder);
    // reorder_score(f_test_score_xgb_new2_71xs, f_test_score_xgb_new2_71xs_reorder);
    // reorder_score(f_test_score_xgb_new2_71ys, f_test_score_xgb_new2_71ys_reorder);
    // reorder_score(f_test_score_xgb_new2_71xys, f_test_score_xgb_new2_71xys_reorder);

    // f_test_score_xgb_new2_71.close();
    // f_test_score_xgb_new2_71xs.close();
    // f_test_score_xgb_new2_71ys.close();
    // f_test_score_xgb_new2_71xys.close();
    // f_test_score_xgb_new2_71_reorder.close();
    // f_test_score_xgb_new2_71xs_reorder.close();
    // f_test_score_xgb_new2_71ys_reorder.close();
    // f_test_score_xgb_new2_71xys_reorder.close();


    // ifstream f_test_score_xgb_new2_90("../train/test_score_xgb_new2_90_20160623210124.csv");
    // ifstream f_test_score_xgb_new2_90xs("../train/test_score_xgb_new2_90xs_20160624102803.csv");
    // ifstream f_test_score_xgb_new2_90ys("../train/test_score_xgb_new2_90ys_20160623210225.csv");
    // ifstream f_test_score_xgb_new2_90xys("../train/test_score_xgb_new2_90xys_20160623210250.csv");

    // ofstream f_test_score_xgb_new2_90_reorder("../train/reorder_test_score_xgb_new2_90_20160623210124.csv");
    // ofstream f_test_score_xgb_new2_90xs_reorder("../train/reorder_test_score_xgb_new2_90xs_20160624102803.csv");
    // ofstream f_test_score_xgb_new2_90ys_reorder("../train/reorder_test_score_xgb_new2_90ys_20160623210225.csv");
    // ofstream f_test_score_xgb_new2_90xys_reorder("../train/reorder_test_score_xgb_new2_90xys_20160623210250.csv");

    // reorder_score(f_test_score_xgb_new2_90, f_test_score_xgb_new2_90_reorder);
    // reorder_score(f_test_score_xgb_new2_90xs, f_test_score_xgb_new2_90xs_reorder);
    // reorder_score(f_test_score_xgb_new2_90ys, f_test_score_xgb_new2_90ys_reorder);
    // reorder_score(f_test_score_xgb_new2_90xys, f_test_score_xgb_new2_90xys_reorder);

    // f_test_score_xgb_new2_90.close();
    // f_test_score_xgb_new2_90xs.close();
    // f_test_score_xgb_new2_90ys.close();
    // f_test_score_xgb_new2_90xys.close();
    // f_test_score_xgb_new2_90_reorder.close();
    // f_test_score_xgb_new2_90xs_reorder.close();
    // f_test_score_xgb_new2_90ys_reorder.close();
    // f_test_score_xgb_new2_90xys_reorder.close();


    // ifstream f_test_score_xgb_new2_63("../train/test_score_xgb_new2_63_20160627123102.csv");
    // ifstream f_test_score_xgb_new2_63xs("../train/test_score_xgb_new2_63xs_20160627123139.csv");
    // ifstream f_test_score_xgb_new2_63ys("../train/test_score_xgb_new2_63ys_20160627123204.csv");
    // ifstream f_test_score_xgb_new2_63xys("../train/test_score_xgb_new2_63xys_20160627123228.csv");

    // ofstream f_test_score_xgb_new2_63_reorder("../train/reorder_test_score_xgb_new2_63_20160627123102.csv");
    // ofstream f_test_score_xgb_new2_63xs_reorder("../train/reorder_test_score_xgb_new2_63xs_20160627123139.csv");
    // ofstream f_test_score_xgb_new2_63ys_reorder("../train/reorder_test_score_xgb_new2_63ys_20160627123204.csv");
    // ofstream f_test_score_xgb_new2_63xys_reorder("../train/reorder_test_score_xgb_new2_63xys_20160627123228.csv");

    // reorder_score(f_test_score_xgb_new2_63, f_test_score_xgb_new2_63_reorder);
    // reorder_score(f_test_score_xgb_new2_63xs, f_test_score_xgb_new2_63xs_reorder);
    // reorder_score(f_test_score_xgb_new2_63ys, f_test_score_xgb_new2_63ys_reorder);
    // reorder_score(f_test_score_xgb_new2_63xys, f_test_score_xgb_new2_63xys_reorder);

    // f_test_score_xgb_new2_63.close();
    // f_test_score_xgb_new2_63xs.close();
    // f_test_score_xgb_new2_63ys.close();
    // f_test_score_xgb_new2_63xys.close();
    // f_test_score_xgb_new2_63_reorder.close();
    // f_test_score_xgb_new2_63xs_reorder.close();
    // f_test_score_xgb_new2_63ys_reorder.close();
    // f_test_score_xgb_new2_63xys_reorder.close();


    // ifstream f_test_score_xgb_aug_nn_100("../train/test_score_xgb_100_20160702160701.csv");
    // ifstream f_test_score_xgb_aug_nn_100xs("../train/test_score_xgb_aug_nn_100xs_20160702161826.csv");
    // ifstream f_test_score_xgb_aug_nn_100ys("../train/test_score_xgb_aug_nn_100ys_20160702162024.csv");
    // ifstream f_test_score_xgb_aug_nn_100xys("../train/test_score_xgb_aug_nn_100xys_20160702162115.csv");

    // ofstream f_test_score_xgb_aug_nn_100_reorder("../train/reorder_test_score_xgb_100_20160702160701.csv");
    // ofstream f_test_score_xgb_aug_nn_100xs_reorder("../train/reorder_test_score_xgb_aug_nn_100xs_20160702161826.csv");
    // ofstream f_test_score_xgb_aug_nn_100ys_reorder("../train/reorder_test_score_xgb_aug_nn_100ys_20160702162024.csv");
    // ofstream f_test_score_xgb_aug_nn_100xys_reorder("../train/reorder_test_score_xgb_aug_nn_100xys_20160702162115.csv");    

    // reorder_score(f_test_score_xgb_aug_nn_100, f_test_score_xgb_aug_nn_100_reorder);
    // reorder_score(f_test_score_xgb_aug_nn_100xs, f_test_score_xgb_aug_nn_100xs_reorder);
    // reorder_score(f_test_score_xgb_aug_nn_100ys, f_test_score_xgb_aug_nn_100ys_reorder);
    // reorder_score(f_test_score_xgb_aug_nn_100xys, f_test_score_xgb_aug_nn_100xys_reorder);

    // f_test_score_xgb_aug_nn_100.close();
    // f_test_score_xgb_aug_nn_100xs.close();
    // f_test_score_xgb_aug_nn_100ys.close();
    // f_test_score_xgb_aug_nn_100xys.close();
    // f_test_score_xgb_aug_nn_100_reorder.close();
    // f_test_score_xgb_aug_nn_100xs_reorder.close();
    // f_test_score_xgb_aug_nn_100ys_reorder.close();
    // f_test_score_xgb_aug_nn_100xys_reorder.close();

    // ifstream f_test_score_xgb_aug_nn_90("../train/test_score_xgb_aug_nn_90_20160703214350.csv");
    // ifstream f_test_score_xgb_aug_nn_90xs("../train/test_score_xgb_aug_nn_90xs_20160703214507.csv");
    // ifstream f_test_score_xgb_aug_nn_90ys("../train/test_score_xgb_aug_nn_90ys_20160703214623.csv");
    // ifstream f_test_score_xgb_aug_nn_90xys("../train/test_score_xgb_aug_nn_90xys_20160703214813.csv");

    // ofstream f_test_score_xgb_aug_nn_90_reorder("../train/reorder_test_score_xgb_aug_nn_90_20160703214350.csv");
    // ofstream f_test_score_xgb_aug_nn_90xs_reorder("../train/reorder_test_score_xgb_aug_nn_90xs_20160703214507.csv");
    // ofstream f_test_score_xgb_aug_nn_90ys_reorder("../train/reorder_test_score_xgb_aug_nn_90ys_20160703214623.csv");
    // ofstream f_test_score_xgb_aug_nn_90xys_reorder("../train/reorder_test_score_xgb_aug_nn_90xys_20160703214813.csv");    

    // reorder_score(f_test_score_xgb_aug_nn_90, f_test_score_xgb_aug_nn_90_reorder);
    // reorder_score(f_test_score_xgb_aug_nn_90xs, f_test_score_xgb_aug_nn_90xs_reorder);
    // reorder_score(f_test_score_xgb_aug_nn_90ys, f_test_score_xgb_aug_nn_90ys_reorder);
    // reorder_score(f_test_score_xgb_aug_nn_90xys, f_test_score_xgb_aug_nn_90xys_reorder);

    // f_test_score_xgb_aug_nn_90.close();
    // f_test_score_xgb_aug_nn_90xs.close();
    // f_test_score_xgb_aug_nn_90ys.close();
    // f_test_score_xgb_aug_nn_90xys.close();
    // f_test_score_xgb_aug_nn_90_reorder.close();
    // f_test_score_xgb_aug_nn_90xs_reorder.close();
    // f_test_score_xgb_aug_nn_90ys_reorder.close();
    // f_test_score_xgb_aug_nn_90xys_reorder.close();

// -rw-rw-r-- 1 rzhang rzhang 3510638748 Jul  6 06:33 test_score_xgb_aug_nn_63_20160702172227.csv
// -rw-rw-r-- 1 rzhang rzhang  654397713 Jul  6 08:27 test_score_xgb_aug_nn_63_20160702172227.csv.reverse
// -rw-rw-r-- 1 rzhang rzhang 3510638748 Jul  6 06:02 test_score_xgb_aug_nn_63xs_20160702172327.csv
// -rw-rw-r-- 1 rzhang rzhang  626842644 Jul  6 08:09 test_score_xgb_aug_nn_63xs_20160702172327.csv.reverse
// -rw-rw-r-- 1 rzhang rzhang 2620207967 Jul  5 12:42 test_score_xgb_aug_nn_63xys_20160702172549.csv
// -rw-rw-r-- 1 rzhang rzhang 3510638748 Jul  6 11:19 test_score_xgb_aug_nn_63xys_20160702172549.csv.cp
// -rw-rw-r-- 1 rzhang rzhang  626842644 Jul  6 08:17 test_score_xgb_aug_nn_63xys_20160702172549.csv.reverse
// -rw-rw-r-- 1 rzhang rzhang 3510638748 Jul  6 07:47 test_score_xgb_aug_nn_63ys_20160702172433.csv
// -rw-rw-r-- 1 rzhang rzhang  654397713 Jul  6 08:21 test_score_xgb_aug_nn_63ys_20160702172433.csv.reverse
// -rw-rw-r-- 1 rzhang rzhang 3510638748 Jul  6 09:44 test_score_xgb_aug_nn_71_20160702165843.csv
// -rw-rw-r-- 1 rzhang rzhang  776813135 Jul  6 11:49 test_score_xgb_aug_nn_71_20160702165843.csv.reverse
// -rw-rw-r-- 1 rzhang rzhang 1828561751 Jul  4 15:57 test_score_xgb_aug_nn_71xs_20160702165921.csv
// -rw-rw-r-- 1 rzhang rzhang 3510638748 Jul  6 12:03 test_score_xgb_aug_nn_71xs_20160702165921.csv.cp
// -rw-rw-r-- 1 rzhang rzhang  752809416 Jul  6 11:35 test_score_xgb_aug_nn_71xs_20160702165921.csv.reverse
// -rw-rw-r-- 1 rzhang rzhang 3510638748 Jul  6 11:35 test_score_xgb_aug_nn_71xys_20160702170420.csv
// -rw-rw-r-- 1 rzhang rzhang  752809416 Jul  6 11:48 test_score_xgb_aug_nn_71xys_20160702170420.csv.reverse
// -rw-rw-r-- 1 rzhang rzhang 3510638748 Jul  6 11:01 test_score_xgb_aug_nn_71ys_20160702170115.csv
// -rw-rw-r-- 1 rzhang rzhang  776813135 Jul  6 12:22 test_score_xgb_aug_nn_71ys_20160702170115.csv.reverse

    // ifstream f_test_score_xgb_aug_nn_71("../train/test_score_xgb_aug_nn_71_20160702165843.csv");
    // ifstream f_test_score_xgb_aug_nn_71xs("../train/test_score_xgb_aug_nn_71xs_20160702165921.csv.cp");
    // ifstream f_test_score_xgb_aug_nn_71ys("../train/test_score_xgb_aug_nn_71ys_20160702170115.csv");
    // ifstream f_test_score_xgb_aug_nn_71xys("../train/test_score_xgb_aug_nn_71xys_20160702170420.csv");

    // ofstream f_test_score_xgb_aug_nn_71_reorder("../train/reorder_test_score_xgb_aug_nn_71_20160702165843.csv");
    // ofstream f_test_score_xgb_aug_nn_71xs_reorder("../train/reorder_test_score_xgb_aug_nn_71xs_20160702165921.csv");
    // ofstream f_test_score_xgb_aug_nn_71ys_reorder("../train/reorder_test_score_xgb_aug_nn_71ys_20160702170115.csv");
    // ofstream f_test_score_xgb_aug_nn_71xys_reorder("../train/reorder_test_score_xgb_aug_nn_71xys_20160702170420.csv");    

    // reorder_score(f_test_score_xgb_aug_nn_71, f_test_score_xgb_aug_nn_71_reorder);
    // reorder_score(f_test_score_xgb_aug_nn_71xs, f_test_score_xgb_aug_nn_71xs_reorder);
    // reorder_score(f_test_score_xgb_aug_nn_71ys, f_test_score_xgb_aug_nn_71ys_reorder);
    // reorder_score(f_test_score_xgb_aug_nn_71xys, f_test_score_xgb_aug_nn_71xys_reorder);

    // f_test_score_xgb_aug_nn_71.close();
    // f_test_score_xgb_aug_nn_71xs.close();
    // f_test_score_xgb_aug_nn_71ys.close();
    // f_test_score_xgb_aug_nn_71xys.close();
    // f_test_score_xgb_aug_nn_71_reorder.close();
    // f_test_score_xgb_aug_nn_71xs_reorder.close();
    // f_test_score_xgb_aug_nn_71ys_reorder.close();
    // f_test_score_xgb_aug_nn_71xys_reorder.close();

    // ifstream f_test_score_xgb_aug_nn_63("../train/test_score_xgb_aug_nn_63_20160702172227.csv");
    // ifstream f_test_score_xgb_aug_nn_63xs("../train/test_score_xgb_aug_nn_63xs_20160702172327.csv");
    // ifstream f_test_score_xgb_aug_nn_63ys("../train/test_score_xgb_aug_nn_63ys_20160702172433.csv");
    // ifstream f_test_score_xgb_aug_nn_63xys("../train/test_score_xgb_aug_nn_63xys_20160702172549.csv.cp");

    // ofstream f_test_score_xgb_aug_nn_63_reorder("../train/reorder_test_score_xgb_aug_nn_63_20160702172227.csv");
    // ofstream f_test_score_xgb_aug_nn_63xs_reorder("../train/reorder_test_score_xgb_aug_nn_63xs_20160702172327.csv");
    // ofstream f_test_score_xgb_aug_nn_63ys_reorder("../train/reorder_test_score_xgb_aug_nn_63ys_20160702172433.csv");
    // ofstream f_test_score_xgb_aug_nn_63xys_reorder("../train/reorder_test_score_xgb_aug_nn_63xys_20160702172549.csv");    

    // reorder_score(f_test_score_xgb_aug_nn_63, f_test_score_xgb_aug_nn_63_reorder);
    // reorder_score(f_test_score_xgb_aug_nn_63xs, f_test_score_xgb_aug_nn_63xs_reorder);
    // reorder_score(f_test_score_xgb_aug_nn_63ys, f_test_score_xgb_aug_nn_63ys_reorder);
    // reorder_score(f_test_score_xgb_aug_nn_63xys, f_test_score_xgb_aug_nn_63xys_reorder);

    // f_test_score_xgb_aug_nn_63.close();
    // f_test_score_xgb_aug_nn_63xs.close();
    // f_test_score_xgb_aug_nn_63ys.close();
    // f_test_score_xgb_aug_nn_63xys.close();
    // f_test_score_xgb_aug_nn_63_reorder.close();
    // f_test_score_xgb_aug_nn_63xs_reorder.close();
    // f_test_score_xgb_aug_nn_63ys_reorder.close();
    // f_test_score_xgb_aug_nn_63xys_reorder.close();

    ifstream f_test_score_knn_count_20("../train/test_score_knn_count_20_20160706044948.csv");
    ofstream f_test_score_knn_count_20_reorder("../train/reorder_test_score_knn_count_20_20160706045201.csv");

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
    // ifstream f_valid_test("../valid/valid_test.csv");
    
    // ifstream f_reorder_score_100("../train/reorder_score_100.csv");
    // ifstream f_reorder_score_100xs("../train/reorder_score_100xs.csv");
    // ifstream f_reorder_score_100ys("../train/reorder_score_100ys.csv");
    // ifstream f_reorder_score_100xys("../train/reorder_score_100xys.csv");

    // ifstream f_reorder_score_71("../train/reorder_score_71.csv");
    // ifstream f_reorder_score_71xs("../train/reorder_score_71xs.csv");
    // ifstream f_reorder_score_71ys("../train/reorder_score_71ys.csv");
    // ifstream f_reorder_score_71xys("../train/reorder_score_71xys.csv");
    
    // ifstream f_reorder_score_83("../train/reorder_score_83.csv");
    // ifstream f_reorder_score_83xs("../train/reorder_score_83xs.csv");
    // ifstream f_reorder_score_83ys("../train/reorder_score_83ys.csv");
    // ifstream f_reorder_score_83xys("../train/reorder_score_83xys.csv");

    ofstream f_test_sub("../output/grid_all_sub_add_2.csv");

    // vector<double> weights(4, 1.0);

    // vector<ifstream*> f_reorder_scores_100;
    // f_reorder_scores_100.push_back(&f_reorder_score_100);
    // f_reorder_scores_100.push_back(&f_reorder_score_100xs);
    // f_reorder_scores_100.push_back(&f_reorder_score_100ys);
    // f_reorder_scores_100.push_back(&f_reorder_score_100xys);

    // // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_scores_100, weights, max_sub, num_valid_test);

    // // cout << mapk_score << endl;

    // vector<ifstream*> f_reorder_scores_71;
    // f_reorder_scores_71.push_back(&f_reorder_score_71);
    // f_reorder_scores_71.push_back(&f_reorder_score_71xs);
    // f_reorder_scores_71.push_back(&f_reorder_score_71ys);
    // f_reorder_scores_71.push_back(&f_reorder_score_71xys);

    // // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_scores_71, weights, max_sub, num_valid_test);

    // // cout << mapk_score << endl;

    // vector<ifstream*> f_reorder_scores_83;
    // f_reorder_scores_83.push_back(&f_reorder_score_83);
    // f_reorder_scores_83.push_back(&f_reorder_score_83xs);
    // f_reorder_scores_83.push_back(&f_reorder_score_83ys);
    // f_reorder_scores_83.push_back(&f_reorder_score_83xys);

    // // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_scores_83, weights, max_sub, num_valid_test);

    // // cout << mapk_score << endl;

    // // vector<double> all_weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0};   
    // vector<double> all_weights(12, 1.0);   
    // vector<ifstream*> f_reorder_scores(12);
    // copy(f_reorder_scores_100.begin(), f_reorder_scores_100.end(), f_reorder_scores.begin());
    // copy(f_reorder_scores_71.begin(), f_reorder_scores_71.end(), f_reorder_scores.begin()+4);
    // copy(f_reorder_scores_83.begin(), f_reorder_scores_83.end(), f_reorder_scores.begin()+8);

    // // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_scores, all_weights, max_sub, num_valid_test);

    // // cout << mapk_score << endl;


    // // mapk_score = file_generate_validate_apk(f_valid_test, f_valid_scores, weights, max_sub, num_valid_test);

    // // cout << mapk_score << endl;

    // file_generate_sub(f_test_sub, f_reorder_scores, all_weights, max_sub, num_test);

    ifstream reorder_f_test_score_100("../train/reorder_test_score_100_20160605000504.csv");
    ifstream reorder_f_test_score_100xs("../train/reorder_test_score_100xs_20160606172637.csv");
    ifstream reorder_f_test_score_100ys("../train/reorder_test_score_100ys_20160606172711.csv");
    ifstream reorder_f_test_score_100xys("../train/reorder_test_score_100xys_20160607130744.csv");

    ifstream reorder_f_test_score_71("../train/reorder_test_score_71_20160605004036.csv");
    ifstream reorder_f_test_score_71xs("../train/reorder_test_score_71xs_20160607234123.csv");
    ifstream reorder_f_test_score_71ys("../train/reorder_test_score_71ys_20160607234206.csv");
    ifstream reorder_f_test_score_71xys("../train/reorder_test_score_71xys_20160607234405.csv");

    ifstream reorder_f_test_score_90("../train/reorder_test_score_90_20160608142409.csv");
    ifstream reorder_f_test_score_90xs("../train/reorder_test_score_90xs_20160608143143.csv");
    ifstream reorder_f_test_score_90ys("../train/reorder_test_score_90ys_20160609202512.csv");
    ifstream reorder_f_test_score_90xys("../train/reorder_test_score_90xys_20160609202658.csv");

    ifstream reorder_f_test_score_63("../train/reorder_test_score_63_20160610165455.csv");
    ifstream reorder_f_test_score_63xs("../train/reorder_test_score_63xs_20160610165529.csv");
    ifstream reorder_f_test_score_63ys("../train/reorder_test_score_63ys_20160610165711.csv");
    ifstream reorder_f_test_score_63xys("../train/reorder_test_score_63xys_20160610171448.csv");

    ifstream reorder_f_test_score_83("../train/reorder_test_score_83_20160621012105.csv");
    ifstream reorder_f_test_score_83xs("../train/reorder_test_score_83xs_20160621012258.csv");
    ifstream reorder_f_test_score_83ys("../train/reorder_test_score_83ys_20160621012335.csv");
    ifstream reorder_f_test_score_83xys("../train/reorder_test_score_83xys_20160621012424.csv");  

    // ifstream reorder_f_test_score_knn_20("../train/reorder_test_score_knn_20_20160701024005.csv");
    // using knn below !

    ifstream reorder_f_test_score_xgb_new2_100("../train/reorder_test_score_xgb_new2_100_20160622121911.csv");
    ifstream reorder_f_test_score_xgb_new2_100xs("../train/reorder_test_score_xgb_new2_100xs_20160623205834.csv");
    ifstream reorder_f_test_score_xgb_new2_100ys("../train/reorder_test_score_xgb_new2_100ys_20160623205919.csv");
    ifstream reorder_f_test_score_xgb_new2_100xys("../train/reorder_test_score_xgb_new2_100xys_20160623210050.csv");

    ifstream reorder_f_test_score_xgb_new2_71("../train/reorder_test_score_xgb_new2_71_20160627122639.csv");
    ifstream reorder_f_test_score_xgb_new2_71xs("../train/reorder_test_score_xgb_new2_71xs_20160627122713.csv");
    ifstream reorder_f_test_score_xgb_new2_71ys("../train/reorder_test_score_xgb_new2_71ys_20160627122814.csv");
    ifstream reorder_f_test_score_xgb_new2_71xys("../train/reorder_test_score_xgb_new2_71xys_20160627122942.csv");

    ifstream reorder_f_test_score_xgb_new2_90("../train/reorder_test_score_xgb_new2_90_20160623210124.csv");
    ifstream reorder_f_test_score_xgb_new2_90xs("../train/reorder_test_score_xgb_new2_90xs_20160624102803.csv");
    ifstream reorder_f_test_score_xgb_new2_90ys("../train/reorder_test_score_xgb_new2_90ys_20160623210225.csv");
    ifstream reorder_f_test_score_xgb_new2_90xys("../train/reorder_test_score_xgb_new2_90xys_20160623210250.csv");

    ifstream reorder_f_test_score_xgb_new2_63("../train/reorder_test_score_xgb_new2_63_20160627123102.csv");
    ifstream reorder_f_test_score_xgb_new2_63xs("../train/reorder_test_score_xgb_new2_63xs_20160627123139.csv");
    ifstream reorder_f_test_score_xgb_new2_63ys("../train/reorder_test_score_xgb_new2_63ys_20160627123204.csv");
    ifstream reorder_f_test_score_xgb_new2_63xys("../train/reorder_test_score_xgb_new2_63xys_20160627123228.csv");    

    // aug nn
    ifstream reorder_f_test_score_xgb_aug_nn_100("../train/reorder_test_score_xgb_100_20160702160701.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_100xs("../train/reorder_test_score_xgb_aug_nn_100xs_20160702161826.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_100ys("../train/reorder_test_score_xgb_aug_nn_100ys_20160702162024.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_100xys("../train/reorder_test_score_xgb_aug_nn_100xys_20160702162115.csv");

    ifstream reorder_f_test_score_xgb_aug_nn_71("../train/reorder_test_score_xgb_aug_nn_71_20160702165843.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_71xs("../train/reorder_test_score_xgb_aug_nn_71xs_20160702165921.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_71ys("../train/reorder_test_score_xgb_aug_nn_71ys_20160702170115.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_71xys("../train/reorder_test_score_xgb_aug_nn_71xys_20160702170420.csv");  

    ifstream reorder_f_test_score_xgb_aug_nn_90("../train/reorder_test_score_xgb_aug_nn_90_20160703214350.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_90xs("../train/reorder_test_score_xgb_aug_nn_90xs_20160703214507.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_90ys("../train/reorder_test_score_xgb_aug_nn_90ys_20160703214623.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_90xys("../train/reorder_test_score_xgb_aug_nn_90xys_20160703214813.csv");  

    ifstream reorder_f_test_score_xgb_aug_nn_63("../train/reorder_test_score_xgb_aug_nn_63_20160702172227.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_63xs("../train/reorder_test_score_xgb_aug_nn_63xs_20160702172327.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_63ys("../train/reorder_test_score_xgb_aug_nn_63ys_20160702172433.csv");
    ifstream reorder_f_test_score_xgb_aug_nn_63xys("../train/reorder_test_score_xgb_aug_nn_63xys_20160702172549.csv");  

    // ifstream reorder_f_test_score_knn_new2_20("../train/reorder_test_score_knn_20_20160701170046.csv");  
    ifstream reorder_f_test_score_knn_count_20("../train/reorder_test_score_knn_count_20_20160706045201.csv");

    vector<double> weights_all(49, 1.0);
    vector<ifstream*> f_reorder_test_scores_all = {
    &reorder_f_test_score_xgb_new2_100, &reorder_f_test_score_xgb_new2_100xs, &reorder_f_test_score_xgb_new2_100ys, &reorder_f_test_score_xgb_new2_100xys,
    &reorder_f_test_score_xgb_new2_71, &reorder_f_test_score_xgb_new2_71xs, &reorder_f_test_score_xgb_new2_71ys, &reorder_f_test_score_xgb_new2_71xys,
    &reorder_f_test_score_xgb_new2_90, &reorder_f_test_score_xgb_new2_90xs, &reorder_f_test_score_xgb_new2_90ys, &reorder_f_test_score_xgb_new2_90xys,
    &reorder_f_test_score_xgb_new2_63, &reorder_f_test_score_xgb_new2_63xs, &reorder_f_test_score_xgb_new2_63ys, &reorder_f_test_score_xgb_new2_63xys,
    &reorder_f_test_score_100, &reorder_f_test_score_100xs, &reorder_f_test_score_100ys, &reorder_f_test_score_100xys,
    &reorder_f_test_score_71, &reorder_f_test_score_71xs, &reorder_f_test_score_71ys, &reorder_f_test_score_71xys,
    &reorder_f_test_score_90, &reorder_f_test_score_90xs, &reorder_f_test_score_90ys, &reorder_f_test_score_90xys,
    &reorder_f_test_score_63, &reorder_f_test_score_63xs, &reorder_f_test_score_63ys, &reorder_f_test_score_63xys,
    &reorder_f_test_score_xgb_aug_nn_100, &reorder_f_test_score_xgb_aug_nn_100xs, &reorder_f_test_score_xgb_aug_nn_100ys, &reorder_f_test_score_xgb_aug_nn_100xys,
    &reorder_f_test_score_xgb_aug_nn_71, &reorder_f_test_score_xgb_aug_nn_71xs, &reorder_f_test_score_xgb_aug_nn_71ys, &reorder_f_test_score_xgb_aug_nn_71xys,
    &reorder_f_test_score_xgb_aug_nn_90, &reorder_f_test_score_xgb_aug_nn_90xs, &reorder_f_test_score_xgb_aug_nn_90ys, &reorder_f_test_score_xgb_aug_nn_90xys,
    &reorder_f_test_score_xgb_aug_nn_63, &reorder_f_test_score_xgb_aug_nn_63xs, &reorder_f_test_score_xgb_aug_nn_63ys, &reorder_f_test_score_xgb_aug_nn_63xys,
    &reorder_f_test_score_knn_count_20 
    };    

    size_t cache_size = 1000000;
    // partial check
    // num_valid_test = cache_size; 

    for (int i = 32; i < 48; i++){
        weights_all[i] = 3.0;
    } 

    cout << f_reorder_test_scores_all.size() << endl;
    
    // // mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);
    // // // mapk_score = file_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test);

    // // cout << mapk_score << endl;

    // weights_all[48] = 12.0;

    // cout << weights_all[48] << endl;

    // mapk_score = file_cache_generate_validate_apk(f_valid_test, f_reorder_valid_scores_all, weights_all, max_sub, num_valid_test, cache_size);

    // cout << mapk_score << endl;


    // file_generate_sub(f_test_sub, f_reorder_scores, all_weights, max_sub, num_test);


    weights_all[48] = 8.0;

    cout << weights_all[48] << endl;

    file_cache_generate_sub(f_test_sub, f_reorder_test_scores_all, weights_all,
            max_sub, num_test, cache_size);


    f_test_sub.close();

    reorder_f_test_score_100.close();
    reorder_f_test_score_100xs.close();
    reorder_f_test_score_100ys.close();
    reorder_f_test_score_100xys.close();

    reorder_f_test_score_71.close();
    reorder_f_test_score_71xs.close();
    reorder_f_test_score_71ys.close();
    reorder_f_test_score_71xys.close();

    reorder_f_test_score_90.close();
    reorder_f_test_score_90xs.close();
    reorder_f_test_score_90ys.close();
    reorder_f_test_score_90xys.close();

    reorder_f_test_score_63.close();
    reorder_f_test_score_63xs.close();
    reorder_f_test_score_63ys.close();
    reorder_f_test_score_63xys.close();   

    // reorder_f_valid_score_83.close();
    // reorder_f_valid_score_83xs.close();
    // reorder_f_valid_score_83ys.close();
    // reorder_f_valid_score_83xys.close();  

    reorder_f_test_score_xgb_new2_100.close();
    reorder_f_test_score_xgb_new2_100xs.close();
    reorder_f_test_score_xgb_new2_100ys.close();
    reorder_f_test_score_xgb_new2_100xys.close();

    reorder_f_test_score_xgb_new2_71.close();
    reorder_f_test_score_xgb_new2_71xs.close();
    reorder_f_test_score_xgb_new2_71ys.close();
    reorder_f_test_score_xgb_new2_71xys.close();

    reorder_f_test_score_xgb_new2_90.close();
    reorder_f_test_score_xgb_new2_90xs.close();
    reorder_f_test_score_xgb_new2_90ys.close();
    reorder_f_test_score_xgb_new2_90xys.close();

    reorder_f_test_score_xgb_new2_63.close();
    reorder_f_test_score_xgb_new2_63xs.close();
    reorder_f_test_score_xgb_new2_63ys.close();
    reorder_f_test_score_xgb_new2_63xys.close(); 

    reorder_f_test_score_xgb_aug_nn_100.close();
    reorder_f_test_score_xgb_aug_nn_100xs.close();
    reorder_f_test_score_xgb_aug_nn_100ys.close();
    reorder_f_test_score_xgb_aug_nn_100xys.close();

    reorder_f_test_score_xgb_aug_nn_71.close();
    reorder_f_test_score_xgb_aug_nn_71xs.close();
    reorder_f_test_score_xgb_aug_nn_71ys.close();
    reorder_f_test_score_xgb_aug_nn_71xys.close();

    reorder_f_test_score_xgb_aug_nn_90.close();
    reorder_f_test_score_xgb_aug_nn_90xs.close();
    reorder_f_test_score_xgb_aug_nn_90ys.close();
    reorder_f_test_score_xgb_aug_nn_90xys.close();

    reorder_f_test_score_xgb_aug_nn_63.close();
    reorder_f_test_score_xgb_aug_nn_63xs.close();
    reorder_f_test_score_xgb_aug_nn_63ys.close();
    reorder_f_test_score_xgb_aug_nn_63xys.close();


    // reorder_f_test_score_knn_new2_20.close();
    reorder_f_test_score_knn_count_20.close();

    // f_reorder_scores_100.clear();
    // f_reorder_scores_71.clear();
    // f_reorder_scores_83.clear();

    // f_reorder_score_100.close();
    // f_reorder_score_100xs.close();
    // f_reorder_score_100ys.close();
    // f_reorder_score_100xys.close();

    // f_reorder_score_71.close();
    // f_reorder_score_71xs.close();
    // f_reorder_score_71ys.close();
    // f_reorder_score_71xys.close();

    // f_reorder_score_83.close();
    // f_reorder_score_83xs.close();
    // f_reorder_score_83ys.close();
    // f_reorder_score_83xys.close();

    // // f_valid_test.close();

    // f_test_sub.close();

    /* new version end */
}

void direct_run_merge(){
    int max_rank = 8, max_sub = 3;
    size_t num_train = 29118021, num_test = 8607230;
    double mapk_score = 0, weight = 1;

    cout << "direct_run_merge" << endl;

    ifstream f_train ("../input/train.csv");
    ifstream f_test("../input/test.csv");

    ifstream f_test_score_knn_count_20("../train/test_score_knn_count_20_20160706044948.csv");

    // ifstream f_score_100("../train/test_score_100_20160605000504.csv");
    // ifstream f_score_100xs("../train/test_score_100xs_20160606172637.csv");
    // ifstream f_score_100ys("../train/test_score_100ys_20160606172711.csv");
    // ifstream f_score_100xys("../train/test_score_100xys_20160607130744.csv");
    
    // ifstream f_score_knn_100("../train/test_score_knn_100_20160621112130.csv");
    // ifstream f_score_xgb_new_100("../train/test_score_xgb_new_100_20160622102600.csv");

    // ifstream f_score_xgb_new2_100("../train/test_score_xgb_new2_100_20160622121911.csv");
    // ifstream f_score_xgb_new2_100xs("../train/test_score_xgb_new2_100xs_20160623205834.csv");
    // ifstream f_score_xgb_new2_100ys("../train/test_score_xgb_new2_100ys_20160623205919.csv");
    // ifstream f_score_xgb_new2_100xys("../train/test_score_xgb_new2_100xys_20160623210050.csv");

    ofstream f_sub("../output/sub_knn_count_20.csv");

    // if (!f_train.is_open() || !f_test.is_open() || !f_score_100.is_open() ||
    //  !f_score_100xs.is_open() || !f_score_100ys.is_open() || !f_score_100xys.is_open()) return;

    vector<unordered_map<unsigned long, double> > score(num_test);
    unordered_map<unsigned int, unsigned int> row_id_map;
    const clock_t begin_time = clock();

    generate_row_id_map(f_test, row_id_map);

    file_update_score(f_test_score_knn_count_20, weight, score, row_id_map);

    f_test_score_knn_count_20.close();

    // file_update_score(f_score_100xs, weight, score, row_id_map);

    // file_update_score(f_score_100ys, weight, score, row_id_map);

    // file_update_score(f_score_100xys, weight, score, row_id_map);

    // file_update_score(f_score_xgb_new2_100, weight, score, row_id_map);

    // file_update_score(f_score_xgb_new2_100xs, weight, score, row_id_map);

    // file_update_score(f_score_xgb_new2_100ys, weight, score, row_id_map);

    // file_update_score(f_score_xgb_new2_100xys, weight, score, row_id_map);

    generate_sub(f_sub, max_sub, score);

    cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

    f_sub.close();
    // f_score_100.close();
    // f_score_100xs.close();
    // f_score_100ys.close();
    // f_score_100xys.close();
    // f_score_xgb_new2_100.close();
    // f_score_xgb_new2_100xs.close();
    // f_score_xgb_new2_100ys.close();
    // f_score_xgb_new2_100xys.close();
    f_train.close();
    f_test.close();    

}

int main(){
    int max_rank = 8, max_sub = 3, weight = 1;
    size_t num_train = 29118021, num_test = 8607230;
    
    string timestamp = get_timestamp();

    // ifstream f_train ("../input/train.csv");
    // ifstream f_test("../input/test.csv");

    /* generate validation data set begin */

    // ofstream f_valid_train("../valid/valid_train.csv");
    // ofstream f_valid_test("../valid/valid_test.csv");

    // time_split_generate_validate(655200, f_train, f_valid_train, f_valid_test);

    // f_valid_train.close();
    // f_valid_test.close();

    /* generate validation dat set end */

    /* generate hash separated file begin*/
    // ifstream f_train ("../input/train.csv");
    // ofstream f_train_hash("../input/train_hash.csv");

    // split_by_grid(100, 200, f_train, f_train_hash, nulltime);

    // f_train_hash.close();
    // f_train.close();
    /* generate hash separated file end*/

    /* generate train test hash separated file begin*/
    // ifstream f_train ("../input/train.csv");
    // ifstream f_test("../input/test.csv");

    // // // ofstream f_train_test_hash("../input/train_test_hash_63_159.csv");
    // // // ofstream f_train_test_hash_xs("../input/train_test_hash_63_159_xs.csv");
    // // // ofstream f_train_test_hash_ys("../input/train_test_hash_63_159_ys.csv");
    // // ofstream f_train_test_hash_xys("../input/train_test_hash_63_159_xys.csv");


    // // // ofstream f_train_test_hash("../input/train_test_hash_90_223.csv");
    // // // ofstream f_train_test_hash_xs("../input/train_test_hash_90_223_xs.csv");
    // // // ofstream f_train_test_hash_ys("../input/train_test_hash_90_223_ys.csv");
    // // // ofstream f_train_test_hash_xys("../input/train_test_hash_90_223_xys.csv");


    // // // ofstream f_train_test_hash("../input/train_test_hash_83_167.csv");
    // // // ofstream f_train_test_hash_xs("../input/train_test_hash_83_167_xs.csv");
    // // // ofstream f_train_test_hash_ys("../input/train_test_hash_83_167_ys.csv");
    // // // ofstream f_train_test_hash_xys("../input/train_test_hash_83_167_xys.csv");

    // // // ofstream f_train_test_hash_xs("../input/train_test_hash_71_141_xs.csv");
    // // // ofstream f_train_test_hash_ys("../input/train_test_hash_71_141_ys.csv");
    // // // ofstream f_train_test_hash_xys("../input/train_test_hash_71_141_xys.csv");
    // // // ofstream f_train_test_hash("../input/train_test_hash_100_200_ys.csv");
    // // // ofstream f_train_test_hash("../input/train_test_hash_100_200_xys.csv");

    // ofstream f_train_test_hash("../input/train_test_hash_100_200_gaussian_weighted.csv");

    // // // train_test_split_by_grid(63, 159, f_train, f_test, f_train_test_hash, nulltime, prep_xy);
    // // // train_test_split_by_grid(63, 159, f_train, f_test, f_train_test_hash_xs, nulltime, prep_xy_xshift);
    // // // train_test_split_by_grid(63, 159, f_train, f_test, f_train_test_hash_ys, nulltime, prep_xy_yshift);
    // // train_test_split_by_grid(63, 159, f_train, f_test, f_train_test_hash_xys, nulltime, prep_xy_xyshift);


    // // // train_test_split_by_grid(90, 223, f_train, f_test, f_train_test_hash, nulltime, prep_xy);
    // // // train_test_split_by_grid(90, 223, f_train, f_test, f_train_test_hash_xs, nulltime, prep_xy_xshift);
    // // // train_test_split_by_grid(90, 223, f_train, f_test, f_train_test_hash_ys, nulltime, prep_xy_yshift);
    // // // train_test_split_by_grid(90, 223, f_train, f_test, f_train_test_hash_xys, nulltime, prep_xy_xyshift);

    // // // train_test_split_by_grid(83, 167, f_train, f_test, f_train_test_hash, nulltime, prep_xy);
    // // // train_test_split_by_grid(83, 167, f_train, f_test, f_train_test_hash_xs, nulltime, prep_xy_xshift);
    // // // train_test_split_by_grid(83, 167, f_train, f_test, f_train_test_hash_ys, nulltime, prep_xy_yshift);
    // // // train_test_split_by_grid(83, 167, f_train, f_test, f_train_test_hash_xys, nulltime, prep_xy_xyshift);
    
    // // // train_test_split_by_grid(71, 141, f_train, f_test, f_train_test_hash_xs, nulltime, prep_xy_xshift);
    // // // train_test_split_by_grid(71, 141, f_train, f_test, f_train_test_hash_ys, nulltime, prep_xy_yshift);
    // // // train_test_split_by_grid(71, 141, f_train, f_test, f_train_test_hash_xys, nulltime, prep_xy_xyshift);

    // // train_test_split_by_grid_weight_by_acc(100, 200, 2.0, 0.0, f_train, f_test, f_train_test_hash, nulltime, prep_xy, anchor_xy);

    // train_test_split_by_grid_weight_gaussian(100, 200, 2.0, 0.0, f_train, f_test, f_train_test_hash, nulltime, prep_xy, anchor_xy);

    // f_train_test_hash.close();
    // // // f_train_test_hash_xs.close();
    // // // f_train_test_hash_ys.close();
    // // f_train_test_hash_xys.close();
    // f_train.close();
    // f_test.close();
    /* generate hash separated file end*/

    /* check validation score from existing results */
    // reorder_validation();
    check_validation();
    // direct_check_validation();
    /* check validation score from existing results end */

    /* merge result using score*/
    // reorder_test();
    // run_merge();
    // direct_run_merge();
    /* merge result using score*/

    /* run validation */
    // const clock_t begin_time = clock();

    // run_validation();

    // cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;
    /* end run validation */


    // ofstream f_sub("../output/submission_" + timestamp +".csv");
    // if (!f_train.is_open() || !f_test.is_open() || !f_sub.is_open()) return 0;

    // unordered_map<unsigned int, unordered_map<unsigned long, unsigned int> > grid;
    // unordered_map<unsigned int, vector<unsigned long> > grid_top;
    // vector<unordered_map<unsigned long, double> > score(num_test);


    // const clock_t begin_time = clock();

    // run_validation();

    // run_validation_ratio();
    
    // generate_grid(500, 1000, f_train, itime, grid);

    // generate_top(max_rank, grid_top, grid);

    // update_score(500, 1000, f_test, max_rank, 2 * weight, itime, score, grid_top);

    // cout << "itime done" << endl;

    // generate_grid(150, 300, f_train, dayround0, grid);

    // generate_top(max_rank, grid_top, grid);

    // update_score(150, 300, f_test, max_rank, weight, dayround0, score, grid_top);

    // cout << "dayround0 done" << endl;

    // generate_grid(250, 500, f_train, dayround, grid);

    // generate_top(max_rank, grid_top, grid);

    // update_score(250, 500, f_test, max_rank, weight, dayround, score, grid_top);

    // cout << "dayround done" << endl;

    // generate_grid(400, 800, f_train, dayround2, grid);

    // generate_top(max_rank, grid_top, grid);

    // update_score(400, 800, f_test, max_rank, weight, dayround2, score, grid_top);

    // cout << "dayround2 done" << endl;

    // generate_grid(150, 300, f_train, weekround, grid);

    // generate_top(max_rank, grid_top, grid);

    // update_score(150, 300, f_test, max_rank, weight, weekround, score, grid_top);

    // cout << "weekround done" << endl; 

    // generate_grid(200, 400, f_train, weekround2, grid);

    // generate_top(max_rank, grid_top, grid);

    // update_score(200, 400, f_test, max_rank, weight, weekround2, score, grid_top);

    // cout << "weekround2 done" << endl;

    // generate_grid(300, 600, f_train, weekhalfday, grid);

    // generate_top(max_rank, grid_top, grid);

    // update_score(300, 600, f_test, max_rank, weight, weekhalfday, score, grid_top);

    // cout << "weekhalfday done" << endl;    

    // generate_grid(450, 900, f_train, weekday, grid);

    // generate_top(max_rank, grid_top, grid);

    // update_score(450, 900, f_test, max_rank, weight, weekday, score, grid_top);

    // cout << "weekday done" << endl; 

    // generate_grid(500, 1000, f_train, nulltime, grid);

    // generate_top(max_rank, grid_top, grid);

    // update_score(500, 1000, f_test, max_rank, weight, nulltime, score, grid_top);

    // cout << "nulltimedone" << endl; 

    // generate_sub(f_sub, max_sub, score);
    
    // cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

    // f_train.close();
    // f_test.close();
    // f_sub.close();
}

