#include "grid_utils.h"
#include <cmath>

using namespace std;

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