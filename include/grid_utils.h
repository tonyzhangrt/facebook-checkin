#ifndef __GRID_UTILS_H__
#define __GRID_UTILS_H__

#include <utility>
#include <vector>

std::vector<double> anchor_xy(const unsigned int, const unsigned int, const unsigned int, const unsigned int);

std::vector<double> anchor_xy_xshift(const unsigned int, const unsigned int, const unsigned int, const unsigned int);

std::vector<double> anchor_xy_yshift(const unsigned int, const unsigned int, const unsigned int, const unsigned int);

std::vector<double> anchor_xy_xyshift(const unsigned int, const unsigned int, const unsigned int, const unsigned int);

std::pair<unsigned int, unsigned int> prep_xy(const double, const double, const unsigned int, const unsigned int);

std::pair<unsigned int, unsigned int> prep_xy_xshift(const double, const double, const unsigned int, const unsigned int);

std::pair<unsigned int, unsigned int> prep_xy_yshift(const double, const double, const unsigned int, const unsigned int);

std::pair<unsigned int, unsigned int> prep_xy_xyshift(const double, const double, const unsigned int, const unsigned int);

unsigned int nulltime(unsigned int);

unsigned int itime(unsigned int);

unsigned int halfdayround0(unsigned int);

unsigned int halfdayround(unsigned int);

unsigned int halfdayround2(unsigned int);

unsigned int dayround0(unsigned int);

unsigned int dayround(unsigned int);

unsigned int dayround2(unsigned int);

unsigned int dayround3(unsigned int);

unsigned int dayround4(unsigned int);

unsigned int dayround8(unsigned int);

unsigned int dayroundhalfday(unsigned int);

unsigned int weekround0(unsigned int);

unsigned int weekround(unsigned int);

unsigned int weekround1(unsigned int);

unsigned int weekround2(unsigned int);

unsigned int weekround4(unsigned int);

unsigned int weekquaterday(unsigned int);

unsigned int weekhalfday(unsigned int);

unsigned int weekday(unsigned int);

unsigned int yearroundday(unsigned int);

unsigned int yearroundweek(unsigned int);

unsigned int weekdayshift(unsigned int);

unsigned int weekcount(unsigned int);

double rect_intersect_global(double, double, double, double);

double rect_intersect_local(const double, const double, const double, const double, 
    const unsigned int, const unsigned int, const unsigned int, const unsigned int,
    std::vector<double> (*)(const unsigned int, const unsigned int, const unsigned int, const unsigned int));

double gaussain_prob_local(const double, const double, const double, const double,  
    const unsigned int, const unsigned int, const unsigned, const unsigned,
    std::vector<double> (*)(const unsigned int, const unsigned int, const unsigned int, const unsigned int));    


#endif