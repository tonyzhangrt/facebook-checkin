#ifndef __BASIC_COUNT_H__
#define __BASIC_COUNT_H__

#include <unordered_map>
#include <fstream>
#include <vector>
#include <utility>

void generate_grid(const unsigned int, const unsigned int y_range, std::ifstream &, unsigned int (*) (unsigned int),
        std::unordered_map<unsigned int, std::unordered_map<unsigned long, unsigned int> > &,
        std::pair<unsigned int, unsigned int> (*)(const double, const double, const unsigned int, const unsigned int));

void generate_top(int, std::unordered_map<unsigned int, std::vector<unsigned long> > &,
        std::unordered_map<unsigned int, std::unordered_map<unsigned long, unsigned int> > &);

void generate_top_ratio(int, std::unordered_map<unsigned int, std::vector<std::pair<unsigned long, double> > > &,
        std::unordered_map<unsigned int, std::unordered_map<unsigned long, unsigned int> > &);

void update_score(const unsigned int, const unsigned int, std::ifstream &, int, double, unsigned int (*) (unsigned int),
    std::vector<std::unordered_map<unsigned long, double> > &, std::unordered_map<unsigned int, std::vector<unsigned long> > &);

void update_score_ratio(const unsigned int, const unsigned int, std::ifstream &, int, double, unsigned int (*) (unsigned int),
    std::vector<std::unordered_map<unsigned long, double> > &, std::unordered_map<unsigned int, std::vector<std::pair<unsigned long, double> > > &);


#endif