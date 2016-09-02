#ifndef __GRID_SPLIT_H__
#define __GRID_SPLIT_H__

#include <fstream>
#include <utility>
#include <vector>

void split_by_grid(const unsigned int, const unsigned int, std::ifstream &, std::ofstream &, unsigned int (*) (unsigned int),
        std::pair<unsigned int, unsigned int> (*)(const double, const double, const unsigned int, const unsigned int));

void train_test_split_by_grid(const unsigned int, const unsigned int, std::ifstream &, std::ifstream &, std::ofstream &, 
        unsigned int (*) (unsigned int), 
        std::pair<unsigned int, unsigned int> (*)(const double, const double, const unsigned int, const unsigned int));

void train_test_split_by_grid_weight_by_acc(const unsigned int, const unsigned int, double, double, std::ifstream &, std::ifstream &, std::ofstream &, 
        unsigned int (*) (unsigned int), std::pair<unsigned int, unsigned int> (*)(const double, const double, const unsigned int, const unsigned int),
        std::vector<double> (*)(const unsigned int, const unsigned int, const unsigned int, const unsigned int));

void train_test_split_by_grid_weight_gaussian(const unsigned int, const unsigned int, double, double, std::ifstream &, std::ifstream &, std::ofstream &, 
        unsigned int (*) (unsigned int), std::pair<unsigned int, unsigned int> (*)(const double, const double, const unsigned int, const unsigned int),
        std::vector<double> (*)(const unsigned int, const unsigned int, const unsigned int, const unsigned int));

#endif