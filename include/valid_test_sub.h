#ifndef __VALID_TEST_SUB_H__
#define __VALID_TEST_SUB_H__

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>

double apk(std::unordered_set<unsigned long> &, std::vector<unsigned long> &, const int);

void time_split_generate_validate(unsigned int, std::ifstream &, std::ofstream &, std::ofstream &);

void generate_valid_score(std::ofstream &, int, std::vector<std::unordered_map<unsigned long, double> > &);

void generate_sub(std::ofstream &, int, std::vector<std::unordered_map<unsigned long, double> > &);

double validate_apk(std::ifstream &, int, std::vector<std::unordered_map<unsigned long, double>> &);

void reorder_score(std::ifstream &, std::ofstream &);

void generate_row_id_map(std::ifstream &, std::unordered_map<unsigned int, unsigned int> &);

void file_update_score(std::ifstream &, double, std::vector<std::unordered_map<unsigned long, double> > &,  std::unordered_map<unsigned int, unsigned int> &);

void file_update_score_limited(std::ifstream &, double, unsigned int,
    std::vector<std::unordered_map<unsigned long, double> > &,  std::unordered_map<unsigned int, unsigned int> &);

void file_generate_sub(std::ofstream &, std::vector<std::ifstream*> &, std::vector<double> &, int, size_t);

void file_cache_generate_sub(std::ofstream &, std::vector<std::ifstream*> &, std::vector<double> &, int, size_t, size_t);

double file_generate_validate_apk(std::ifstream &, std::vector<std::ifstream*> &, std::vector<double> &, int, size_t);

double file_cache_generate_validate_apk(std::ifstream &, std::vector<std::ifstream*> &, std::vector<double> &, int, size_t, size_t);

double file_cache_generate_validate_apk_log(std::ifstream &, std::vector<std::ifstream*> &, std::vector<double> &, int, size_t, size_t);

#endif