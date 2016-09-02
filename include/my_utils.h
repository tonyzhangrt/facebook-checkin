#ifndef __MY_UTILS_H__
#define __MY_UTILS_H__

#include <string>
#include <vector>

std::string get_timestamp();

void split_line(const std::string &, char, std::vector<std::string>&);

#endif