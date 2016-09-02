#include "my_utils.h"
#include <ctime>
#include <sstream>
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

void split_line(const string &line, char delim, vector<string> &result){
    istringstream iss(line);
    string word;
    while (getline(iss, word, delim)){
        result.push_back(word);
    }
}