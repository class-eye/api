#ifndef _TIMER_H_
#define _TIMER_H_
#include <chrono>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include<vector>
#include<string>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace std;
class Timer {
	using Clock = std::chrono::high_resolution_clock;

public:
	void Tic();
	void Toc();
	double Elasped();
	const string getCurrentSystemTime();	
	string getFileCreateTime(string &file);		

private:
	struct time_record{
		Clock::time_point start_, end_;
	};
	vector<time_record>record;
	vector<time_record>record_pair;

};
#endif