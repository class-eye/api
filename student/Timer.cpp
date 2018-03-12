#include "student/Timer.hpp"

using Clock = std::chrono::high_resolution_clock;

void Timer::Tic() {
	time_record time1;
	time1.start_ = Clock::now();
	record.push_back(time1);
}
/*! \brief stop timer */
void Timer::Toc() {
	time_record time2 = record.back();
	time2.end_ = Clock::now();
	record_pair.push_back(time2);
	record.pop_back();
}
/*! \brief return time in ms */
double Timer::Elasped() {

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(record_pair[0].end_-record_pair[0].start_);
	record_pair.clear();
	return duration.count();
}

const string Timer::getCurrentSystemTime()
{
	auto tt = std::chrono::system_clock::to_time_t
		(std::chrono::system_clock::now());
	struct tm* ptm = localtime(&tt);
	char date[60] = { 0 };
	sprintf(date, "%d-%02d-%02d %02d:%02d:%02d",
		(int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
		(int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);
	return std::string(date);
}

string Timer::getFileCreateTime(string &file){
	struct stat buf;
	int result;
	result = stat(file.c_str(), &buf);

	if (result != 0)
		perror("No such file or directory");
	else
	{
		string str_time = ctime(&buf.st_mtime);
		char buff[300];
		sprintf(buff, "%s_%s_%s %s", str_time.substr(20, 4).c_str(), str_time.substr(4, 3).c_str(), str_time.substr(8, 2).c_str(), str_time.substr(11, 8).c_str());
		string create_time = buff;
		return create_time;
	}
}