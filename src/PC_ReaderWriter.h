#ifndef PC_READER
#define PC_READER

//#include <string>
#include <iostream>
#include <fstream>
#include "stdlib.h"

using namespace std;

class PC_ReaderWriter {

public:

	double* points;
	int no_of_rows;
	int no_of_cols;

	PC_ReaderWriter(double* points, int no_of_rows, int no_of_cols);
	PC_ReaderWriter(const char* fileName);

	~PC_ReaderWriter();

	void load(const char* fileName);
	void save(const char* fileName);
};

#endif