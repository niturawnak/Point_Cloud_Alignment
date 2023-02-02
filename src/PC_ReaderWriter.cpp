#include "PC_ReaderWriter.h"
#include <string>

PC_ReaderWriter::PC_ReaderWriter(double* points, int no_of_rows, int no_of_cols) {

	this->points = new double[no_of_rows * no_of_cols];
	for (int i = 0; i < (no_of_cols * no_of_rows); i++) this->points[i] = points[i];
	this->no_of_rows = no_of_rows;
	this->no_of_cols = no_of_cols;

} //end MatrixReaderWriter(double* data, int rownum, int columnNum)

PC_ReaderWriter::~PC_ReaderWriter() {
	if (this->points != NULL) delete[] points;
}

PC_ReaderWriter::PC_ReaderWriter(const char* fileName) {
	this->points = NULL;
	load(fileName);
}//end MatrixReaderWriter(const char* fileName)

void PC_ReaderWriter::load(const char* fileName) {
	ifstream datafile(fileName);
	string line;
	int lineCounter = 0;
	//Count lines
	if (datafile.is_open()) {
		while (!datafile.eof())
		{
			getline(datafile, line);
			if (line.length() != 0)
				if (line.at(0) != '#')
					lineCounter++;
		}//end while
		datafile.close();
	}//end if
	else {
		cout << "Unable to open file";
	}


	int columnCounter = 0;
	ifstream datafile2(fileName);

	if (datafile2.is_open()) {
		getline(datafile2, line);
		while (line.at(0) == '#')
			getline(datafile2, line);


		for (int i = line.find_first_not_of(" ,"); i <= line.find_last_not_of(" ,"); i++)
			if ((line.at(i) == ' ') || (line.at(i) == ','))
				columnCounter++;
		datafile2.close();
		columnCounter++;
	}


	no_of_rows = lineCounter;
	no_of_cols = columnCounter;

	if (points != NULL) delete[] points;
	points = new double[no_of_rows * no_of_cols];

	ifstream datafile3(fileName);

	lineCounter = 0;
	if (datafile3.is_open()) {
		getline(datafile3, line);
		while (line.at(0) == '#')
			getline(datafile3, line);


		while ((!datafile3.eof()) && (line.length() > 0)) {
			int index = line.find_first_not_of(" ,");
			for (int i = 0; i < columnCounter; i++) {
				string number = line.substr(index, line.find(" ", index + 1) - index + 1);
				double num = strtod((char*)&number[0], NULL);
				points[lineCounter * no_of_cols + i] = num;

				index = line.find(" ", index + 1);
			}//end for i
			getline(datafile3, line);
			lineCounter++;
		}//end while
		datafile3.close();
	}



} //end load(string filename)

void PC_ReaderWriter::save(const char* fileName) {
	ofstream myfile;
	myfile.open(fileName);
	myfile << "#Created by C++ matrix writer.\n";
	for (int i = 0; i < (no_of_rows); i++) {
		for (int j = 0; j < no_of_cols; j++) {
			myfile << points[i * no_of_cols + j] << " ";
		}//end for j
		myfile << endl;
	}//end for i

	myfile << endl;


	myfile.close();


} //end save(string filename)
