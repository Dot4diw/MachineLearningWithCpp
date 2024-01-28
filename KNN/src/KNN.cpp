#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include <algorithm>

struct DataPoint
{
    double m_x {};
    double m_y {};
    int group {};
    double distance {};
};

// Read the training data from a file in CSV format.
// The first two columns in the file indicate coordinates, 
// and the third column indicates the category to which it belongs.
std::vector<DataPoint>& read_data(std::string file)
{
    std::ifstream infile;
    static std::vector<DataPoint> data {};
    const char split_char { ','};
    std::string line {};

    infile.open(file, std::ios::in);

    if (!infile.is_open())
    {
        std::cerr << "!!! Could not open file - '" << file << "'" << std::endl;
        exit(0);
    }

    while (std::getline(infile, line))
    {
        std::stringstream line_stream {line};
        std::vector<std::string> split_vector {};
        DataPoint tmpPoint {};

        while (std::getline(line_stream, line, split_char))
        {
            split_vector.push_back(line);
        }

        tmpPoint.m_x = stod(split_vector[0]);
        tmpPoint.m_y = stod(split_vector[1]);
        tmpPoint.group = stod(split_vector[2]);

        data.push_back(tmpPoint);
    }
    infile.close();

    return data;
}

int knn_predict(std::vector<DataPoint>& point_data, int k, DataPoint& tp)
{
	// Calculates the distance from all points in the training set data 
    // to the test data point.
	for (std::size_t i = 0; i < point_data.size(); i++)
    {
        // Euclidean Distance
        point_data[i].distance = 
			sqrt((point_data[i].m_x - tp.m_x) * (point_data[i].m_x - tp.m_x) +
				(point_data[i].m_y - tp.m_y) * (point_data[i].m_y - tp.m_y));
    }

	// Sort the Points by distance.
    sort(point_data.begin(), point_data.end(), [](DataPoint& dpa, DataPoint& dpb) -> bool
    {
        return (dpa.distance < dpb.distance);
    });

    // Calculate the number of points belonging to each class 
    // in the first k minimum distances.
	int n_neighbor0 { 0 };	 // neighbor counts of group 0
	int n_neighbor1 { 0 };	 // neighbor counts of group 1
    int n_neighbor2 { 0 };	 // neighbor counts of group 2

	for (int i = 0; i < k; ++i)
	{
		if (point_data[i].group == 0) {n_neighbor0++;}
		else if (point_data[i].group == 1) { n_neighbor1++; }
        else if (point_data[i].group == 2) { n_neighbor2++; }
	}
    // Returns classification results for unknown points.
    if (n_neighbor0 > n_neighbor1 &&  n_neighbor0 > n_neighbor2) { return 0; }
    else if (n_neighbor1 > n_neighbor0 &&  n_neighbor1 > n_neighbor2) { return 1; }
    else if (n_neighbor2 > n_neighbor0 &&  n_neighbor2 > n_neighbor1) { return 2; }
    
    /*
     * In other cases, the result cannot be predicted, and -1 is returned, 
     * and it is recommended to reset the k value.
    */
    else { return -1; }
}

int main()
{
    std::string filename { "..\\data.csv" };
    std::vector<DataPoint> train_data { read_data(filename) };

	// Testing Point
	std::vector<DataPoint> test_points {};

    DataPoint test_p1 {4.5, 10};
    test_points.push_back(test_p1);

    DataPoint test_p2 {8, 2.5};
    test_points.push_back(test_p2);

    DataPoint test_p3 {12, 7.5};
    test_points.push_back(test_p3);

    DataPoint test_p4 {6, 6};
    test_points.push_back(test_p4);

	// The value of k.
	int k { 5 };

    // Testing.
    for (std::size_t i = 0; i < test_points.size(); ++i)
    {
	std::cout << "The classification prediction for unknown points (" 
              << test_points[i].m_x << ", " 
              << test_points[i].m_y << ')' << " belong to group " 
              << knn_predict(train_data, k, test_points[i]) << ".\n";
    }
    return EXIT_SUCCESS;
}
