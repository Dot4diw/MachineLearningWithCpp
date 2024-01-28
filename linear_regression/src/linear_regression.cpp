#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <vector>
#include <math.h>
#include<numeric>


// Define a structure for storing linear coordinate values
struct DataPoint
{
    double m_x {};
    double m_y {};
};

// Read the training data from a file in CSV format.
const std::vector<DataPoint>& ReadData(std::string file)
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

        data.push_back(tmpPoint);
    }
    infile.close();

    return data;
}

// Use the least square method to linearly fit the input training set data, 
// and return the fitting parameters beta0 and beta1.
const std::pair<double, double>& LinearRegressionModelTrain(const std::vector<DataPoint>& point_data)
{
    static std::pair<double, double> model_parameter {};

    // Calculate the x-coordinate mean.
    double x_mean = std::accumulate( point_data.begin(), point_data.end(), 0.0,
        [](double a, DataPoint tmp) -> double { 
            return a + tmp.m_x; 
            } ) / point_data.size();

    // Calculate the y-coordinate mean.
    double y_mean = std::accumulate( point_data.begin(), point_data.end(), 0.0,
        [](double a, DataPoint tmp) -> double { 
            return a + tmp.m_y; 
            } ) / point_data.size();

    // Calculate the beta1 value.
    double beta1_numerator {};
    double beta1_denominator {};

	for (std::size_t i = 0; i < point_data.size(); ++i)
    {
        beta1_numerator += point_data[i].m_y * (point_data[i].m_x - x_mean);
        beta1_denominator += std::pow(point_data[i].m_x, 2) - ( point_data[i].m_x * x_mean );
    }
    
    // beta1
    model_parameter.second = beta1_numerator / beta1_denominator;
    // beta0
    model_parameter.first = y_mean - x_mean * model_parameter.second;

    return model_parameter;
}

double ModelPrediction(const std::pair<double,double>& beta, const double x)
{
    return beta.first + beta.second * x;
}

int main()
{
    // Read the data from csv file.
    std::string filename { "..\\lr_data.csv" };
    const std::vector<DataPoint>& train_data { ReadData(filename) };

    // Model fitting.
    const std::pair<double, double>& beta_parameter { LinearRegressionModelTrain(train_data) };
    
    std::cout << "The linear regression fitting model is: "
              << "y = " << beta_parameter.first << " + " 
              << beta_parameter.second << "x" << ".\n"
              << std::string(85, '+') << '\n';

	// Model prediction
    std::array<double, 5> test_data { 23.456, 34.567, 45.678, 56.789, 67.666 };
    for (double value : test_data)
    {
        double model_predict_result { ModelPrediction(beta_parameter, value) };
        std::cout << "When x = " << value 
                  << ", the result predicted by the linear regression model is y = " 
                  << model_predict_result << ".\n"; 
    }

    return EXIT_SUCCESS;
}
