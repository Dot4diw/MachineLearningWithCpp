// Perceptron 
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "eigen-3.4.0/Eigen/Dense"

struct ExampleData
{
    Eigen::VectorXd feature_vector {};
    int label {};
};

// Read data from a file in CSV format.
const std::vector<ExampleData>& ReadData(std::string file)
{
    std::ifstream infile;
    const char split_char { ','};
    std::string line {};

    static std::vector<ExampleData> example_data {};

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
        ExampleData tmp_data {};

        while (std::getline(line_stream, line, split_char))
        {
            split_vector.push_back(line);
        }
        // std::cout << split_vector.size() << '\n';
        std::vector<double> tmp_feature_data {};
        // tmp_feature_data.push_back(1.0); // x0 = 1, theta0 = b
        for ( std::size_t index = 0; index < (split_vector.size() - 1); ++index )
        {
            tmp_feature_data.push_back(stod(split_vector[index]));
        }
        
        tmp_data.feature_vector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tmp_feature_data.data(), tmp_feature_data.size());
        tmp_data.label = stod(split_vector[(split_vector.size() - 1)]);

        example_data.push_back(tmp_data);
    }
    infile.close();

    return example_data;
}

// Model train
const std::pair<Eigen::VectorXd, double>& PerceptronGradientDescentSolution(const std::vector<ExampleData>& train_data, Eigen::VectorXd& omega, double b_value = 0.0, double eta = 0.001)
{
    static std::pair<Eigen::VectorXd, double> model_parameters {};
    int n_features { static_cast<int>(train_data[0].feature_vector.rows()) };
    if (omega.rows() == 0){ omega = Eigen::VectorXd::Zero(n_features); }
    
    while (true)
    {
        bool miss_flag { false };
        for (const auto& it  : train_data)
        {
            Eigen::VectorXd feature_vector { it.feature_vector };
            int yi_value { it.label };
            double miss_value { yi_value * ( omega.dot(feature_vector) + b_value ) };
           
            if (miss_value <= 0)
            {
                miss_flag = true;
                omega = omega + eta * yi_value * feature_vector;
                b_value = b_value + eta * yi_value;
            }
        }

        double loss_value { 0.0 };
        for (const auto& it  : train_data)
        {
            Eigen::VectorXd feature_vector { it.feature_vector };
            int yi_value { it.label };
            double miss_value { yi_value * ( omega.dot(feature_vector) + b_value ) };
            loss_value = loss_value - miss_value;
        }
        
        std::cout << "loss value : " << loss_value;
        std::cout << "\tomega: [" << omega.transpose() << "]";
        std::cout << "\tb : " << b_value << std::endl;
        
        if (!miss_flag)
        {
            model_parameters.first = omega;
            model_parameters.second = b_value;
            return model_parameters;
        }
    }
}

// Model testing
const std::vector<std::pair<int, int>>& ModelTesting(const std::vector<ExampleData>& test_data, const Eigen::VectorXd& omega, const double b_value)
{
    static std::vector<std::pair<int,int>> op_label_vector { };
    int predicted_label { 0 };

    for (const auto& it  : test_data)
    {
        std::pair<int,int> op_label_pair { };
        Eigen::VectorXd feature_vector { it.feature_vector };
        int yi_true_label { it.label };

        double yi_predicted_value { omega.dot(feature_vector) + b_value };

        if (yi_predicted_value > 0)
        {
            predicted_label = 1;
        }
        else if (yi_predicted_value < 0)
        {
            predicted_label = -1;
        }

        op_label_pair.first = yi_true_label;
        op_label_pair.second = predicted_label;
        op_label_vector.push_back(op_label_pair);     
    }

    return op_label_vector;
}

// Unknown sample category attribute predictions
int ModelPrediction(const Eigen::VectorXd& unknown_data, const Eigen::VectorXd& omega, const double b_value)
{
    int y_predicted_label { 0 };
    double y_predicted_value { omega.dot(unknown_data) + b_value};
    
    if (y_predicted_value > 0)
    {
        y_predicted_label = 1;
    }
    else if (y_predicted_value < 0)
    {
        y_predicted_label = -1;
    }

    return y_predicted_label;
}

// Calculation accuracy
double CalcAccuracy(const std::vector<std::pair<int, int>>& tested_result)
{
    int n_error_predicted { 0 };
    for (const auto& it  : tested_result)
    {
        int original_label { it.first };
        int predicted_label { it.second };
        
        if (predicted_label != original_label)
        {
            n_error_predicted += 1;
        }
        std::cout << "Original Label:\t" << original_label 
                  << "\tPredictive label:\t" << predicted_label 
                  << "\t" << ((predicted_label == original_label) ? 'v' : 'x') 
                  << std::endl;    
    }
    
    double accuracy = {1.0 - static_cast<double>(n_error_predicted) / static_cast<double>(tested_result.size()) };

    return accuracy;
}


int main()
{
    // Read the data from csv file.
    std::string train_file = "..\\class_train.csv";
    std::string test_file  = "..\\class_test.csv";

    const std::vector<ExampleData>& train_data { ReadData(train_file) };
    const std::vector<ExampleData>& test_data { ReadData(test_file) };

    // Parameter initialization
    Eigen::VectorXd omega {}; // Initialize the theta matrix
    double b { 0.0 };
    double eta { 0.01 }; // Learning rate

    // Model training
    const std::pair<Eigen::VectorXd, double>& final_model_parameters { PerceptronGradientDescentSolution(train_data, omega, b, eta) };
    
    // Model testing
    const std::vector<std::pair<int, int>>& tested_result { ModelTesting(test_data, final_model_parameters.first, final_model_parameters.second) };
    
    std::cout << std::string(75, '-') << std::endl;
    // Calculation accuracy and print it.
    double test_accuracy { CalcAccuracy(tested_result) };
    // Print the model parameter
    std::cout << std::string(75, '-') << std::endl;
    std::cout << "The model parameter omega is : ["  
              << final_model_parameters.first.transpose() 
              << "]\t b value is : " 
              << final_model_parameters.second << std::endl;

    std::cout << std::string(75, '-') << std::endl;
    std::cout << "Prediction Accuracy : " << test_accuracy << std::endl;
  
    return 0;
}