#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "eigen-3.4.0/Eigen/Dense"
#include "eigen-3.4.0/Eigen/Core"

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
        //std::cout << split_vector.size() << '\n';
        std::vector<double> tmp_feature_data {};
        tmp_feature_data.push_back(1.0); // x0 = 1, theta0 = b
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

//double GradientDescentSolution(const std::vector<ExampleData>& train_data, Eigen::VectorXd theta , double alpha = 0.001 , double eplison = 10E-8 )

const Eigen::VectorXd& GradientDescentSolution(const std::vector<ExampleData>& train_data, Eigen::VectorXd& theta_vector, double alpha = 0.001, int iteration = 1000)
{
    int n_examples { static_cast<int>(train_data.size()) };
    int n_features { static_cast<int>(train_data[0].feature_vector.rows()) };
    if (theta_vector.rows() == 0){ theta_vector = Eigen::VectorXd::Zero(n_features); }
    
    int n_iteration { 0 };
    while (n_iteration < iteration)
    {
        for (int j = 0 ; j < n_features; ++j)
        {
            double gradient_descent {};
            for (const auto& it  : train_data)
            {
                Eigen::VectorXd feature_vector { it.feature_vector };
                int yi_value { it.label };
                gradient_descent += (( (1.0 / ( 1.0 + (exp( -(theta_vector.transpose().dot(feature_vector)))))) - yi_value ) * feature_vector[j]);
            }
            gradient_descent = gradient_descent / n_examples;
            theta_vector[j] = theta_vector[j] - alpha*gradient_descent;
        }

        double loss {};
        for (const auto& it  : train_data)
        {
            Eigen::VectorXd feature_vector { it.feature_vector };
            int yi_value { it.label };
            loss += yi_value * (theta_vector.transpose().dot(feature_vector)) - log(1.0 + exp((theta_vector.transpose().dot(feature_vector))));

        }

        loss = -(loss / n_examples);

        std::cout << "Loss value : " << loss << "\t\t";
        std::cout << "theta transpose :  "<< theta_vector.transpose() << std::endl;
        
        n_iteration += 1;
    }
    std::cout << std::string(70, '-') << std::endl;

    return theta_vector;
}

std::vector<std::pair<int, int>>& ModelTesting(const std::vector<ExampleData>& test_data, const Eigen::VectorXd& final_theta_vector)
{
    static std::vector<std::pair<int,int>> op_label_vector { };
    int predicted_label { 0 };

    for (const auto& it  : test_data)
    {
        std::pair<int,int> op_label_pair { };
        Eigen::VectorXd feature_vector { it.feature_vector };
        int yi_true_label { it.label };
        double yi_predicted_value { 1.0 / (1.0 + exp(-(final_theta_vector.transpose().dot(feature_vector))))};

        if (yi_predicted_value > 0.5)
        {
            predicted_label = 1;
        }
        else if (yi_predicted_value < 0.5)
        {
            predicted_label = 0;
        }
        else
        {
            predicted_label = -1;
        }
        
        op_label_pair.first = yi_true_label;
        op_label_pair.second = predicted_label;
        
        op_label_vector.push_back(op_label_pair);

        std::cout << "Original Label: " << yi_true_label << "    Predictive label: " << predicted_label 
                  << "\t" << ((predicted_label == yi_true_label) ? 'v' : 'x') << std::endl;     
    }

    return op_label_vector;
}

double CalcAccuracy(std::vector<std::pair<int, int>>& tested_result)
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
    }
    
    double accuracy = {1.0 - static_cast<double>(n_error_predicted) / static_cast<double>(tested_result.size()) };

    return accuracy;
}

// Unknown sample category attribute predictions
int ModelPrediction(const Eigen::VectorXd& unknown_data, const Eigen::VectorXd& final_theta_vector)
{
    int y_predicted_label { -1 };
    double y_predicted_value { 1.0 / (1.0 + exp(-(final_theta_vector.transpose().dot(unknown_data))))};

    if (y_predicted_value > 0.5)
    {
        y_predicted_label = 1;
    }
    else if (y_predicted_value < 0.5)
    {
        y_predicted_label = 0;
    }
    else
    {
        return y_predicted_label ;
    }

    return y_predicted_label;
}

int main()
{
    // Read the data from csv file.
    std::string train_file = "..\\moons_training_data.csv";
    std::string test_file  = "..\\moons_testing_data.csv";

    const std::vector<ExampleData>& train_data { ReadData(train_file) };
    const std::vector<ExampleData>& test_data { ReadData(test_file) };

    // Parameter initialization
    Eigen::VectorXd theta_vector {}; // Initialize the theta matrix
    double alpha { 0.05 }; // Learning rate
    int iteration { 4000 }; // Number of iterations

    // Model training
    const Eigen::VectorXd& final_theta_vector { GradientDescentSolution(train_data, theta_vector, alpha, iteration) };

    // Model testing
    std::vector<std::pair<int, int>>& tested_result { ModelTesting(test_data, final_theta_vector) };
    
    // Print the model parameter
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "The model parameter theta is : "  << final_theta_vector.transpose() << std::endl;
    
    // Calculation accuracy and print it.
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "Prediction Accuracy : " << CalcAccuracy(tested_result) << std::endl;
    
    // Unknown sample category attribute predictions
    //std::vector<double> new_data { 1.8842326, 0.050685}; 
    //Eigen::VectorXd new_data_vector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(new_data.data(), new_data.size());
    Eigen::VectorXd new_data_vector (3, 1);
    new_data_vector << 1, 1.8842326, 0.050685;

    int unknown_data_predicted_label { ModelPrediction(new_data_vector, final_theta_vector) };
    std::cout << std::endl << std::string(50, '-') << std::endl;;
    std::cout << "---Unknown sample category attribute predictions--" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "Unknown sampe : " << new_data_vector.transpose() << std::endl;
    std::cout << "Unknown sample label : " << unknown_data_predicted_label << "." << std::endl;

    return 0;
}