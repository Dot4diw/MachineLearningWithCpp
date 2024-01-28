#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "eigen-3.4.0/Eigen/Dense"
#include "eigen-3.4.0/Eigen/Core"

int main()
{
    std::vector<Eigen::VectorXd> train_data {};
    
    Eigen::VectorXd ev1 (2, 1);
    ev1 << 3, 3;
    train_data.push_back(ev1);

    Eigen::VectorXd ev2 (2, 1);
    ev2 << 4, 3;
    train_data.push_back(ev2);

    Eigen::VectorXd ev3 (2, 1);
    ev3 << 1, 1;
    train_data.push_back(ev3);

    int n_examples = 3;

    Eigen::MatrixXd GramMatrix(n_examples, n_examples);

    for(int i = 0; i < n_examples; ++i)
    {
        for ( int j = 0; j < n_examples; ++j)
        {
            GramMatrix(i,j) = train_data[j].transpose().dot(train_data[i]);
        }
    }
    for(int i = 0; i < n_examples; ++i)
    {
        for ( int j = 0; j < n_examples; ++j)
        {
           std::cout <<  GramMatrix(i,j) <<std::endl;
           std::cout << "+++++++++++++++++++++++++++\n";
           std::cout <<  GramMatrix(i,j) << "\n\n";
        }
    }
    std::cout << GramMatrix << '\n';
    return 0;
}