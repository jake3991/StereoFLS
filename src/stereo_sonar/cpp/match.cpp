#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <random>

typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXb;

using namespace std;

/**
 * This method takes the features in a pair of sonar images (horizontal/vertical) 
 * and performs feature association between features. We will return the matches
 * as an eigen matrix including a confidence score. 
 * 
 * @param range_horizontal_discrete the discretized range data
 * @param range_horizontal the horizontal sonar range data, here kept as float
 * @param bearing_horizontal the horizontal sonar bearing data, here kept as float
 * @param range_vertical_discrete the discretzed range data from the vertical sonar
 * @param range_vertical the vertical sonar range data, here kept as float
 * @param x_vertical the vertical sonar X-axis data, here kept as float
 * @param patches_horizontal the patches around the provided features in the horizontal sonar
 * @param patches_vertical the patches around the provided features in the vertical sonar
 * @return Eigen::MatrixXf 
 */
Eigen::MatrixXf matchFeatures(const pybind11::array_t<int>& range_horizontal_discrete,
                    const pybind11::array_t<float>& range_horizontal,
                    const pybind11::array_t<float>& bearing_horizontal,
                    const pybind11::array_t<int>& range_vertical_discrete,
                    const pybind11::array_t<float>& range_vertical,
                    const pybind11::array_t<float>& x_vertical,
                    const MatrixXf & patches_horizontal,
                    const MatrixXf & patches_vertical){

    //containers for subproblems here we will map discrete range to index in input
    unordered_map<int,vector<int>> horizontal_bins = unordered_map<int,vector<int>>();
    unordered_map<int,vector<int>> vertical_bins = unordered_map<int,vector<int>>();

    //loop over the discrete ranges in the horizontal axis
    //here we want to gather all the indexes that are at the same range
    //we will use the unordered_map to give us constant look up time for
    //a given discrete range
    for(int i = 0; i < range_horizontal_discrete.size(); ++i){
      if(horizontal_bins.count(range_horizontal_discrete.at(i)) == 0){
        vector<int> temp = {i};
        horizontal_bins.insert(make_pair(range_horizontal_discrete.at(i),temp));
      }else{
        horizontal_bins.at(range_horizontal_discrete.at(i)).push_back(i);
      }
    }

    //same for vertical axis
    for(int i = 0; i < range_vertical_discrete.size(); ++i){
      if(vertical_bins.count(range_vertical_discrete.at(i)) == 0){
        vector<int> temp = {i};
        vertical_bins.insert(make_pair(range_vertical_discrete.at(i),temp));
      }else{
        vertical_bins.at(range_vertical_discrete.at(i)).push_back(i);
      }
    }

    //vectors to log the matched points and confidence
    vector<float> range_horizontal_matched = {};
    vector<float> bearing_horizontal_matched = {};
    vector<float> range_vertical_matched = {};
    vector<float> x_vertical_matched = {};
    vector<float> matching_confidence = {};

    //loop over the possible sub problems
    for (auto & subproblem : horizontal_bins){

      //check if we even have any features to match with in the vertical axis
      if(vertical_bins.count(subproblem.first) == 0)
        continue;

      //check if this is to small a case to be relavant
      if(subproblem.second.size() <= 2 || vertical_bins.at(subproblem.first).size() <= 2)
        continue;

      //copy the subproblem vectors
      vector<int> horizontal_problem = vector<int>(subproblem.second);
      vector<int> vertical_problem = vertical_bins.at(subproblem.first);

      //parameters for feature matching
      int best_cost = INT_MAX;
      vector<int> best_vector;
      auto rng = std::default_random_engine {};

      //containers for SAD values between features
      vector<int> min_values = vector<int>(vertical_problem.size(),INT_MAX);
      vector<int> max_values = vector<int>(vertical_problem.size(),0);

      //test N combinations of feature associations
      for(int step = 0; step < max(horizontal_problem.size(),vertical_problem.size()); ++step){
        
        //shuffle one of the vectors
        std::shuffle(horizontal_problem.begin(), horizontal_problem.end(), rng);

        //compare all the patches
        int i = 0; 
        int diff_total = 0;
        while(i < horizontal_problem.size() && i < vertical_problem.size()){

          //get the sum of the absolute difference between patches (SAD)
          int diff = (patches_vertical.row(vertical_problem.at(i)) - 
                      patches_horizontal.row(horizontal_problem.at(i))).cwiseAbs().sum();

          //track the best/worst difference between patches
          if(diff < min_values.at(i));
            min_values.at(i) = diff;
          if(diff > max_values.at(i))
            max_values.at(i) = diff;

          //get the tottal difference between the patches, how we make our selection
          diff_total += diff;
          i += 1;
        }

        //check if this new set is better
        if(diff_total < best_cost){
          best_cost = diff_total;
          best_vector = vector<int>(horizontal_problem);
        }

      }

      //when the loop exits, log the best outcome we found
      int i = 0;
      while(i < best_vector.size() && i < vertical_problem.size()){
        int j = best_vector.at(i);
        int k = vertical_problem.at(i);

        //use the vectors to track results
        range_horizontal_matched.push_back(range_horizontal.at(j));
        bearing_horizontal_matched.push_back(bearing_horizontal.at(j));
        range_vertical_matched.push_back(range_vertical.at(k));
        x_vertical_matched.push_back(x_vertical.at(k));
        matching_confidence.push_back((float) min_values.at(i) / (float) max_values.at(i));
        i += 1;
      }
    }

    //build an eigen matrix from the vectors
    //the matrix column structure is [range_horizontal; bearing_horizontal; x_vertical; range_vertical]
    Eigen::MatrixXf output_matrix(range_horizontal_matched.size(),4);

    float* ptr_1 = &range_horizontal_matched[0];
    Eigen::Map<Eigen::VectorXf> col_1(ptr_1, range_horizontal_matched.size()); 

    float* ptr_2 = &bearing_horizontal_matched[0];
    Eigen::Map<Eigen::VectorXf> col_2(ptr_2, bearing_horizontal_matched.size()); 

    float* ptr_3 = &x_vertical_matched[0];
    Eigen::Map<Eigen::VectorXf> col_3(ptr_3, x_vertical_matched.size()); 

    float* ptr_4 = &range_vertical_matched[0];
    Eigen::Map<Eigen::VectorXf> col_4(ptr_4, range_vertical_matched.size()); 

    output_matrix.col(0) = col_1;
    output_matrix.col(1) = col_2;
    output_matrix.col(2) = col_3;
    output_matrix.col(3) = col_4;

    return output_matrix;

}
    


PYBIND11_MODULE(match, m)
{
  m.def("matchFeatures", &matchFeatures);
}