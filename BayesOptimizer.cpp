#include "BayesOptimizer.hpp"
#include <iostream>
#include <chrono>
#include <assert.h>
#include <limbo/kernel/exp.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/tools.hpp>
#include <limbo/tools/macros.hpp>

#include <limbo/serialize/text_archive.hpp>
namespace BayesianOptimization
{
  BayesOptimizer::BayesOptimizer(std::map<std::string, std::vector< std::vector<int16_t> > > _space)
  {
    this->config_space = _space;
    this->build_search_space();
    this->gp = new GP_t(this->feature_length, 1);
    std::cout << this->search_space[100000] << std::endl;
  }

  void BayesOptimizer::print_space(){
    for (auto const& pair: this->config_space) {
        std::cout << "{" << pair.first << ": " << pair.second[0][0] << "}\n";
    }
  }

  BayesOptimizer::~BayesOptimizer(){
    // for (std::vector<std::vector<int16_t> >::iterator it = this->search_space.begin(); it != this->search_space.end(); ++it) {
    //   (&(*it))->clear();
    // }
    this->search_space.clear();
    this->config_space.clear();
  }

  // std::vector<int> BayesOptimizer::d_index_space(std::vector<std::vector<std::vector<int16_t> > > space){
  //   std::vector<unsigned> dims;
  //   // buggy code...
  //   int len = 1;
  //   for (auto const& pair: this->config_space)
  //   {
  //     dims.push_back(len);
  //     len *= pair.second.size();
  //   }
  //   std::vector<int> res;
  //
  //   for (auto const& item:space) {
  //     std::vector<int> indices;
  //     int i = 0;
  //     for (auto const& pair: this->config_space) {
  //       int match_index = 0;
  //       for(auto const& it:pair.second){
  //         if(std::equal(it.begin(), it.end(), item[i].begin()))
  //           indices.push_back(match_index);
  //         match_index++;
  //       }
  //       std::cout<< "hey";
  //       i++;
  //     }
  //     assert(indices.size() == dims.size());
  //     int index = 0;
  //     for (size_t i = 0; i < indices.size(); i++) {
  //       index += indices[i]*dims[i];
  //     }
  //     res.push_back(index);
  //   }
  //   return res;
  //
  // }

  void BayesOptimizer::build_search_space(){
    std::vector<int16_t> dims;
    #ifdef DEBUG0
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    #endif

    for (auto const& pair: this->config_space)
    {
      this->space_length *= pair.second.size();
      dims.push_back(pair.second.size());
      this->feature_length += pair.second[0].size();
    }
    #ifdef DEBUG0
      std::cout<<(this->space_length) << std::endl;
    #endif

    for (unsigned i = 0; i < this->space_length; ++i)
    {
      std::vector<int> indices;
      int index = i;
      for(std::vector<int16_t>::iterator it = dims.begin(); it != dims.end(); ++it){
        indices.push_back(index%(*it));
        index /= *it;
      }
      index = 0;
      Eigen::VectorXd config(this->feature_length);
      int ii = 0;
      for(auto const& pair: this->config_space){
        for(auto const& it: pair.second[indices[index]]){
          config[ii] = it;
          ii++;
        }
        index++;
      }
      this->search_space.push_back(config);
    }
    #ifdef DEBUG0
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
      assert(this->search_space[0].size() == std::accumulate(dims.begin(), dims.end(), 0));
    #endif
  }

}
