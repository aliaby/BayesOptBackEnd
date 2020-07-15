#include "BayesOptimizer.hpp"
#include <unordered_map>
#include <vector>
#include <queue>
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
#include <math.h>
#include <limbo/serialize/text_archive.hpp>
namespace BayesianOptimization
{

  // vector<double> normalize_array_method1(const vector<double> &arr) {
  //     vector<double> tmp(arr), ret(arr.size());
  //
  //     sort(tmp.begin(), tmp.end());
  //
  //     transform(arr.cbegin(), arr.cend(), ret.begin(), [&tmp](int x) {
  //         return distance(tmp.begin(), lower_bound(tmp.begin(), tmp.end(), x));
  //     });
  //
  //     return ret;
  //   }

  double normalCDF(double value)
  {
     return 0.5 * erfc(-value * M_SQRT1_2);
  }

  BayesOptimizer::BayesOptimizer(std::map<std::string, std::vector< std::vector<int16_t> > > _space)
  {
    this->config_space = _space;
    this->build_search_space();
    this->gp = new GP_t(this->feature_length, 1);
    // std::cout << this->search_space[100000] << std::endl;
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

  void BayesOptimizer::fit(std::vector<int> xs, std::vector<double> ys){
    best = std::max(*std::max_element(ys.begin(), ys.end()), best);
    std::vector<Eigen::VectorXd> samples;
    std::vector<Eigen::VectorXd> results;
    for(auto const& index:xs)
      samples.push_back(this->search_space[index]);
    for(auto const& y:ys)
      results.push_back(limbo::tools::make_vector(y));

    this->gp->compute(samples, results);
  }

  std::vector<int> BayesOptimizer::next_batch(int batch_size, std::vector<int> visited){
    typedef std::pair<int, double> PAIR;

    auto cmp = [](const PAIR &a, const PAIR &b) {
      return a.second > b.second;
    };

    std::priority_queue<PAIR, std::vector<PAIR>, std::function<bool(const PAIR, const PAIR)> > maximums(cmp);

    for (unsigned index = 0; index < this->space_length; index++) {
      if (std::find(visited.begin(), visited.end(), index) != visited.end())
        continue;

      Eigen::VectorXd mu;
      double sigma;
      std::tie(mu, sigma) = gp->query(this->search_space[index]);
      double result = normalCDF((mu[0] - this->best) / (sigma + 1e-9));

      if(maximums.size() < batch_size){
        maximums.push(PAIR (index, result));
      }
      else if(result > maximums.top().second){
        maximums.pop();
        maximums.push(PAIR (index, result));
      }
    }

    std::vector<int> res;
    while(!maximums.empty()){
      res.push_back(maximums.top().first);
      maximums.pop();
    }

    return res;
  }

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
