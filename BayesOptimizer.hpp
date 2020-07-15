#include <vector>
#include <unordered_map>
#include <map>

#include <fstream>
#include <limbo/kernel/exp.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/tools.hpp>
#include <limbo/tools/macros.hpp>

#include <limbo/serialize/text_archive.hpp>

#define DEBUG0

namespace BayesianOptimization
{

  class BayesOptimizer{
  public:
      BayesOptimizer(std::map<std::string, std::vector< std::vector<int16_t> > > _space);
      ~BayesOptimizer();
      void print_space();
      void fit(std::vector<int> xs, std::vector<double> ys);
      std::vector<int> next_batch(int batch_size, std::vector<int> visited);

  private:
	  std::map<std::string, std::vector< std::vector<int16_t> > > config_space;
	  // std::vector<std::vector<std::vector<int16_t> > > search_space;
    std::vector<Eigen::VectorXd> search_space;

    unsigned space_length = 1;
    unsigned feature_length = 0;
    double best = 0;


    struct Params {
      struct kernel_exp {
          BO_PARAM(double, sigma_sq, 1.0);
          BO_PARAM(double, l, 0.2);
      };

      struct kernel : public limbo::defaults::kernel {};
      struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {};
      struct opt_rprop : public limbo::defaults::opt_rprop {};
      };

      using Kernel_t  = limbo::kernel::Exp<Params>;
      using Mean_t    = limbo::mean::Data<Params>;
      using GP_t      = limbo::model::GP<Params, Kernel_t, Mean_t>;
      GP_t* gp;


  	  void build_search_space();
    // std::vector<int> d_index_space(std::vector<std::vector<std::vector<int16_t> > > space);
  };
}
