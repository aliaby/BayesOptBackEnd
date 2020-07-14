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

  private:
	  std::map<std::string, std::vector< std::vector<int16_t> > > config_space;
	  // std::vector<std::vector<std::vector<int16_t> > > search_space;
    std::vector<std::vector<int16_t> > search_space;

	  unsigned space_length = 1;
	  void build_search_space();
    // std::vector<int> d_index_space(std::vector<std::vector<std::vector<int16_t> > > space);
  };
}
