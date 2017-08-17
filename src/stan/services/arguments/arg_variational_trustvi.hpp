#ifndef STAN_SERVICES_ARGUMENTS_VARIATIONAL_TRUSTVI_HPP
#define STAN_SERVICES_ARGUMENTS_VARIATIONAL_TRUSTVI_HPP

#include <stan/services/arguments/categorical_argument.hpp>

namespace stan {

  namespace services {

    class arg_variational_trustvi: public categorical_argument {
    public:
      arg_variational_trustvi() {
        _name = "trustvi";
        _description = "fast black-box variational inference through stochastic trust-region optimization";
      }
    };
  }  // services
}  // stan

#endif

