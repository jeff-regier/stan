#ifndef STAN_VARIATIONAL_TRUSTVI_HPP
#define STAN_VARIATIONAL_TRUSTVI_HPP

#include <stan/math.hpp>
#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/io/dump.hpp>
#include <stan/services/io/write_iteration.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/variational/print_progress.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>
#include <limits>
#include <numeric>
#include <ostream>
#include <vector>
#include <queue>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <stan/model/hessian_times_vector.hpp>
#include <stan/model/hessian.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

using namespace std;
using namespace Eigen;
using boost::math::normal;

typedef long long int64;
typedef unsigned long long uint64;


#include <cstdlib>

extern "C" {
      #include "math.h"
      #include "string.h"
      // trlib is available from https://github.com/felixlen/trlib with an MIT license
      #include "trlib.h"

      // blas
      void daxpy_(trlib_int_t *n, trlib_flt_t *alpha, trlib_flt_t *x, trlib_int_t *incx, trlib_flt_t *y, trlib_int_t *incy);
      void dscal_(trlib_int_t *n, trlib_flt_t *alpha, trlib_flt_t *x, trlib_int_t *incx);
      void dcopy_(trlib_int_t *n, trlib_flt_t *x, trlib_int_t *incx, trlib_flt_t *y, trlib_int_t *incy);
      trlib_flt_t dnrm2_(trlib_int_t *n, trlib_flt_t *x, trlib_int_t *incx);
      trlib_flt_t ddot_(trlib_int_t *n, trlib_flt_t *x, trlib_int_t *incx, trlib_flt_t *y, trlib_int_t *incy);
      // lapack
      void dgemv_(char *trans, trlib_int_t *m, trlib_int_t *n, trlib_flt_t *alpha, trlib_flt_t *a, trlib_int_t *lda, trlib_flt_t *x, trlib_int_t *incx, trlib_flt_t *beta, trlib_flt_t *y, trlib_int_t *incy);
}

namespace stan {
namespace variational {


struct trlib_qpdata {
       trlib_int_t n;                     ///< dimension of problem
       trlib_int_t maxiter;               ///< maximum number of Krylov subspace iterations
       trlib_int_t *iwork;                ///< integer work space
       trlib_flt_t *fwork;                ///< floating point workspace
       trlib_flt_t *gradient;             ///< gradient of QP
       trlib_int_t hotstart;              ///< flag that determines if hotstarted or not
       trlib_int_t iter;                  ///< iteration counter
       trlib_flt_t *g;                    ///< gradient of Krylov iteration
       trlib_flt_t *gm;                   ///< previous gradient of Krylov iteration
       trlib_flt_t *p;                    ///< direction
       trlib_flt_t *Hp;                   ///< hessian product
       trlib_flt_t *Q;                    ///< matrix with Lanczos directions
 };


/**
 * TrustVI
 *
 * @tparam Model                 class of model
 * @tparam BaseRNG               class of random number generator
 */
template <class Model, class BaseRNG>
class trustvi {
public:
/**
 * Constructor
 *
 * @param  m                    stan model
 * @param  cont_params          initialization of continuous parameters
 * @param  rng                  random number generator
 * @param  n_monte_carlo_grad   number of samples for gradient computation
 * @param  n_monte_carlo_elbo   number of samples for ELBO computation
 * @param  eval_elbo            evaluate ELBO at every "eval_elbo" iters
 * @param  n_posterior_samples  number of samples to draw from posterior
 * @throw  runtime_error   if n_monte_carlo_grad is not positive
 * @throw  runtime_error   if n_monte_carlo_elbo is not positive
 * @throw  runtime_error   if eval_elbo is not positive
 * @throw  runtime_error   if n_posterior_samples is not positive
 */
trustvi(Model& m,
     VectorXd& cont_params,
     BaseRNG& rng,
     int n_monte_carlo_grad,
     int n_monte_carlo_elbo,
     int eval_elbo,
     int n_posterior_samples)
  : model_(m),
    cont_params_(cont_params),
    rng_(rng),
    x_(VectorXd::Zero(2 * cont_params.size())),
    // divide by 3 to keep amount of parallelism the same as the gradient
    n_monte_carlo_hess_(n_monte_carlo_grad / 3),
    n_monte_carlo_grad_(n_monte_carlo_grad),
    n_monte_carlo_elbo_(n_monte_carlo_elbo),
    n_monte_carlo_grad_reps_(1),
    eval_elbo_(eval_elbo),
    n_posterior_samples_(n_posterior_samples),
    D_(cont_params.size()) {
  static const char* function = "stan::variational::trustvi";
  math::check_positive(function,
                       "Number of Monte Carlo samples for gradients",
                       n_monte_carlo_grad_);
  math::check_positive(function,
                       "Number of Monte Carlo samples for ELBO",
                       n_monte_carlo_elbo_);
  math::check_positive(function,
                       "Evaluate ELBO at every eval_elbo iteration",
                       eval_elbo_);
  math::check_positive(function,
                       "Number of posterior samples for output",
                       n_posterior_samples_);
}

// for integration with trlib
int prepare_qp(trlib_int_t n, trlib_int_t maxiter,
       const double *gradient,
       struct trlib_qpdata *data) {
    data->n = n;
    data->maxiter = maxiter;
    data->gradient = (double *)gradient;
    data->hotstart = 0;
    trlib_int_t iwork_size, fwork_size, h_pointer;
    trlib_krylov_memory_size(maxiter, &iwork_size, &fwork_size, &h_pointer);
    data->iwork = new trlib_int_t[iwork_size];
    data->fwork = new trlib_flt_t[fwork_size];
    data->g = new double[n];
    data->gm = new double[n];
    data->p = new double[n];
    data->Hp = new double[n];
    data->Q = new double[(maxiter+1)*n];
    data->iter = 0;
    return 0;
}

// for integration with trlib
int destroy_qp(struct trlib_qpdata *data) {
    free(data->iwork);
    free(data->fwork);
    free(data->g);
    free(data->gm);
    free(data->p);
    free(data->Hp);
    free(data->Q);
    return 0;
}

// for integration with trlib
int solve_qp(struct trlib_qpdata *data, trlib_flt_t radius, double *sol, double *lam) {
    // some default settings
    trlib_int_t equality = 0;
    trlib_int_t maxlanczos = 100;
    trlib_int_t ctl_invariant = 0;
    trlib_int_t refine = 1;
    trlib_int_t verbose = 1;
    trlib_int_t unicode = 0;
    trlib_flt_t tol_rel_i = 1e-2;
    trlib_flt_t tol_abs_i = 0.0;
    trlib_flt_t tol_rel_b = 1e-1;
    trlib_flt_t tol_abs_b = 0.0;
    trlib_flt_t obj_lo = -1e20;
    trlib_int_t convexify = 1;
    trlib_int_t earlyterm = 0;

    trlib_int_t ret = 0;

    trlib_int_t n = data->n;
    trlib_int_t init = 0, inc = 1, itp1 = 0;
    trlib_flt_t minus = -1.0, one = 1.0, z = 0.0;
    if(!data->hotstart) {
        init = TRLIB_CLS_INIT;
        trlib_krylov_prepare_memory(data->maxiter, data->fwork);
    }
    else { init = TRLIB_CLS_HOTSTART_G; }

    trlib_flt_t v_dot_g = 0.0, p_dot_Hp = 0.0, flt1, flt2, flt3;
    trlib_int_t action, ityp;

    trlib_int_t iwork_size, fwork_size, h_pointer;
    trlib_krylov_memory_size(data->maxiter, &iwork_size, &fwork_size, &h_pointer);

    trlib_flt_t *temp = new trlib_flt_t[n];

    int hv_count = 0;

    while(1) {
        ret = trlib_krylov_min(init, radius, equality, data->maxiter, maxlanczos,
                tol_rel_i, tol_abs_i, tol_rel_b, tol_abs_b,
                TRLIB_EPS*TRLIB_EPS, obj_lo, ctl_invariant, convexify, earlyterm,
                v_dot_g, v_dot_g, p_dot_Hp, data->iwork, data->fwork,
                refine, verbose, unicode, (char *)"", stdout, NULL,
                &action, &(data->iter), &ityp, &flt1, &flt2, &flt3);
        init = 0;
        switch(action) {
            case TRLIB_CLA_INIT:
                memset(sol, 0, n*sizeof(trlib_flt_t)); memset(data->gm, 0, n*sizeof(trlib_flt_t));
                dcopy_(&n, data->gradient, &inc, data->g, &inc);
                v_dot_g = ddot_(&n, data->g, &inc, data->g, &inc);
                // p = -g
                dcopy_(&n, data->g, &inc, data->p, &inc); dscal_(&n, &minus, data->p, &inc);
                // Hp = H*p
                hessvec_trlib(data->p, data->Hp);
                hv_count++;
                p_dot_Hp = ddot_(&n, data->p, &inc, data->Hp, &inc);
                // Q(0:n) = g
                dcopy_(&n, data->g, &inc, data->Q, &inc);
                // Q(0:n) = g/sqrt(<g,g>)
                flt1 = 1.0/sqrt(v_dot_g); dscal_(&n, &flt1, data->Q, &inc);
                break;
            case TRLIB_CLA_RETRANSF:
                itp1 = data->iter+1;
                // s = Q_i * h_i
                dgemv_((char *)"N", &n, &itp1, &one, data->Q, &n, data->fwork+h_pointer, &inc, &z, sol, &inc);
                break;
            case TRLIB_CLA_UPDATE_STATIO:
                // s += flt1*p
                if (ityp == TRLIB_CLT_CG) { daxpy_(&n, &flt1, data->p, &inc, sol, &inc); };
                break;
            case TRLIB_CLA_UPDATE_GRAD:
                if (ityp == TRLIB_CLT_CG) {
                    // Q(iter*n:(iter+1)*n) = flt2*g
                    dcopy_(&n, data->g, &inc, data->Q+(data->iter)*n, &inc);
                    dscal_(&n, &flt2, data->Q+(data->iter)*n, &inc);
                    // gm = g; g += flt1*Hp
                    dcopy_(&n, data->g, &inc, data->gm, &inc);
                    daxpy_(&n, &flt1, data->Hp, &inc, data->g, &inc);
                }
                if (ityp == TRLIB_CLT_L) {
                    // s = Hp + flt1*g + flt2*gm
                    dcopy_(&n, data->Hp, &inc, sol, &inc);
                    daxpy_(&n, &flt1, data->g, &inc, sol, &inc);
                    daxpy_(&n, &flt2, data->gm, &inc, sol, &inc);
                    // gm = flt3*g
                    dcopy_(&n, data->g, &inc, data->gm, &inc); dscal_(&n, &flt3, data->gm, &inc);
                    // g = s
                    dcopy_(&n, sol, &inc, data->g, &inc);
                }
                v_dot_g = ddot_(&n, data->g, &inc, data->g, &inc);
                break;
            case TRLIB_CLA_UPDATE_DIR:
                if (ityp == TRLIB_CLT_CG) {
                    // p = -g + flt2 * p
                    dscal_(&n, &flt2, data->p, &inc);
                    daxpy_(&n, &minus, data->g, &inc, data->p, &inc);
                }
                if (ityp == TRLIB_CLT_L) {
                    // p = flt1*g
                    dcopy_(&n, data->g, &inc, data->p, &inc);
                    dscal_(&n, &flt1, data->p, &inc);
                }
                // Hp = H*p
                hessvec_trlib(data->p, data->Hp);
                hv_count++;
                p_dot_Hp = ddot_(&n, data->p, &inc, data->Hp, &inc);
                if ( ityp == TRLIB_CLT_L) {
                    // Q(iter*n:(iter+1)*n) = p
                    dcopy_(&n, data->p, &inc, data->Q+(data->iter)*n, &inc);
                }
                break;
            case TRLIB_CLA_CONV_HARD:
                itp1 = data->iter+1;
                // temp = H*s
                hessvec_trlib(sol, temp);
                hv_count++;
                // temp = H*s + g
                daxpy_(&n, &one, data->gradient, &inc, temp, &inc);
                // temp = H*s + g + flt1*s
                daxpy_(&n, &flt1, sol, &inc, temp, &inc);
                v_dot_g = ddot_(&n, temp, &inc, temp, &inc);
                break;
            case TRLIB_CLA_NEW_KRYLOV:
                printf("Hit invariant Krylov subspace. Please implement proper reorthogonalization!");
                break;
            case TRLIB_CLA_OBJVAL: ;
                printf("Objective value requested!\n");
                break;
        }
        if( ret < 10 ) {
            cout << "trlib ret = " << ret << endl;
            break;
        }
    }
    *lam = data->fwork[7];

    printf("trlib took %d hessian-vector multiplies\n", hv_count);

    free(temp);
    if(!data->hotstart) { data->hotstart = 1; }
    return ret;
}

// for integration with trlib
void hessvec_trlib(const double *d_trlib, double *Hd_trlib) {
    VectorXd d(2 * D_);
    VectorXd Bd(2 * D_);

    for (int i=0; i < 2 * D_; i++)
        d(i) = d_trlib[i];

    BaseRNG rng0 = rng_;
    mc_hessian_times_vector(d, Bd);
    Bd *= -1;   // maximizing the elbo is minimizing the negative elbo
    rng_ = rng0;

    for (int i=0; i < 2 * D_; i++)
        Hd_trlib[i] = Bd(i);
}

// for integration with trlib
double gltr(const double radius,
          VectorXd& g,
          const MatrixXd& Minv, //for preconditioning
          VectorXd& s) {
    for (int i=0; i < g.size(); i++)
        data_trlib_->gradient[i] = g(i);

    double sol_trlib[2 * D_];
    double lam_trlib = 0.0;

    printf("trlib attempting to solve trust region problem with radius %f\n", radius);
    solve_qp(data_trlib_, radius, sol_trlib, &lam_trlib);
    printf("Got lagrange multiplier %f\n", lam_trlib);

    for (int i=0; i < g.size(); i++)
        s(i) = sol_trlib[i];

    // predicted change (should be negative)
    return data_trlib_->fwork[8];
}

/**
 * Calculates the "blackbox" hessian-vector products with respect to both the
 * location vector (mu) and the log-std vector (omega).
 */
void mc_hessian_times_vector(VectorXd& v, VectorXd& hv) {
    static const char* function =
      "stan::variational::trustvi::mc_hessian_times_vector";
    VectorXd mu = x_.block(0, 0, D_, 1);
    VectorXd omega = x_.block(D_, 0, D_, 1);

    VectorXd v1 = v.block(0, 0, D_, 1);
    VectorXd v2 = v.block(D_, 0, D_, 1);

    VectorXd hv1 = VectorXd::Zero(D_);
    VectorXd hv2 = VectorXd::Zero(D_);;

    VectorXd normal_sample  = VectorXd::Zero(D_);
    VectorXd zeta = VectorXd::Zero(D_);

    stan::math::check_finite(function, "the factor v", v);

    for (int i = 0; i < n_monte_carlo_hess_; i++) {
        // Draw from standard normal and transform to real-coordinate space
        for (int d = 0; d < D_; ++d)
            normal_sample(d) = stan::math::normal_rng(0, 1, rng_);
        zeta = normal_sample.array().cwiseProduct(omega.array().exp()).matrix() + mu;

        VectorXd h11v1 = VectorXd::Zero(D_);
        VectorXd h12v2 = VectorXd::Zero(D_);
        VectorXd h21v1 = VectorXd::Zero(D_);
        VectorXd h22v2 = VectorXd::Zero(D_);
        VectorXd zeta_grad = VectorXd::Zero(D_);
        VectorXd v2_scaled(D_);
        double dummy_var;

        ArrayXd stdev = omega.array().exp();
        DiagonalMatrix<double, Dynamic, Dynamic> stdev_diag = stdev.matrix().asDiagonal();

        v2_scaled = normal_sample.array().cwiseProduct(stdev).matrix().asDiagonal() * v2;

        stan::model::hessian_times_vector(model_, zeta, v1, dummy_var, h11v1);
        stan::model::hessian_times_vector(model_, zeta, v2_scaled, dummy_var, h12v2);
        stan::model::hessian_times_vector(model_, zeta, v1, dummy_var, h21v1);

        stan::math::check_finite(function, "h11v1", h11v1);
        stan::math::check_finite(function, "h12v2", h12v2);
        stan::math::check_finite(function, "h21v1", h21v1);

        // unnecessary. could cache gradients
        stan::model::gradient(model_, zeta, dummy_var, zeta_grad);
        stan::math::check_finite(function, "zeta_grad", zeta_grad);

        h21v1.array() = h21v1.array().cwiseProduct(omega.array().exp()).cwiseProduct(normal_sample.array());
        h22v2.array() = stdev.array().cwiseProduct(normal_sample.array()).cwiseProduct(h12v2.array());
        h22v2.array() += zeta_grad.array().cwiseProduct(omega.array().exp()).cwiseProduct(normal_sample.array()).cwiseProduct(v2.array());

        hv1 += h11v1;
        hv1 += h12v2;
        hv2 += h21v1;
        hv2 += h22v2;
    }

    hv.block(0, 0, D_, 1) = hv1;
    hv.block(D_, 0, D_, 1) = hv2;
    hv /= static_cast<double>(n_monte_carlo_hess_);
}

/**
 * Calculates the "blackbox" gradient with respect to both the
 * location vector (mu) and the log-std vector (omega).
 */
double mc_grad(VectorXd& g, int n) {
    static const char* function = "stan::variational::trustvi::mc_grad";

    VectorXd mu = x_.block(0, 0, D_, 1);
    VectorXd omega = x_.block(D_, 0, D_, 1);

    VectorXd mu_grad = VectorXd::Zero(D_);
    VectorXd omega_grad = VectorXd::Zero(D_);

    VectorXd normal_sample  = VectorXd::Zero(D_);
    VectorXd zeta = VectorXd::Zero(D_);
    VectorXd zeta_grad = VectorXd::Zero(D_);

    double a_lp;
    double elbo = 0.0;

    for (int i = 0; i < n; i++) {
        for (int d = 0; d < D_; ++d)
            normal_sample(d) = stan::math::normal_rng(0, 1, rng_);

        zeta = normal_sample.array().cwiseProduct(omega.array().exp()).matrix() + mu;

        stan::model::gradient(model_, zeta, a_lp, zeta_grad);
        elbo += a_lp;

        stan::math::check_finite(function, "zeta_grad", zeta_grad);

        mu_grad += zeta_grad;
        omega_grad.array() += zeta_grad.array().cwiseProduct(normal_sample.array());
    }

    mu_grad /= static_cast<double>(n);
    omega_grad /= static_cast<double>(n);
    elbo /= n;

    ArrayXd stdev = omega.array().exp();
    DiagonalMatrix<double, Dynamic, Dynamic> stdev_diag = stdev.matrix().asDiagonal();
    omega_grad.array() = omega_grad.array().cwiseProduct(stdev);

    // entropy
    elbo += 0.5 * D_ * (1.0 + stan::math::LOG_TWO_PI) + omega.sum();
    omega_grad.array() += 1.0;  // add entropy gradient (unit)

    g.block(0, 0, D_, 1) = mu_grad;
    g.block(D_, 0, D_, 1) = omega_grad;

    return elbo;
}

/**
 * Calculates the Evidence Lower BOund (ELBO) by sampling from
 * the variational distribution and then evaluating the log joint,
 * adjusted by the entropy term of the variational distribution.
 */
double mc_elbo(int n, BaseRNG& rng, VectorXd& zeta) {
    static const char* function = "stan::variational::trustvi::mc_elbo";

    VectorXd mu = x_.block(0, 0, D_, 1);
    VectorXd omega = x_.block(D_, 0, D_, 1);

    double elbo = 0.0;
    int n_dropped_evaluations = 0;

    for (int i = 0; i < n;) {
        // Draw from standard normal and transform to real-coordinate space
        for (int d = 0; d < D_; ++d)
            zeta(d) = stan::math::normal_rng(0, 1, rng);
        zeta = zeta.array().cwiseProduct(omega.array().exp()).matrix() + mu;

        try {
            std::stringstream ss;
            double a_lp = model_.template log_prob<false, true>(zeta, &ss);
            if (ss.str().length() > 0)
                    cout << ss.str() << endl;
            stan::math::check_finite(function, "log_prob", a_lp);
            elbo += a_lp;
            ++i;
        } catch (const std::domain_error& e) {
            cout << "infinite log prob in mc_elbo" << endl;
            ++n_dropped_evaluations;
            break;
        }
    }

    elbo /= (n - n_dropped_evaluations);

    // entropy
    elbo += 0.5 * D_ * (1.0 + stan::math::LOG_TWO_PI) + omega.sum();

    return elbo;
}

// Stochastic approximation of the change in ELBO values between x_ and x1.
// epsilon and zeta are just used as working memory here.
pair<double, double> mc_elbo_change(const VectorXd& x1, int n,
                      VectorXd& epsilon, VectorXd& zeta) {
    static const char* function = "stan::variational::trustvi::mc_elbo_change";

    VectorXd mu0 = x_.block(0, 0, D_, 1);
    VectorXd omega0 = x_.block(D_, 0, D_, 1);
    VectorXd mu1 = x1.block(0, 0, D_, 1);
    VectorXd omega1 = x1.block(D_, 0, D_, 1);

    double cur_change;
    double sum_change = 0.0;
    double sum_sq_change = 0.0;

    int n_dropped_evaluations = 0;

    for (int i = 0; i < n; i++) {
        // Draw from standard normal and transform to real-coordinate space
        for (int d = 0; d < D_; ++d)
            epsilon(d) = stan::math::normal_rng(0, 1, rng_);

        try {
            std::stringstream ss;

            zeta << epsilon.array().cwiseProduct(omega0.array().exp()).matrix() + mu0;
            double lp0 = model_.template log_prob<false, true>(zeta, &ss);
            stan::math::check_finite(function, "lp0", lp0);

            zeta << epsilon.array().cwiseProduct(omega1.array().exp()).matrix() + mu1;
            double lp1 = model_.template log_prob<false, true>(zeta, &ss);
            stan::math::check_finite(function, "lp1", lp1);

            cur_change = lp1 - lp0;
            sum_change += cur_change;
            sum_sq_change += pow(cur_change, 2);
            stan::math::check_finite(function, "sum_sq_change", sum_sq_change);
        }
        catch (const std::domain_error& e) {
            cout << "infinite log prob in mc_elbo_change" << endl;
            ++n_dropped_evaluations;
            if (n_dropped_evaluations > 0) {
                const char* name = "The number of dropped evaluations";
                const char* msg1 = "has reached its maximum amount.";
                stan::math::domain_error(function, name, n / 10.0, msg1);
            }
        }
    }

    int n0 = n - n_dropped_evaluations;
    double sample_mean = sum_change / n0;
    double ss = sum_sq_change - (sum_change * sum_change / n0);
    double one_sample_var =    ss / (n0 - 1);

    // entropy
    sample_mean += omega1.sum() - omega0.sum();

    return make_pair(sample_mean, one_sample_var);
}

double elbo_sample_size(const double& fprime,
                        const double& radius,
                        const double& alpha,
                        const double& c,
                        const double& accept_threshold,
                        const double& one_sample_var) {
    double radius_sq = pow(radius, 2);
    double f1t1 = log(4 * fprime + 15 * alpha * radius_sq);
    double f1t2 = -log(4 * c + 3 * alpha);
    double f1t3 = -log(radius_sq);
    double f2 = one_sample_var / pow(fprime - accept_threshold, 2);
    return (f1t1 + f1t2 + f1t3) * f2;
}


/*
 * Here we (retrospectively) find a lower bound on the size sample we needed,
 * though bisection.
 */
double sufficient_elbo_sample(const double& radius,
                              const double& alpha,
                              const double& c,
                              const double& accept_threshold,
                              const double& one_sample_var) {
    double fprime_base = -15.0 * radius * radius * alpha / 4.0;
    double fprime_offset1 = 10000.0;
    double fprime1 = fprime_base + fprime_offset1;
    double n1 = elbo_sample_size(fprime1, radius, alpha, c,
                                 accept_threshold, one_sample_var);
    double n1b = elbo_sample_size(fprime1 + 1e-3, radius, alpha, c,
                                 accept_threshold, one_sample_var);
    double deriv1 = (n1b - n1) / 1e-3;

    double fprime2 = 0.0;
    double fprime_offset2 = 0.0;
    double deriv2 = 0.0;
    double n2, n2b;

    for (int i = 0; ; i++) {
        fprime_offset2 = fprime_offset1 * (deriv1 > 0 ? 2.0 : 0.5);
        fprime2 = fprime_base + fprime_offset2;
        n2 = elbo_sample_size(fprime2, radius, alpha, c,
                                     accept_threshold, one_sample_var);
        n2b = elbo_sample_size(fprime2 + 1e-3, radius, alpha, c,
                                     accept_threshold, one_sample_var);
        deriv2 = (n2b - n2) / 1e-3;

        if ((deriv2 < 1-8 && deriv1 > -1e-8) || (deriv2 > -1e-8 && deriv1 < 1e-8))
            break;

        fprime_offset1 = fprime_offset2;
        fprime1 = fprime2;
        deriv1 = deriv2;
        n1 = n2;
        n1b = n2b;

        if (i == 15) {
            cout << "too many iterations in sufficient_elbo_sample" << endl;
            break;
        }
    }

    return max(n1, n2);
}

double adatrust_step(VectorXd& cur_grad, VectorXd& g, VectorXd& s,
                     const double radius, const MatrixXd& Minv) {
    int n = n_monte_carlo_grad_ * n_monte_carlo_grad_reps_;
    cout << "grad reps: " << n_monte_carlo_grad_reps_ << endl;

    g.fill(0.0);
    double g_terms_sqnorm = 0.0;
    BaseRNG rng0 = rng_;
    for (int i = 0; i < n; i++) {
        mc_grad(cur_grad, 1);
        cur_grad *= -1.0; // maximizing the elbo is minimizing the negative elbo
        g += cur_grad;
        g_terms_sqnorm += cur_grad.squaredNorm();
    }

    double theta_all_denorm = g.squaredNorm() - g_terms_sqnorm;
    double theta_all = theta_all_denorm / (n * (n - 1));

    rng_ = rng0;
    double sum_theta_i = 0.0;
    double sum_theta_i_sq = 0.0;
    for (int i = 0; i < n; i++) {
        // We could have cached the results of these calls to mc_grad a few
        // lines earlier. I'm recomputing them here because its easier.
        mc_grad(cur_grad, 1);
        cur_grad *= -1.0; // maximizing the elbo is minimizing the negative elbo

        double theta_i_denorm = theta_all_denorm - 2 * cur_grad.dot(g) + 2 * cur_grad.squaredNorm();
        double theta_i = theta_i_denorm  / ((n - 1) * (n - 2));

        sum_theta_i += theta_i;
        sum_theta_i_sq += pow(theta_i, 2.0);
    }

    double jackknife_var = ((n - 1.0) / n) * (sum_theta_i_sq - (pow(sum_theta_i, 2) / n));

    if (n_monte_carlo_grad_reps_ < 32 && theta_all - sqrt(jackknife_var) < 0.0) {
        n_monte_carlo_grad_reps_ *= 2;
        n_monte_carlo_hess_ *= 2;
    }
    else if (n_monte_carlo_grad_reps_ > 1 && theta_all - 3 * sqrt(jackknife_var) > 0.0) {
        n_monte_carlo_grad_reps_ /= 2;
        n_monte_carlo_hess_ /= 2;
    }

    cout << "jackknife estimate: " << theta_all  << endl;
    cout << "jackknife sd: " << sqrt(jackknife_var) << endl;

    g /= n;
    cout << "grad norm: " << g.norm() << endl;
    double predicted_change = gltr(radius, g, Minv, s);
    cout << "trlib change: " << predicted_change << endl;
    assert(predicted_change <= 0.0);
    return predicted_change;
}

/* 
 *  The optimization algorithm that TrustVI is based on.
 *  SGD is to ADVI as AdaTrust is to TrustVI.
 */
void adatrust() {
    const double eta = 0.25;

    VectorXd s(2 * D_);
    VectorXd g(2 * D_);
    VectorXd cur_grad(2 * D_);
    VectorXd zeta(D_);
    VectorXd epsilon(D_);

    const int max_elbo_n = pow(2, 14);
    int elbo_reps = 1;

    cout << "\n-------------------\n" << endl;

    double radius = 1e-1;
    const double max_radius = 1e4;
    const double min_radius = 1e-4;

    MatrixXd Minv = MatrixXd::Identity(2 * D_, 2 * D_);

    bool always_grows = true, always_shrinks = true;
    int hotstart_streak = 0;

    for(int iter = 0; iter < 500; iter++) {
        double cur_elbo = mc_elbo(n_monte_carlo_elbo_, rng_, zeta);
        cout << "Iteration " << setw(2) << iter << ". ";
        cout << "Log joint probability = " << setw(10) << cur_elbo << endl;

        double predicted_change = adatrust_step(cur_grad, g, s, radius, Minv);

        try {
            // observed change
            VectorXd x1 = x_ + s;
            int elbo_n = elbo_reps * n_monte_carlo_grad_ / 2;
            cout << "elbo reps: " << elbo_reps << endl;
            pair<double, double> mv = mc_elbo_change(x1, elbo_n, epsilon, zeta);

            double measured_change = -mv.first;
            double elbo_change_var = mv.second;

            double rho = measured_change / min(-1e-12, predicted_change);

            cout << "Elbo increases by " << -measured_change << " ";
            cout << "(" << sqrt(elbo_change_var / elbo_n) << ").  ";
            cout << " Predicted " << -predicted_change << ".";
            cout << " ||s||=" << s.norm() << ".";
            cout << " radius=" << radius  << "." << endl;

            cout << "elbo n: " << elbo_n << endl;
            int needed_elbo_n = sufficient_elbo_sample(radius, 5.0, -2.0,
                              eta * predicted_change, elbo_change_var);
            cout << "elbo needed_n: " << needed_elbo_n << endl;

            double needed_elbo_n2 = -2.0 * elbo_change_var * log(1.0 - 0.5/0.75);
            needed_elbo_n2 /= g.squaredNorm() * pow(radius, 2);
            cout << "elbo needed n2: " << needed_elbo_n2 << endl;
            if (needed_elbo_n2 > needed_elbo_n)
                needed_elbo_n = static_cast<int>(needed_elbo_n2);

            // step evaluation and reporting
            if (rho >= eta && elbo_n <= max_elbo_n && !(always_shrinks || always_grows)) {
              cout << "accepting step\n" << endl;
              data_trlib_->hotstart = 0;
              hotstart_streak = 0;
              x_ += s;
            }
            else {
              cout << "rejecting step\n" << endl;
              if (hotstart_streak >= 2 && !(always_shrinks || always_grows)) {
                  data_trlib_->hotstart = 0;
                  hotstart_streak = 0;
              } else
                  hotstart_streak += 1;
            }

            cout << "rho: " << rho << endl;
            if (rho >= eta && s.norm() > 0.7 * radius) {
                cout << "growing trust region\n" << endl;
                radius = min(max_radius,  2.0 * radius);
                always_shrinks = false;
            }
            else {
                cout << "shrinking trust region\n" << endl;
                radius /= 2.0;
                always_grows = false;
            }

            if (needed_elbo_n > elbo_n && elbo_reps < 32)
                elbo_reps *= 2;
            else if (needed_elbo_n < 0.5 * elbo_n && elbo_reps > 1)
                elbo_reps /= 2;
        }
        catch (domain_error& de) {
            cout << "domain error: inf/nan calculation" << endl;
            // reject this step
            cout << "shrinking trust region\n" << endl;
            radius *= 0.5;
        }

        cout << "\n-------------------\n" << endl;
        cout.flush();

        if (radius < min_radius)
            break;
    }

    double final_elbo = mc_elbo(n_monte_carlo_elbo_, rng_, zeta);
    cout << "[TrustVI] final ELBO: " << final_elbo << endl;
}


/**
 * Runs TrustVI and writes to output.
 */
int run(double eta, bool dummy, int adapt_iterations,
        double tol_rel_obj, int max_iterations,
        interface_callbacks::writer::base_writer& message_writer,
        interface_callbacks::writer::base_writer& parameter_writer,
        interface_callbacks::writer::base_writer& diagnostic_writer) {
    x_.block(D_, 0, D_, 1) = -1 * VectorXd::Ones(D_);
    cout << "D: " << D_ << endl;

    data_trlib_ = new trlib_qpdata();
    trlib_int_t maxiter_trlib = 10 * D_; // maximum number of CG iterations
    double g_trlib[2 * D_];
    prepare_qp(2 * D_, maxiter_trlib, g_trlib, data_trlib_);

    adatrust();

    return stan::services::error_codes::OK;
}

protected:
    Model& model_;
    VectorXd& cont_params_;
    BaseRNG& rng_;
    VectorXd x_;
    int n_monte_carlo_hess_;
    int n_monte_carlo_grad_;
    int n_monte_carlo_elbo_;
    int n_monte_carlo_grad_reps_;
    int eval_elbo_;
    int n_posterior_samples_;
    int D_;
    trlib_qpdata* data_trlib_;
}; // end of class
}  // end of variational namespace
}  // end of stan namespace

#endif
