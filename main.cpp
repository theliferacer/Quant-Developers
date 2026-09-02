#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>

namespace Quant {

    // Mathematical constant 1 / sqrt(2*pi)
    constexpr double INV_SQRT_2PI = 0.398942280401432677939946059934;
    // Mathematical constant 1 / sqrt(2)
    constexpr double INV_SQRT_2   = 0.707106781186547524400844362104;

    // Cumulative normal distribution function N(x) using std::erfc
    inline double norm_cdf(double x) {
        return 0.5 * std::erfc(-x * INV_SQRT_2);
    }

    // Standard normal probability density function N'(x)
    inline double norm_pdf(double x) {
        return INV_SQRT_2PI * std::exp(-0.5 * x * x);
    }

    struct Greeks {
        double delta;
        double gamma;
        double vega;   // Scaled per 1% move in implied vol (0.01)
        double theta;  // Scaled per calendar day (1/365)
        double rho;    // Scaled per 100 bps move in interest rate (0.01)
    };

    struct OptionResult {
        double price;
        Greeks greeks;
    };

    enum class OptionType { Call, Put };

    class BlackScholesEngine {
    public:
        BlackScholesEngine(double spot, double strike, double rate, double vol, double maturity)
            : S_(spot), K_(strike), r_(rate), sigma_(vol), T_(maturity) {
            d1_ = (std::log(S_ / K_) + (r_ + 0.5 * sigma_ * sigma_) * T_) / (sigma_ * std::sqrt(T_));
            d2_ = d1_ - sigma_ * std::sqrt(T_);
        }

        OptionResult price(OptionType type) const {
            OptionResult res;
            const double n_d1 = norm_cdf(d1_);
            const double n_d2 = norm_cdf(d2_);
            const double pdf_d1 = norm_pdf(d1_);
            const double discount = std::exp(-r_ * T_);
            const double sqrt_T = std::sqrt(T_);

            // Gamma and Vega are identical for calls and puts
            res.greeks.gamma = pdf_d1 / (S_ * sigma_ * sqrt_T);
            res.greeks.vega  = 0.01 * (S_ * sqrt_T * pdf_d1);

            if (type == OptionType::Call) {
                res.price = S_ * n_d1 - K_ * discount * n_d2;
                res.greeks.delta = n_d1;
                res.greeks.theta = (-(S_ * sigma_ * pdf_d1) / (2.0 * sqrt_T) - r_ * K_ * discount * n_d2) / 365.0;
                res.greeks.rho   = (0.01 * K_ * T_ * discount * n_d2);
            } else {
                const double n_md1 = norm_cdf(-d1_);
                const double n_md2 = norm_cdf(-d2_);
                res.price = K_ * discount * n_md2 - S_ * n_md1;
                res.greeks.delta = n_d1 - 1.0;
                res.greeks.theta = (-(S_ * sigma_ * pdf_d1) / (2.0 * sqrt_T) + r_ * K_ * discount * n_md2) / 365.0;
                res.greeks.rho   = (-0.01 * K_ * T_ * discount * n_md2);
            }
            return res;
        }

    private:
        double S_, K_, r_, sigma_, T_;
        double d1_, d2_;
    };

    class MonteCarloEngine {
    public:
        MonteCarloEngine(double spot, double strike, double rate, double vol, double maturity, uint64_t sims)
            : S_(spot), K_(strike), r_(rate), sigma_(vol), T_(maturity), sims_(sims) {}

        std::pair<double, double> price_european(OptionType type) {
            // Seed 42 guarantees identical deterministic outputs on every run
            std::mt19937_64 rng(42);
            std::normal_distribution<double> dist(0.0, 1.0);

            const double drift = (r_ - 0.5 * sigma_ * sigma_) * T_;
            const double vol_sqrt_T = sigma_ * std::sqrt(T_);
            const double discount = std::exp(-r_ * T_);

            double sum = 0.0;
            double sq_sum = 0.0;

            for (uint64_t i = 0; i < sims_; ++i) {
                const double z = dist(rng);
                const double ST = S_ * std::exp(drift + vol_sqrt_T * z);
                const double payoff = (type == OptionType::Call) 
                                      ? std::max(ST - K_, 0.0) 
                                      : std::max(K_ - ST, 0.0);

                sum += payoff;
                sq_sum += payoff * payoff;
            }

            const double mean = sum / static_cast<double>(sims_);
            const double price = discount * mean;
            const double variance = (sq_sum / static_cast<double>(sims_)) - (mean * mean);
            const double standard_error = discount * std::sqrt(variance / static_cast<double>(sims_));

            return {price, standard_error};
        }

    private:
        double S_, K_, r_, sigma_, T_;
        uint64_t sims_;
    };
}

int main() {
    // Benchmark Inputs
    const double S = 100.0;       // Spot price
    const double K = 100.0;       // Strike price
    const double r = 0.05;        // 5% Risk-free rate
    const double vol = 0.20;      // 20% Annual volatility
    const double T = 1.0;         // 1 Year maturity
    const uint64_t sims = 2000000;

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "====================================================\n";
    std::cout << "      QUANT MODELING VERIFICATION RUN (C++17)       \n";
    std::cout << "====================================================\n";

    // 1. Black-Scholes-Merton Closed-Form
    Quant::BlackScholesEngine bsm(S, K, r, vol, T);
    auto call_bsm = bsm.price(Quant::OptionType::Call);
    auto put_bsm  = bsm.price(Quant::OptionType::Put);

    std::cout << "--- Analytical Black-Scholes ---\n";
    std::cout << "Call Price: " << call_bsm.price << "\n";
    std::cout << "  Delta: " << call_bsm.greeks.delta 
              << " | Gamma: " << call_bsm.greeks.gamma 
              << " | Vega: "  << call_bsm.greeks.vega 
              << " | Theta: " << call_bsm.greeks.theta 
              << " | Rho: "   << call_bsm.greeks.rho << "\n\n";

    std::cout << "Put Price:  " << put_bsm.price << "\n";
    std::cout << "  Delta: " << put_bsm.greeks.delta 
              << " | Gamma: " << put_bsm.greeks.gamma 
              << " | Vega: "  << put_bsm.greeks.vega 
              << " | Theta: " << put_bsm.greeks.theta 
              << " | Rho: "   << put_bsm.greeks.rho << "\n\n";

    // 2. Monte Carlo Simulation (2,000,000 Paths)
    Quant::MonteCarloEngine mc(S, K, r, vol, T, sims);
    auto [call_mc, call_err] = mc.price_european(Quant::OptionType::Call);
    auto [put_mc, put_err]   = mc.price_european(Quant::OptionType::Put);

    std::cout << "--- Monte Carlo (2,000,000 Paths) ---\n";
    std::cout << "Call MC Price: " << call_mc << " (SE: +/- " << call_err << ")\n";
    std::cout << "Put MC Price:  " << put_mc  << " (SE: +/- " << put_err  << ")\n\n";

    std::cout << "--- Validation Delta ---\n";
    std::cout << "|BSM Call - MC Call|: " << std::abs(call_bsm.price - call_mc) << "\n";
    std::cout << "|BSM Put  - MC Put |: " << std::abs(put_bsm.price - put_mc) << "\n";
    std::cout << "====================================================\n";

    return 0;
}