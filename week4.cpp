#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>

namespace Quant {

inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

inline double norm_pdf(double x) {
    static const double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

enum class OptionType { Call, Put };

struct Greeks {
    double delta{0.0};
    double gamma{0.0};
};

class BSMAnalytics {
public:
    static double calculate_d1(double S, double K, double r, double sigma, double T) {
        if (T < 1e-6 || sigma < 1e-6) return 0.0;
        return (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    }

    static double price(OptionType type, double S, double K, double r, double sigma, double T) {
        if (T <= 1e-6) {
            return (type == OptionType::Call) ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
        }
        double d1 = calculate_d1(S, K, r, sigma, T);
        double d2 = d1 - sigma * std::sqrt(T);
        double discount = std::exp(-r * T);

        if (type == OptionType::Call) {
            return S * norm_cdf(d1) - K * discount * norm_cdf(d2);
        } else {
            return K * discount * norm_cdf(-d2) - S * norm_cdf(-d1);
        }
    }

    static Greeks compute_greeks(OptionType type, double S, double K, double r, double sigma, double T) {
        Greeks g;
        if (T <= 1e-6) return g;
        double sqrt_T = std::sqrt(T);
        double d1 = calculate_d1(S, K, r, sigma, T);
        double pdf_d1 = norm_pdf(d1);

        g.gamma = pdf_d1 / (S * sigma * sqrt_T);
        g.delta = (type == OptionType::Call) ? norm_cdf(d1) : (norm_cdf(d1) - 1.0);
        return g;
    }
};

struct OptimizationConfig {
    double gamma_tol;
    double delta_tol;
    int t_min_days;
};

struct OptimizationResult {
    OptimizationConfig config;
    double mean_cost;
    double mean_pnl;
    double pnl_std_dev;
    double max_drawdown;
};

class ParameterOptimizer {
public:
    ParameterOptimizer(double S0, double r, double sigma, double mu, int days)
        : S0_(S0), r_(r), sigma_(sigma), mu_(mu), days_(days), dt_(1.0 / 252.0) {}

    std::vector<std::vector<double>> generate_monte_carlo_paths(int n_paths, uint64_t seed = 42) {
        std::vector<std::vector<double>> paths(n_paths, std::vector<double>(days_ + 1));
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);

        double drift = (mu_ - 0.5 * sigma_ * sigma_) * dt_;
        double vol = sigma_ * std::sqrt(dt_);

        for (int p = 0; p < n_paths; ++p) {
            paths[p][0] = S0_;
            for (int d = 0; d < days_; ++d) {
                paths[p][d + 1] = paths[p][d] * std::exp(drift + vol * dist(rng));
            }
        }
        return paths;
    }

    OptimizationResult evaluate_config(const OptimizationConfig& cfg, const std::vector<std::vector<double>>& paths) {
        const double N1 = -100.0;
        const double K1 = 100.0, T1_init = 1.0;
        const double K2 = 110.0, T2_init = 0.5;
        const double stock_cost_rate = 0.0001; // 1 bps
        const double opt_cost_rate = 0.0050;   // 50 bps

        std::vector<double> path_pnls;
        std::vector<double> path_costs;
        std::vector<double> path_mdds;

        for (const auto& path : paths) {
            double cash = 50000.0;
            double opt2_qty = 0.0;
            double stock_qty = 0.0;
            double total_cost = 0.0;
            double peak_equity = 50000.0;
            double max_dd = 0.0;

            for (int step = 0; step <= days_; ++step) {
                double current_time = step * dt_;
                double S = path[step];
                double tau1 = std::max(T1_init - current_time, 1e-5);
                double tau2 = std::max(T2_init - current_time, 1e-5);

                double v1 = BSMAnalytics::price(OptionType::Call, S, K1, r_, sigma_, tau1);
                double v2 = BSMAnalytics::price(OptionType::Call, S, K2, r_, sigma_, tau2);
                Greeks g1 = BSMAnalytics::compute_greeks(OptionType::Call, S, K1, r_, sigma_, tau1);
                Greeks g2 = BSMAnalytics::compute_greeks(OptionType::Call, S, K2, r_, sigma_, tau2);

                int days_to_t2 = static_cast<int>(std::round(tau2 * 252.0));

                if (days_to_t2 >= cfg.t_min_days) {
                    double current_gamma = N1 * g1.gamma + opt2_qty * g2.gamma;
                    if (std::abs(current_gamma) > cfg.gamma_tol) {
                        double target_opt2 = -(N1 * g1.gamma) / g2.gamma;
                        double d_opt2 = target_opt2 - opt2_qty;
                        double fee = std::abs(d_opt2) * v2 * opt_cost_rate;
                        cash -= (d_opt2 * v2 + fee);
                        total_cost += fee;
                        opt2_qty = target_opt2;
                    }
                } else if (opt2_qty != 0.0) {
                    // Liquidate near expiry cutoff to prevent pin risk
                    double fee = std::abs(opt2_qty) * v2 * opt_cost_rate;
                    cash += (opt2_qty * v2 - fee);
                    total_cost += fee;
                    opt2_qty = 0.0;
                }

                double current_delta = N1 * g1.delta + opt2_qty * g2.delta + stock_qty;
                if (std::abs(current_delta) > cfg.delta_tol) {
                    double target_stock = -(N1 * g1.delta + opt2_qty * g2.delta);
                    double d_stock = target_stock - stock_qty;
                    double fee = std::abs(d_stock) * S * stock_cost_rate;
                    cash -= (d_stock * S + fee);
                    total_cost += fee;
                    stock_qty = target_stock;
                }

                double equity = cash + N1 * v1 + opt2_qty * v2 + stock_qty * S;
                peak_equity = std::max(peak_equity, equity);
                max_dd = std::max(max_dd, peak_equity - equity);
            }

            double final_S = path.back();
            double final_v1 = BSMAnalytics::price(OptionType::Call, final_S, K1, r_, sigma_, T1_init - days_ * dt_);
            double final_v2 = BSMAnalytics::price(OptionType::Call, final_S, K2, r_, sigma_, T2_init - days_ * dt_);
            double final_equity = cash + N1 * final_v1 + opt2_qty * final_v2 + stock_qty * final_S;

            path_pnls.push_back(final_equity - 50000.0);
            path_costs.push_back(total_cost);
            path_mdds.push_back(max_dd);
        }

        double mean_cost = std::accumulate(path_costs.begin(), path_costs.end(), 0.0) / paths.size();
        double mean_pnl = std::accumulate(path_pnls.begin(), path_pnls.end(), 0.0) / paths.size();
        double mean_mdd = std::accumulate(path_mdds.begin(), path_mdds.end(), 0.0) / paths.size();

        double sq_sum = 0.0;
        for (double pnl : path_pnls) {
            sq_sum += (pnl - mean_pnl) * (pnl - mean_pnl);
        }
        double std_pnl = std::sqrt(sq_sum / (paths.size() - 1));

        return {cfg, mean_cost, mean_pnl, std_pnl, mean_mdd};
    }

private:
    double S0_, r_, sigma_, mu_, dt_;
    int days_;
};

} // namespace Quant

int main() {
    Quant::ParameterOptimizer optimizer(100.0, 0.05, 0.20, 0.08, 100);
    auto paths = optimizer.generate_monte_carlo_paths(300);

    std::vector<Quant::OptimizationConfig> configs = {
        {0.00, 0.00, 2},
        {0.10, 0.50, 5},
        {0.25, 1.00, 5},
        {0.50, 2.00, 10}
    };

    std::cout << "\n" << std::string(82, '=') << "\n";
    std::cout << "             WEEK 4 PARAMETER OPTIMIZATION & GRID SEARCH RESULTS\n";
    std::cout << std::string(82, '=') << "\n";
    std::cout << std::left << std::setw(12) << "Gamma Tol"
              << std::setw(12) << "Delta Tol"
              << std::setw(10) << "T_min"
              << std::setw(12) << "Costs($)"
              << std::setw(14) << "Mean PnL($)"
              << std::setw(12) << "Std PnL"
              << std::setw(10) << "MDD($)" << "\n";
    std::cout << std::string(82, '-') << "\n";

    for (const auto& cfg : configs) {
        auto res = optimizer.evaluate_config(cfg, paths);
        std::cout << std::fixed << std::setprecision(2)
                  << std::left << std::setw(12) << res.config.gamma_tol
                  << std::setw(12) << res.config.delta_tol
                  << std::setw(10) << res.config.t_min_days
                  << std::setw(12) << res.mean_cost
                  << std::setw(14) << res.mean_pnl
                  << std::setw(12) << res.pnl_std_dev
                  << std::setw(10) << res.max_drawdown << "\n";
    }
    std::cout << std::string(82, '=') << "\n";

    return 0;
}
