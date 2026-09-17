#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>

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
        if (T <= 1e-6 || sigma <= 1e-6) return 0.0;
        return (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    }

    static double price(OptionType type, double S, double K, double r, double sigma, double T) {
        if (T <= 1e-6) {
            return type == OptionType::Call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
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

struct PositionState {
    double timestamp;
    double spot;
    double shares;
    double hedge_opt_qty;
    double port_delta;
    double port_gamma;
    double total_equity;
};

class DeltaGammaHedger {
public:
    DeltaGammaHedger(double target_qty, double K1, double T1, 
                     double K2, double T2, double r, double sigma,
                     double stock_cost = 0.01, double opt_cost = 0.05)
        : n1_(target_qty), k1_(K1), t1_initial_(T1),
          k2_(K2), t2_initial_(T2), r_(r), sigma_(sigma),
          stock_cost_rate_(stock_cost), opt_cost_rate_(opt_cost),
          n2_(0.0), n_stock_(0.0), cash_(50000.0) {}

    void run_rebalance(double spot, double current_time) {
        double tau1 = std::max(t1_initial_ - current_time, 0.0);
        double tau2 = std::max(t2_initial_ - current_time, 0.0);

        if (tau1 <= 1e-4 || tau2 <= 1e-4) return; // Prevent division by zero near expiry

        // Step 0: Compute Greeks for both options
        Greeks g1 = BSMAnalytics::compute_greeks(OptionType::Call, spot, k1_, r_, sigma_, tau1);
        Greeks g2 = BSMAnalytics::compute_greeks(OptionType::Call, spot, k2_, r_, sigma_, tau2);

        // Step 1: Sequential Gamma Neutralization -> N2 = - (N1 * Gamma1) / Gamma2
        double target_n2 = - (n1_ * g1.gamma) / g2.gamma;
        double delta_n2 = target_n2 - n2_;
        double opt2_price = BSMAnalytics::price(OptionType::Call, spot, k2_, r_, sigma_, tau2);
        
        cash_ -= (delta_n2 * opt2_price + std::abs(delta_n2) * opt_cost_rate_);
        n2_ = target_n2;

        // Step 2: Sequential Delta Neutralization -> N_stock = - (N1 * Delta1 + N2 * Delta2)
        double total_option_delta = (n1_ * g1.delta) + (n2_ * g2.delta);
        double target_stock = - total_option_delta;
        double delta_stock = target_stock - n_stock_;

        cash_ -= (delta_stock * spot + std::abs(delta_stock) * stock_cost_rate_);
        n_stock_ = target_stock;

        // Verify neutralization state
        double net_delta = (n1_ * g1.delta) + (n2_ * g2.delta) + n_stock_;
        double net_gamma = (n1_ * g1.gamma) + (n2_ * g2.gamma);
        double equity = get_equity(spot, current_time);

        history_.push_back({current_time, spot, n_stock_, n2_, net_delta, net_gamma, equity});
    }

    double get_equity(double spot, double current_time) const {
        double tau1 = std::max(t1_initial_ - current_time, 0.0);
        double tau2 = std::max(t2_initial_ - current_time, 0.0);

        double v1 = BSMAnalytics::price(OptionType::Call, spot, k1_, r_, sigma_, tau1);
        double v2 = BSMAnalytics::price(OptionType::Call, spot, k2_, r_, sigma_, tau2);

        return cash_ + (n1_ * v1) + (n2_ * v2) + (n_stock_ * spot);
    }

    const std::vector<PositionState>& get_history() const { return history_; }

private:
    double n1_, k1_, t1_initial_;
    double k2_, t2_initial_;
    double r_, sigma_;
    double stock_cost_rate_, opt_cost_rate_;
    double n2_, n_stock_, cash_;
    std::vector<PositionState> history_;
};

} // namespace Quant

int main() {
    // Strategy Initialization
    // Short 100 ATM Calls (K=100, T=1.0)
    // Hedge with traded OTM Calls (K=110, T=0.5) and Stock
    Quant::DeltaGammaHedger hedger(-100.0, 100.0, 1.0, 110.0, 0.5, 0.05, 0.20);

    // Initial rebalance at t = 0, S = $100
    hedger.run_rebalance(100.0, 0.0);

    // Simulated market steps over 5 days with spot volatility
    std::vector<std::pair<double, double>> market_ticks = {
        {1.0 / 365.0, 102.50},
        {2.0 / 365.0, 105.00},
        {3.0 / 365.0, 103.20},
        {4.0 / 365.0, 99.50},
        {5.0 / 365.0, 101.00}
    };

    for (const auto& tick : market_ticks) {
        hedger.run_rebalance(tick.second, tick.first);
    }

    std::cout << "========================================================================================\n";
    std::cout << "               WEEK 2: SEQUENTIAL DELTA-GAMMA NEUTRAL EXECUTION LOG                     \n";
    std::cout << "========================================================================================\n";
    std::cout << std::left << std::setw(8)  << "Day"
              << std::setw(10) << "Spot ($)"
              << std::setw(14) << "Opt2 (Qty)"
              << std::setw(14) << "Stock (Qty)"
              << std::setw(14) << "Net Delta"
              << std::setw(14) << "Net Gamma"
              << std::setw(14) << "Equity ($)" << "\n";
    std::cout << "----------------------------------------------------------------------------------------\n";

    for (const auto& log : hedger.get_history()) {
        std::cout << std::fixed << std::setprecision(4)
                  << std::left << std::setw(8)  << log.timestamp * 365.0
                  << std::setw(10) << log.spot
                  << std::setw(14) << log.hedge_opt_qty
                  << std::setw(14) << log.shares
                  << std::setw(14) << log.port_delta
                  << std::setw(14) << log.port_gamma
                  << std::setprecision(2)
                  << std::setw(14) << log.total_equity << "\n";
    }
    std::cout << "========================================================================================\n";

    return 0;
}
