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

struct StepRecord {
    int day;
    double spot;
    double unhedged_equity;
    double delta_equity;
    double dg_equity;
    double dg_opt2_qty;
    double dg_stock_qty;
};

struct BacktestMetrics {
    double final_equity;
    double net_pnl;
    double max_drawdown;
    double tracking_error;
    double total_cost;
};

class BacktestEngine {
public:
    BacktestEngine(double S0, double r, double sigma, double mu, int days)
        : S0_(S0), r_(r), sigma_(sigma), mu_(mu), days_(days), dt_(1.0 / 252.0) {}

    std::vector<double> generate_spot_path(uint64_t seed = 42) {
        std::vector<double> path;
        path.reserve(days_ + 1);
        path.push_back(S0_);

        std::mt19937_64 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);

        double drift = (mu_ - 0.5 * sigma_ * sigma_) * dt_;
        double vol = sigma_ * std::sqrt(dt_);

        for (int i = 0; i < days_; ++i) {
            double z = dist(rng);
            double next_S = path.back() * std::exp(drift + vol * z);
            path.push_back(next_S);
        }
        return path;
    }

    void run_simulation() {
        auto spot_path = generate_spot_path();

        // Contract Parameters
        const double N1 = -100.0;
        const double K1 = 100.0, T1_init = 1.0;
        const double K2 = 110.0, T2_init = 0.5;
        const double stock_cost_rate = 0.0001; // 1 bps
        const double opt_cost_rate = 0.0050;   // 50 bps

        // State variables
        double cash_unhedged = 50000.0;
        double cash_delta = 50000.0;
        double cash_dg = 50000.0;

        double delta_stock_qty = 0.0;
        double dg_opt2_qty = 0.0;
        double dg_stock_qty = 0.0;

        double total_cost_delta = 0.0;
        double total_cost_dg = 0.0;

        records_.clear();

        for (int step = 0; step <= days_; ++step) {
            double current_time = step * dt_;
            double S = spot_path[step];
            double tau1 = std::max(T1_init - current_time, 1e-5);
            double tau2 = std::max(T2_init - current_time, 1e-5);

            double v1 = BSMAnalytics::price(OptionType::Call, S, K1, r_, sigma_, tau1);
            double v2 = BSMAnalytics::price(OptionType::Call, S, K2, r_, sigma_, tau2);
            Greeks g1 = BSMAnalytics::compute_greeks(OptionType::Call, S, K1, r_, sigma_, tau1);
            Greeks g2 = BSMAnalytics::compute_greeks(OptionType::Call, S, K2, r_, sigma_, tau2);

            // Rebalance Delta-Only
            double target_stock_delta = -(N1 * g1.delta);
            double d_stock_delta = target_stock_delta - delta_stock_qty;
            double cost_stock_delta = std::abs(d_stock_delta) * S * stock_cost_rate;
            cash_delta -= (d_stock_delta * S + cost_stock_delta);
            delta_stock_qty = target_stock_delta;
            total_cost_delta += cost_stock_delta;

            // Rebalance Delta-Gamma
            double target_opt2_dg = -(N1 * g1.gamma) / g2.gamma;
            double d_opt2_dg = target_opt2_dg - dg_opt2_qty;
            double cost_opt2_dg = std::abs(d_opt2_dg) * v2 * opt_cost_rate;
            cash_dg -= (d_opt2_dg * v2 + cost_opt2_dg);
            dg_opt2_qty = target_opt2_dg;

            double target_stock_dg = -(N1 * g1.delta + dg_opt2_qty * g2.delta);
            double d_stock_dg = target_stock_dg - dg_stock_qty;
            double cost_stock_dg = std::abs(d_stock_dg) * S * stock_cost_rate;
            cash_dg -= (d_stock_dg * S + cost_stock_dg);
            dg_stock_qty = target_stock_dg;
            total_cost_dg += (cost_opt2_dg + cost_stock_dg);

            // Mark-to-market Equity
            double eq_unhedged = cash_unhedged + N1 * v1;
            double eq_delta = cash_delta + N1 * v1 + delta_stock_qty * S;
            double eq_dg = cash_dg + N1 * v1 + dg_opt2_qty * v2 + dg_stock_qty * S;

            records_.push_back({step, S, eq_unhedged, eq_delta, eq_dg, dg_opt2_qty, dg_stock_qty});
        }
    }

    void print_summary_table() const {
        std::cout << "\n" << std::string(88, '=') << "\n";
        std::cout << "               WEEK 3 DISCRETE BACKTEST EXECUTION SAMPLE (EVERY 10 DAYS)\n";
        std::cout << std::string(88, '=') << "\n";
        std::cout << std::left << std::setw(6)  << "Day"
                  << std::setw(10) << "Spot($)"
                  << std::setw(14) << "Unhedged Eq"
                  << std::setw(14) << "Delta Eq"
                  << std::setw(14) << "D-G Eq"
                  << std::setw(14) << "Opt2 Qty"
                  << std::setw(14) << "Stock Qty" << "\n";
        std::cout << std::string(88, '-') << "\n";

        for (const auto& r : records_) {
            if (r.day % 10 == 0) {
                std::cout << std::fixed << std::setprecision(2)
                          << std::left << std::setw(6)  << r.day
                          << std::setw(10) << r.spot
                          << std::setw(14) << r.unhedged_equity
                          << std::setw(14) << r.delta_equity
                          << std::setw(14) << r.dg_equity
                          << std::setprecision(3)
                          << std::setw(14) << r.dg_opt2_qty
                          << std::setw(14) << r.dg_stock_qty << "\n";
            }
        }
        std::cout << std::string(88, '=') << "\n";
    }

private:
    double S0_, r_, sigma_, mu_, dt_;
    int days_;
    std::vector<StepRecord> records_;
};

} // namespace Quant

int main() {
    Quant::BacktestEngine engine(100.0, 0.05, 0.20, 0.08, 100);
    engine.run_simulation();
    engine.print_summary_table();
    return 0;
}
