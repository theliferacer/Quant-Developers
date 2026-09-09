#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <random>
#include <vector>
using namespace std;
using namespace std::numbers;

double norm_cdf(double x) {
	return 0.5 * erfc(-x/sqrt(2.0));
}
double norm_pdf(double x) {
	return exp(-x * x * 0.5) / sqrt(2 * pi);
}
enum OptionType { CALL, PUT };

struct greeks {
	double delta;
	double gamma;
	double vega;
	double theta;
	double rho;
};

struct OptionsResults {
	double price;
	greeks greeks;
};

class black_scholes {
	double S, K, r, T, sigma, sqrtT;
	double d1 = 0.0;
	double d2 = 0.0;
public:
	black_scholes(double spot, double strike, double rate, double expiry, double sigma)
    : S(spot), K(strike), r(rate), T(expiry), sigma(sigma), d1(0.0), d2(0.0)
{
    if (T > 0.0 && sigma > 0.0 && S > 0.0 && K > 0.0) {
        sqrtT = sqrt(T);
        d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        d2 = d1 - sigma * sqrtT;
    }
}

	OptionsResults price_greeks(OptionType option_type) {
		double n_d1 = norm_cdf(d1);
		double n_d2 = norm_cdf(d2);
		double nm_d1 = norm_cdf(-d1);
		double nm_d2 = norm_cdf(-d2);
		double pdf_d1 = norm_pdf(d1);
		double discount_factor = exp(-r * T);
		
		OptionsResults res;

		if (option_type == OptionType::CALL) {
			res.price = S*n_d1 - K*discount_factor * n_d2;
			res.greeks.delta = n_d1;
			res.greeks.theta = ((-S * pdf_d1 * sigma) / (2 * sqrtT) - r * K * discount_factor * n_d2) / 365.0;
			res.greeks.rho = K * T * discount_factor * n_d2 * 0.01;
		}
		else {
			res.price = K * discount_factor * nm_d2 - S * nm_d1;
			res.greeks.delta = -nm_d1;
			res.greeks.theta = ((-S * pdf_d1 * sigma) / (2 * sqrtT) + r * K * discount_factor * nm_d2) / 365.0;
			res.greeks.rho = -K * T * discount_factor * nm_d2 * 0.01;
		}
		res.greeks.gamma = pdf_d1 / (S * sigma * sqrtT);
		res.greeks.vega = S * pdf_d1 * sqrtT * 0.01;

		return res;
	}

};

struct mcResults {
	double price;
	double se;
};

class monte_carlo {
	double S0, K, r, T, sigma, sqrtT;
	uint64_t M;
public:
	monte_carlo(double spot, double strike, double rate, double expiry, double sigma, uint64_t nSims) : S0(spot), K(strike), r(rate), T(expiry), sigma(sigma), M(nSims) {
		sqrtT = sqrt(T);
	}

	mcResults price_mc(OptionType option_type) {
		double discount_factor = exp(-r * T);
		double payoffs_sum = 0.0;
		double payoffs_sq_sum = 0.0;

		mt19937_64 rng(42);
		normal_distribution<double> dist(0.0, 1.0);

		for (uint64_t i = 0; i < M; i++) {
			double z = dist(rng);
			double ST = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrtT * z);

			double payoff = 0.0;
			if (option_type == OptionType::CALL) {
				payoff = max(ST - K, 0.0);
			}
			else {
				payoff = max(K - ST, 0.0);
			}

			payoffs_sum += payoff;
			payoffs_sq_sum += payoff * payoff;

		}

		double payoff_mean = payoffs_sum / M;
		double var = payoffs_sq_sum / M - payoff_mean * payoff_mean;

		mcResults mcres;
		mcres.price = discount_factor * payoff_mean;
		mcres.se = discount_factor * sqrt(var / M);

		return mcres;
	}

};

int main() {
	const double S = 100.0, K = 100.0, r = 0.05, vol = 0.20, T = 1.0;
	const uint64_t M = 20000000;
	black_scholes bsm(S, K, r, T, vol);
	OptionsResults res_call;
	OptionsResults	res_put;
	res_call = bsm.price_greeks(CALL);
	res_put = bsm.price_greeks(PUT);

	monte_carlo mc(S, K, r, T, vol, M);
	mcResults mc_call;
	mcResults mc_put;
	mc_call = mc.price_mc(CALL);
	mc_put = mc.price_mc(PUT);


	return 0;
}
