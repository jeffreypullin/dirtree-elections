/******************************************************************************
 * File:             distributions.cpp
 *
 * Author:           Floyd Everest <me@floydeverest.com>
 * Created:          02/27/22
 * Description:      This file implements the required distributions as
 *                   outlined in `distributions.hpp`.
 *****************************************************************************/

#include "distributions.hpp"

#include <algorithm>
#include <vector>

// Helper type for the approximated Dirichlet-Multinomial sample.
struct rem_idx {
  float remainder = 0;
  int idx = 0;
};

bool cmp_pairs(const rem_idx &a, const rem_idx &b) {
  return a.remainder > b.remainder;
}

int *rDirichletMultinomial(int count, float *alpha, int d,
                           bool approximate_dmnom, std::mt19937 *engine) {
  int *out = new int[d];

  float gam, sum_ps, p;
  float *gamma = new float[d];
  float gamma_sum = 0.;

  // Sample the gamma variates for category i.
  for (auto i = 0; i < d; ++i) {
    std::gamma_distribution<float> g(alpha[i], 1.0);
    gam = g(*engine);
    gamma[i] = gam;
    gamma_sum += gam;
  }

  if (approximate_dmnom) {
    // Use the Dirichlet sample to apprixmate the draw by taking the
    // multinomial mean, and then rounding the remainders appropriately.

    // For obtaining the mean multinomial draw, we multiply the Dirichlet
    // probabilities by count, and then floor each value.
    // This results in the sum being less than or equal to count, so we
    // redistribute the difference equally among the probability values
    // which had the highest remainders before 'floor'-ing.
    int count_remaining = count;
    rem_idx temp_pair;
    float unrounded;
    float floor;
    std::vector<rem_idx> sorted_pairs{};

    for (auto i = 0; i < d; ++i) {
      // Temp is the unrounded 'count'
      unrounded = (gamma[i] / gamma_sum) * count;

      // Floor is the count rounded down
      floor = std::floor(unrounded);
      temp_pair = rem_idx{unrounded - floor, i};

      // Update the count with the rounded-down int.
      out[i] = static_cast<int>(floor);
      count_remaining -= out[i];

      // Insert the floor, remainder pair to the sorted vector.
      sorted_pairs.insert(std::upper_bound(sorted_pairs.begin(),
                                           sorted_pairs.end(), temp_pair,
                                           cmp_pairs),
                          std::move(temp_pair));
    }

    // Correct the rounded-down 'counts'.
    for (auto i = 0; i < d; ++i) {
      if (count_remaining == 0)
        break;
      out[sorted_pairs[i].idx] += 1;
      count_remaining -= 1;
    }

  } else {
    // Sample from Multinomial distribution with pi=gamma/gamma_sum.
    sum_ps = 1.0;
    for (auto i = 0; i < d - 1; ++i) {
      // Calculate marginal probability p.
      if (gamma_sum == 0) {
        p = 1.;
      } else {
        p = gamma[i] / gamma_sum;
      }
      // Draw from marginal binomial distribution.
      std::binomial_distribution<int> b(count, p / sum_ps);
      out[i] = b(*engine);
      count -= out[i];
      // Renormalise ps for next categories.
      sum_ps -= p;
    }
    // Remainder goes to last category.
    out[d - 1] = count;
  }

  delete[] gamma;

  return out;
}

float rBeta(float a, float b, std::mt19937 *engine) {
  std::gamma_distribution<float> gx(a);
  std::gamma_distribution<float> gy(b);
  // Avoid zero denominator by adding epsilon.
  float x = gx(*engine) + std::numeric_limits<float>::epsilon();
  float y = gy(*engine) + std::numeric_limits<float>::epsilon();
  return x / (x + y);
}
