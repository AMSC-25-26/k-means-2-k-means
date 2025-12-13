#include "silhouette.h"
#include <cmath>

namespace silhouette
{
    // Calculate Euclidean distance between two points
    double euclidean_distance(const std::vector<double>& p1, const std::vector<double>& p2)
    {
        double sum = 0.0;
        for (size_t i = 0; i < p1.size(); ++i)
        {
            const double diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
}
