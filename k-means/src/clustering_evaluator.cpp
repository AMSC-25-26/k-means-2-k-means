#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <string>
#include <spdlog/spdlog.h>

using namespace std;

double entropy(const vector<double> &p) {
    double h = 0.0;
    for (const double pi: p) {
        if (pi > 0) h -= pi * log2(pi);
    }
    return h;
}

double expected_mi(const int n, const vector<int> &a_sizes, const vector<int> &b_sizes) {
    double emi = 0.0;
    for (const int ai: a_sizes) {
        for (const int bj: b_sizes) {
            const double pij = static_cast<double>(ai * bj) / (n * n);
            emi -= pij * log2(pij);
        }
    }
    return emi;
}

template<typename L>
static string label_to_string(const L &lab) {
    ostringstream oss;
    oss << lab;
    return oss.str();
}

// Build a mapping from predicted-label-string -> most frequent true-label-string
template<typename L1, typename L2>
static unordered_map<string, string> build_majority_mapping(const vector<L1> &true_labels,
                                                            const vector<L2> &pred_labels) {
    unordered_map<string, unordered_map<string, int> > counts; // pred_str -> (true_str -> count)
    int n = static_cast<int>(true_labels.size());
    for (int i = 0; i < n; ++i) {
        string t = label_to_string(true_labels[i]);
        string p = label_to_string(pred_labels[i]);
        counts[p][t] += 1;
    }

    unordered_map<string, string> mapping;
    for (auto &kv: counts) {
        const string &pred_str = kv.first;
        auto &inner = kv.second;
        int best_count = -1;
        string best_true;
        for (auto &tkv: inner) {
            if (tkv.second > best_count) {
                best_count = tkv.second;
                best_true = tkv.first;
            }
        }
        if (best_count >= 0) mapping.emplace(pred_str, best_true);
    }

    spdlog::debug("Mapping from predicted labels to true labels (majority):");
    for (const auto &kv: mapping) {
        spdlog::debug("  Predicted: '{}' -> True: '{}'", kv.first, kv.second);
    }

    return mapping;
}

// Build a unique one-to-one mapping (greedy) from predicted-label-string -> true-label-string.
// Ensures each true label is assigned at most once and each predicted label at most once.
template<typename L1, typename L2>
static unordered_map<string, string> build_unique_mapping(const vector<L1> &true_labels,
                                                          const vector<L2> &pred_labels) {
    unordered_map<string, unordered_map<string, int> > counts; // pred -> (true -> count)
    int n = static_cast<int>(true_labels.size());
    for (int i = 0; i < n; ++i) {
        string t = label_to_string(true_labels[i]);
        string p = label_to_string(pred_labels[i]);
        counts[p][t] += 1;
    }

    // Flatten to vector of (count, pred, true)
    struct Triple {
        int count;
        string pred;
        string truth;
    };
    vector<Triple> triples;
    triples.reserve(n);
    for (auto &pkv: counts) {
        for (auto &tkv: pkv.second) {
            triples.push_back({tkv.second, pkv.first, tkv.first});
        }
    }

    // Sort desc by count
    sort(triples.begin(), triples.end(), [](const Triple &a, const Triple &b) {
        return a.count > b.count;
    });

    unordered_map<string, string> mapping;
    unordered_set<string> used_preds, used_trues;
    for (const auto &tr: triples) {
        if (used_preds.find(tr.pred) != used_preds.end()) continue;
        if (used_trues.find(tr.truth) != used_trues.end()) continue;
        // assign
        mapping.emplace(tr.pred, tr.truth);
        used_preds.insert(tr.pred);
        used_trues.insert(tr.truth);
    }

    spdlog::debug("Mapping from predicted labels to true labels (unique greedy):");
    for (const auto &kv: mapping) {
        spdlog::debug("  Predicted: '{}' -> True: '{}'", kv.first, kv.second);
    }

    return mapping;
}

// Now accept two label types: true labels (L1) and predicted labels (L2).
// Optional: provide pred_to_true_map (pred-string -> true-string) or set use_majority_mapping/use_unique_mapping to auto-generate it.
// If both use_unique_mapping and use_majority_mapping are false, no remapping is applied.
// If both true, unique mapping takes precedence.
template<typename L1, typename L2>
double adjusted_mutual_info(const vector<L1> &true_labels,
                            const vector<L2> &pred_labels,
                            const unordered_map<string, string> *pred_to_true_map = nullptr,
                            bool use_majority_mapping = true,
                            bool use_unique_mapping = true) {
    if (true_labels.size() != pred_labels.size()) {
        return 0.0;
    }
    int n = static_cast<int>(true_labels.size());
    if (n == 0) {
        return 0.0;
    }

    unordered_map<string, string> local_map;
    const unordered_map<string, string> *map_ptr = pred_to_true_map;

    if (use_unique_mapping) {
        local_map = build_unique_mapping(true_labels, pred_labels);
        map_ptr = &local_map;
    } else if (use_majority_mapping) {
        local_map = build_majority_mapping(true_labels, pred_labels);
        map_ptr = &local_map;
    }

    // Map distinct labels to consecutive indices for true and pred (after optional remapping) separately
    unordered_map<string, int> true_map, pred_map;
    vector<int> true_idx(n), pred_idx(n);
    int tcount = 0, pcount = 0;

    for (int i = 0; i < n; ++i) {
        string ts = label_to_string(true_labels[i]);
        auto it = true_map.find(ts);
        if (it == true_map.end()) {
            true_map.emplace(ts, tcount);
            true_idx[i] = tcount++;
        } else {
            true_idx[i] = it->second;
        }

        // convert predicted label to string, then possibly remap to the true-label namespace
        string ps0 = label_to_string(pred_labels[i]);
        string ps = ps0;
        if (map_ptr) {
            auto mit = map_ptr->find(ps0);
            if (mit != map_ptr->end()) ps = mit->second;
        }

        auto it2 = pred_map.find(ps);
        if (it2 == pred_map.end()) {
            pred_map.emplace(ps, pcount);
            pred_idx[i] = pcount++;
        } else {
            pred_idx[i] = it2->second;
        }
    }

    // sizes
    vector<int> true_sizes(tcount, 0), pred_sizes(pcount, 0);
    for (int i = 0; i < n; ++i) {
        ++true_sizes[true_idx[i]];
        ++pred_sizes[pred_idx[i]];
    }

    // contingency matrix
    vector contingency(tcount, vector(pcount, 0));
    for (int i = 0; i < n; ++i) {
        ++contingency[true_idx[i]][pred_idx[i]];
    }

    // marginal probabilities
    vector<double> pa(tcount), pb(pcount);
    for (int i = 0; i < tcount; ++i) pa[i] = static_cast<double>(true_sizes[i]) / n;
    for (int j = 0; j < pcount; ++j) pb[j] = static_cast<double>(pred_sizes[j]) / n;

    // Mutual Information
    double mi = 0.0;
    for (int i = 0; i < tcount; ++i) {
        for (int j = 0; j < pcount; ++j) {
            int nij = contingency[i][j];
            if (nij <= 0) continue;
            double p_ab = static_cast<double>(nij) / n;
            double denom = pa[i] * pb[j];
            if (denom > 0) {
                mi += p_ab * log2(p_ab / denom);
            }
        }
    }

    double h_a = entropy(pa);
    double h_b = entropy(pb);
    double nmi = (max(h_a, h_b) > 0) ? mi / max(h_a, h_b) : 0.0;

    double emi = expected_mi(n, true_sizes, pred_sizes);
    double denominator = max(emi, 1e-10);
    double ami = max(0.0, (mi - emi) / denominator);

    return ami;
}
