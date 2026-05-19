#pragma once

#include "disease_classifier.hpp"
#include <vector>
#include <cmath>

/**
 * Disease Aggregator - weighted aggregation for multi-instance disease classification
 * 
 * Implements three aggregation strategies:
 * 1. Simple Detection-Confidence Weighting (baseline)
 * 2. Uncertainty-Aware Weighting (entropy-based)
 * 3. Hybrid Weighting (detection + uncertainty + spatial quality)
 * 
 * Based on Multi-Instance Learning (MIL) principles used in medical pathology
 * and precision agriculture systems.
 */

namespace disease_aggregation {

// Bounding box information for spatial quality calculation
struct BboxInfo {
    int x1, y1, x2, y2;  // Coordinates
    float det_confidence; // Detection confidence from YOLO
    
    BboxInfo(int x1, int y1, int x2, int y2, float conf) 
        : x1(x1), y1(y1), x2(x2), y2(y2), det_confidence(conf) {}
};

// Aggregation configuration
struct AggregationConfig {
    // Weighting strategy
    bool use_entropy_weighting;   // Weight by prediction certainty
    bool use_spatial_weighting;   // Weight by bbox quality (size, position, aspect)
    
    // Spatial quality parameters
    float min_relative_size;      // Minimum bbox_area / image_area (e.g., 0.005 = 0.5%)
    float max_relative_size;      // Maximum bbox_area / image_area (e.g., 0.5 = 50%)
    float ideal_aspect_ratio;     // Ideal aspect ratio for grape leaves (1.0 = square)
    float aspect_tolerance;       // Tolerance for aspect ratio (0.5 means 0.5 to 1.5 is good)
    
    // Default configuration: Simple detection-confidence weighting
    AggregationConfig() 
        : use_entropy_weighting(false),
          use_spatial_weighting(false),
          min_relative_size(0.005f),
          max_relative_size(0.5f),
          ideal_aspect_ratio(1.0f),
          aspect_tolerance(0.5f) {}
};

// Aggregation result
struct AggregationResult {
    int final_class_id;
    float final_confidence;
    const char* class_name;
    std::vector<float> class_scores;     // Weighted score for each class
    std::vector<float> weights;          // Computed weight for each leaf
    std::vector<float> entropies;        // Entropy for each leaf (if computed)
    std::vector<float> bbox_qualities;   // Bbox quality for each leaf (if computed)
    float total_weight;                  // Sum of all weights
    int num_leaves_analyzed;
    
    AggregationResult() 
        : final_class_id(-1), 
          final_confidence(0.0f), 
          class_name("unknown"),
          total_weight(0.0f),
          num_leaves_analyzed(0) {}
};

class DiseaseAggregator {
public:
    /**
     * Aggregate disease predictions from multiple leaves
     * 
     * @param disease_results Vector of DiseaseResult from each leaf
     * @param bbox_info Vector of bounding box information for each leaf
     * @param img_width Image width in pixels
     * @param img_height Image height in pixels
     * @param config Aggregation configuration
     * @return AggregationResult with final diagnosis and detailed breakdown
     */
    static AggregationResult aggregate(
        const std::vector<DiseaseResult>& disease_results,
        const std::vector<BboxInfo>& bbox_info,
        int img_width,
        int img_height,
        const AggregationConfig& config = AggregationConfig()
    );
    
    /**
     * Print detailed aggregation results
     */
    static void print_results(const AggregationResult& result, const char* tag);
    
private:
    /**
     * Calculate Shannon entropy of a probability distribution
     * Lower entropy = more confident prediction
     * 
     * H = -Σ p(c) * log(p(c))
     * Normalized to [0, 1] range
     */
    static float calculate_entropy(const std::vector<std::pair<int, float>>& class_probs, int num_classes);
    
    /**
     * Calculate bounding box quality score based on:
     * - Relative size (bbox_area / image_area)
     * - Aspect ratio (should be close to ideal for grape leaves)
     * - Centrality (distance from image center)
     * 
     * Returns score in [0, 1] range
     */
    static float calculate_bbox_quality(
        const BboxInfo& bbox,
        int img_width,
        int img_height,
        const AggregationConfig& config
    );
    
    /**
     * Calculate relative size score
     * Penalizes too-small (noise) and too-large (misdetection) bboxes
     */
    static float calculate_size_score(
        float relative_size,
        float min_size,
        float max_size
    );
    
    /**
     * Calculate aspect ratio score
     * Penalizes aspect ratios far from ideal (grape leaves are roughly circular)
     */
    static float calculate_aspect_score(
        float aspect_ratio,
        float ideal_aspect,
        float tolerance
    );
    
    /**
     * Calculate centrality score
     * Penalizes leaves at image edges (may be cut off or distorted)
     */
    static float calculate_centrality_score(
        float center_x,
        float center_y,
        float img_center_x,
        float img_center_y
    );
    
    /**
     * Compute final weight for a leaf
     * weight = det_conf * (1 - entropy) * bbox_quality
     */
    static float compute_hybrid_weight(
        float det_confidence,
        float entropy,
        float bbox_quality,
        const AggregationConfig& config
    );
};

} // namespace disease_aggregation
