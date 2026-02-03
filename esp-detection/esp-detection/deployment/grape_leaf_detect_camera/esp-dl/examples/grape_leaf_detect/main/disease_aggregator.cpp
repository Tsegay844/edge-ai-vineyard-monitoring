#include "disease_aggregator.hpp"
#include "esp_log.h"
#include <algorithm>
#include <cmath>

namespace disease_aggregation {

static const char* TAG = "DiseaseAggregator";

// ========== Entropy Calculation ==========
float DiseaseAggregator::calculate_entropy(
    const std::vector<std::pair<int, float>>& class_probs,
    int num_classes)
{
    float entropy = 0.0f;
    
    for (const auto& [class_id, prob] : class_probs) {
        if (prob > 0.0f) {  // log(0) is undefined
            entropy -= prob * logf(prob);
        }
    }
    
    // Normalize to [0, 1] range
    // Maximum entropy = log(num_classes)
    float max_entropy = logf((float)num_classes);
    if (max_entropy > 0.0f) {
        entropy /= max_entropy;
    }
    
    return entropy;
}

// ========== Spatial Quality Calculation ==========

float DiseaseAggregator::calculate_size_score(
    float relative_size,
    float min_size,
    float max_size)
{
    // Too small → likely noise or distant leaf
    if (relative_size < min_size) {
        return 0.3f;  // Penalty but not zero
    }
    
    // Too large → likely misdetection or multiple leaves
    if (relative_size > max_size) {
        return 0.5f;  // Penalty
    }
    
    // Good size range → full score
    return 1.0f;
}

float DiseaseAggregator::calculate_aspect_score(
    float aspect_ratio,
    float ideal_aspect,
    float tolerance)
{
    // Grape leaves are roughly circular to oval
    // Ideal aspect ratio ~1.0 (square)
    // Tolerance allows 0.5 to 1.5 range
    
    float deviation = fabsf(aspect_ratio - ideal_aspect);
    
    if (deviation > tolerance * 2.0f) {
        // Very elongated or very wide → bad crop
        return 0.2f;
    }
    
    // Linear penalty based on deviation
    float score = 1.0f - (deviation / (tolerance * 2.0f));
    return fmaxf(score, 0.2f);  // Minimum 0.2
}

float DiseaseAggregator::calculate_centrality_score(
    float center_x,
    float center_y,
    float img_center_x,
    float img_center_y)
{
    // Calculate distance from bbox center to image center
    float dx = center_x - img_center_x;
    float dy = center_y - img_center_y;
    float distance = sqrtf(dx * dx + dy * dy);
    
    // Maximum possible distance (corner to center)
    float max_distance = sqrtf(img_center_x * img_center_x + img_center_y * img_center_y);
    
    // Normalize and invert (closer to center = higher score)
    if (max_distance > 0.0f) {
        float normalized_distance = distance / max_distance;
        return 1.0f - normalized_distance;
    }
    
    return 1.0f;
}

float DiseaseAggregator::calculate_bbox_quality(
    const BboxInfo& bbox,
    int img_width,
    int img_height,
    const AggregationConfig& config)
{
    // Bbox dimensions
    int width = bbox.x2 - bbox.x1;
    int height = bbox.y2 - bbox.y1;
    float bbox_area = (float)(width * height);
    float img_area = (float)(img_width * img_height);
    
    // 1. Size score
    float relative_size = bbox_area / img_area;
    float size_score = calculate_size_score(
        relative_size, 
        config.min_relative_size, 
        config.max_relative_size
    );
    
    // 2. Aspect ratio score
    float aspect_ratio = (height > 0) ? ((float)width / (float)height) : 1.0f;
    float aspect_score = calculate_aspect_score(
        aspect_ratio,
        config.ideal_aspect_ratio,
        config.aspect_tolerance
    );
    
    // 3. Centrality score
    float bbox_center_x = (bbox.x1 + bbox.x2) / 2.0f;
    float bbox_center_y = (bbox.y1 + bbox.y2) / 2.0f;
    float img_center_x = img_width / 2.0f;
    float img_center_y = img_height / 2.0f;
    float centrality_score = calculate_centrality_score(
        bbox_center_x, bbox_center_y,
        img_center_x, img_center_y
    );
    
    // Combined quality (geometric mean for balanced contribution)
    float quality = size_score * aspect_score * centrality_score;
    
    return quality;
}

// ========== Hybrid Weight Calculation ==========

float DiseaseAggregator::compute_hybrid_weight(
    float det_confidence,
    float entropy,
    float bbox_quality,
    const AggregationConfig& config)
{
    float weight = det_confidence;  // Start with detection confidence
    
    // Add uncertainty weighting
    if (config.use_entropy_weighting) {
        float certainty = 1.0f - entropy;  // High entropy → low weight
        weight *= certainty;
    }
    
    // Add spatial quality weighting
    if (config.use_spatial_weighting) {
        weight *= bbox_quality;
    }
    
    return weight;
}

// ========== Main Aggregation Function ==========

AggregationResult DiseaseAggregator::aggregate(
    const std::vector<DiseaseResult>& disease_results,
    const std::vector<BboxInfo>& bbox_info,
    int img_width,
    int img_height,
    const AggregationConfig& config)
{
    AggregationResult result;
    result.num_leaves_analyzed = disease_results.size();
    
    // Validate input
    if (disease_results.empty() || bbox_info.empty()) {
        ESP_LOGW(TAG, "No disease results to aggregate");
        return result;
    }
    
    if (disease_results.size() != bbox_info.size()) {
        ESP_LOGE(TAG, "Mismatch: %d disease results but %d bbox info",
                 disease_results.size(), bbox_info.size());
        return result;
    }
    
    // Determine number of classes
    int num_classes = 4;  // 0: Black_rot, 1: Esca, 2: Healthy, 3: Leaf_blight (ImageFolder alphabetical order)
    
    // Initialize weighted sums
    result.class_scores.resize(num_classes, 0.0f);
    result.weights.resize(disease_results.size(), 0.0f);
    result.entropies.resize(disease_results.size(), 0.0f);
    result.bbox_qualities.resize(disease_results.size(), 0.0f);
    
    // Calculate weights and weighted sums
    for (size_t i = 0; i < disease_results.size(); i++) {
        const auto& dr = disease_results[i];
        const auto& bbox = bbox_info[i];
        
        // Calculate entropy (uncertainty)
        float entropy = 0.0f;
        if (config.use_entropy_weighting) {
            entropy = calculate_entropy(dr.all_classes, num_classes);
            result.entropies[i] = entropy;
        }
        
        // Calculate bbox quality
        float bbox_quality = 1.0f;
        if (config.use_spatial_weighting) {
            bbox_quality = calculate_bbox_quality(bbox, img_width, img_height, config);
            result.bbox_qualities[i] = bbox_quality;
        }
        
        // Compute hybrid weight
        float weight = compute_hybrid_weight(
            bbox.det_confidence,
            entropy,
            bbox_quality,
            config
        );
        
        result.weights[i] = weight;
        result.total_weight += weight;
        
        // Add weighted contribution for each class
        for (const auto& [class_id, prob] : dr.all_classes) {
            if (class_id >= 0 && class_id < num_classes) {
                result.class_scores[class_id] += weight * prob;
            }
        }
    }
    
    // Guard against division by zero
    if (result.total_weight <= 0.0f) {
        ESP_LOGW(TAG, "Total weight is zero - cannot aggregate");
        return result;
    }
    
    // Normalize by total weight to get weighted average
    for (int c = 0; c < num_classes; c++) {
        result.class_scores[c] /= result.total_weight;
    }
    
    // Find class with highest weighted score
    result.final_class_id = 0;
    result.final_confidence = result.class_scores[0];
    for (int c = 1; c < num_classes; c++) {
        if (result.class_scores[c] > result.final_confidence) {
            result.final_confidence = result.class_scores[c];
            result.final_class_id = c;
        }
    }
    
    result.class_name = DiseaseClassifier::get_class_name(result.final_class_id);
    
    return result;
}

// ========== Print Results ==========

void DiseaseAggregator::print_results(const AggregationResult& result, const char* tag)
{
    //ESP_LOGI(tag, "════════════════════════════════════════════════════");
    ESP_LOGI(tag, "  WEIGHTED DISEASE AGGREGATION RESULTS              ");
    ESP_LOGI(tag, "════════════════════════════════════════════════════");
    
    // Individual leaf weights
    ESP_LOGI(tag, "\n Individual Leaf Weights:");
    for (size_t i = 0; i < result.weights.size(); i++) {
        ESP_LOGI(tag, "   Leaf[%d]: weight=%.4f", i, result.weights[i]);
        
        if (!result.entropies.empty() && result.entropies[i] > 0.0f) {
            ESP_LOGI(tag, "           entropy=%.3f (certainty=%.1f%%)", 
                     result.entropies[i], (1.0f - result.entropies[i]) * 100.0f);
        }
        
        if (!result.bbox_qualities.empty() && result.bbox_qualities[i] > 0.0f) {
            ESP_LOGI(tag, "           bbox_quality=%.3f", result.bbox_qualities[i]);
        }
    }
    
    // Weighted class scores
    ESP_LOGI(tag, "\n Weighted Class Scores (S_c):");
    for (size_t c = 0; c < result.class_scores.size(); c++) {
        const char* cls = DiseaseClassifier::get_class_name(c);
        ESP_LOGI(tag, "   • %s: %.2f%%", cls, result.class_scores[c] * 100.0f);
    }
    
    // Final diagnosis
    //ESP_LOGI(tag, "════════════════════════════════════════════════════");
    ESP_LOGI(tag, "  FINAL DIAGNOSIS                                   ");
    ESP_LOGI(tag, "════════════════════════════════════════════════════");
    ESP_LOGI(tag, "  Disease:     %-36s ", result.class_name);
    ESP_LOGI(tag, "  Confidence:  %.1f%%                                 ", 
             result.final_confidence * 100.0f);
    ESP_LOGI(tag, "  Leaves:      %d analyzed                           ", 
             result.num_leaves_analyzed);
    ESP_LOGI(tag, "  Total Weight: %.4f                                ", 
             result.total_weight);
    //ESP_LOGI(tag, "════════════════════════════════════════════════════\n");
}

} // namespace disease_aggregation
