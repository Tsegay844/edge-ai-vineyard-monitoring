# ============================================================================
# NEW CELL TO ADD TO NOTEBOOK AFTER STEP 7
# Copy this entire cell into your MobileNetV2_Quantization_Official.ipynb
# Insert it as a new cell after the espdl_quantize_onnx() step
# ============================================================================

"""
## Step 8: Export INT8 ONNX for Evaluation

The .espdl format is ESP32-specific and cannot be evaluated with standard tools.
Export the quantized graph to INT8 ONNX format for accuracy evaluation with ONNXRuntime.
"""

from ppq import export_ppq_graph, TargetPlatform
import os

# Output path for INT8 ONNX
int8_onnx_output = output_dir / "quantized" / "mobilenetv2_128_int8.onnx"

print(f"\n{'='*70}")
print(f"📦 EXPORTING INT8 ONNX FOR EVALUATION")
print(f"{'='*70}")

# Check if we have the quantized graph from previous step
if 'quant_ppq_graph' not in locals():
    print("❌ Error: quant_ppq_graph not found!")
    print("   Please run Step 7 (espdl_quantize_onnx) first")
else:
    print(f"\n🔄 Exporting quantized model to ONNX format...")
    print(f"   Platform: ONNXRUNTIME (for evaluation)")
    print(f"   Output: {int8_onnx_output}")
    
    # Export the quantized graph to ONNX format
    # This creates an INT8 ONNX that can be evaluated with ONNXRuntime
    export_ppq_graph(
        graph=quant_ppq_graph,  # Quantized graph from Step 7
        platform=TargetPlatform.ONNXRUNTIME,  # Export for ONNX Runtime
        graph_save_to=str(int8_onnx_output)
    )
    
    # Verify the file was created
    if int8_onnx_output.exists():
        size_mb = os.path.getsize(int8_onnx_output) / (1024 * 1024)
        print(f"\n✅ INT8 ONNX exported successfully!")
        print(f"   📁 Path: {int8_onnx_output}")
        print(f"   📊 Size: {size_mb:.2f} MB")
        print(f"\n💡 This model can now be evaluated with:")
        print(f"   python evaluate_int8_model.py \\")
        print(f"       --int8_model {int8_onnx_output} \\")
        print(f"       --fp32_model {fp32_onnx_path}")
    else:
        print(f"\n❌ Export failed - file not created")

print(f"\n{'='*70}")
print(f"✅ STEP 8 COMPLETE")
print(f"{'='*70}\n")
