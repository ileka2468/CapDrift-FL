"""
CLI Diagnostic Classifier for Chest Radiographs.
Uses the trained Federated MobileNetV3 model to predict NORMAL vs PNEUMONIA.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import FederatedClassifier


def main():
    parser = argparse.ArgumentParser(description="Classify Chest X-Ray using Federated Model")
    parser.add_argument("--image", type=str, required=True, help="Path to radiograph image (.jpeg, .png, .jpg)")
    parser.add_argument("--weights", type=str, default="results/best_model.pth", help="Model weights file")
    parser.add_argument("--save_cam", type=str, default=None, help="Optional output path to save Grad-CAM overlay")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        sys.exit(1)

    print("\n=======================================================")
    print("   FEDERATED MEDICAL DIAGNOSTIC CLASSIFIER (CLI)       ")
    print("=======================================================")
    print(f"Target Image:    {args.image}")
    print(f"Loading Model:   {args.weights}...")

    classifier = FederatedClassifier(weights_path=args.weights)
    result = classifier.predict(args.image, generate_gradcam=args.save_cam is not None)

    pred = result["prediction"]
    conf = result["confidence"]
    normal_pct = result["probabilities"]["NORMAL"]
    pneu_pct = result["probabilities"]["PNEUMONIA"]
    latency = result["latency_ms"]
    device = result["device"]

    print("\n----------------- DIAGNOSTIC RESULTS -----------------")
    print(f"  PREDICTION:    [{pred}]")
    print(f"  CONFIDENCE:    {conf:.2f}%")
    print(f"  PROBABILITIES: NORMAL: {normal_pct:.1f}%  |  PNEUMONIA: {pneu_pct:.1f}%")
    print(f"  INFERENCE TIME:{latency:.2f} ms ({device})")
    print(f"  SEVERITY:      {result['severity']}")
    print(f"\n  CLINICAL FINDINGS:\n  {result['clinical_notes']}")
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    main()
