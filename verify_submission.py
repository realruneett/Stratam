import os
import sys
import json
import pandas as pd

def main():
    run_num = ""
    if len(sys.argv) > 1:
        run_num = f"_{sys.argv[1]}"
        
    metrics_file = f"metrics{run_num}.json"
    submission_file = f"submission{run_num}.csv"

    print("=" * 60)
    print(f"      Stratam Submission & Metrics Verification Tool ({metrics_file})      ")
    print("=" * 60)

    # 1. Check metrics file
    if not os.path.exists(metrics_file):
        print(f"[ERROR] {metrics_file} not found!")
        sys.exit(1)
        
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
        
    ens_r2 = metrics.get("ensemble_r2", 0)
    val_r2 = metrics.get("validation_r2", 0)
    
    target_ens_r2 = 0.95
    target_val_r2 = 0.93
    
    print(f"Checking metrics against targets:")
    print(f"  • Ensemble OOF R²      : {ens_r2:.5f} (Target: >= {target_ens_r2:.2f})")
    print(f"  • Holdout Validation R²: {val_r2:.5f} (Target: >= {target_val_r2:.2f})")
    
    metrics_ok = True
    if ens_r2 < target_ens_r2:
        print(f"    [FAIL] Ensemble OOF R² is below the target!")
        metrics_ok = False
    else:
        print(f"    [PASS] Ensemble OOF R² target met.")
        
    if val_r2 < target_val_r2:
        print(f"    [FAIL] Holdout Validation R² is below the target!")
        metrics_ok = False
    else:
        print(f"    [PASS] Holdout Validation R² target met.")
        
    # 2. Check submission file
    if not os.path.exists(submission_file):
        print(f"[ERROR] {submission_file} not found!")
        sys.exit(1)
        
    sub = pd.read_csv(submission_file)
    expected_rows = 41778
    actual_rows = len(sub)
    nulls = sub.isnull().sum().sum()
    
    print(f"\nChecking submission structure:")
    print(f"  • Expected Row Count  : {expected_rows}")
    print(f"  • Actual Row Count    : {actual_rows}")
    print(f"  • Null Values Count   : {nulls}")
    
    struct_ok = True
    if actual_rows != expected_rows:
        print(f"    [FAIL] Submission row count mismatch!")
        struct_ok = False
    else:
        print(f"    [PASS] Row count verified.")
        
    if nulls > 0:
        print(f"    [FAIL] Submission contains nulls!")
        struct_ok = False
    else:
        print(f"    [PASS] No null values verified.")
        
    print("=" * 60)
    if metrics_ok and struct_ok:
        print("🎉 [SUCCESS] ALL TARGETS AND METRICS ARE VERIFIED & PASSED!")
        print("Your submission is 100% compliant and ready for the leaderboard.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ [FAILURE] Target checks failed. Please adjust codes or parameters and re-train.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
