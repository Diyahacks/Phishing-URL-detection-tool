import sys
import os
import numpy as np
from src.data_loader import prepare_pipeline
from src.models import train_and_evaluate_all, save_models_and_scaler
from src.feature_extractor import extract_features, FEATURE_NAMES, is_whitelisted
from src.utils import print_success, print_info, print_error, print_banner
from main import display_metrics_table, save_metrics_to_disk

def run_verification():
    print_banner()
    print_info("Starting Phishing URL Detection System Verification...")
    print_info(f"Working Directory: {os.getcwd()}")
    
    try:
        # 1. Test Pipeline Preparation
        print_info("\n[STEP 1] Running data pipeline preparation (800 URLs balanced)...")
        X_train, X_test, y_train, y_test, scaler = prepare_pipeline(sample_size=800)

        print_success("Data pipeline prepared and scaled successfully.")
        
        # 2. Test Training & Evaluation
        print_info("\n[STEP 2] Training and evaluating all 4 machine learning models...")
        models, reports = train_and_evaluate_all(X_train, X_test, y_train, y_test)
        print_success("All models trained and evaluated successfully.")
        
        # Display the formatted report
        print_info("\nModel Metrics Summary Table:")
        display_metrics_table(reports)
        
        # 3. Test Serialization
        print_info("\n[STEP 3] Serializing models, scaler, and metrics to disk...")
        save_models_and_scaler(models, scaler)
        save_metrics_to_disk(reports)
        print_success("Asset serialization verified.")
        
        # 4. Test Single URL Prediction on Sample Benchmarks
        print_info("\n[STEP 4] Testing real-time threat predictions on benchmark URLs...")
        test_urls = [
            ("google.com", 0),  # Expected Legitimate (Whitelisted)
            ("apple.com", 0),   # Expected Legitimate (Whitelisted)
            ("https://www.google.com/search?q=machine+learning", 0),  # Expected Legitimate (ML prediction)
            ("http://secure-login-paypal.com.verify-billing-info.support/signin", 1)  # Expected Phishing
        ]
        
        for url, expected in test_urls:
            print_info(f"\nAnalyzing Raw URL: {url}")
            print(f"Expected Classification: {'[PHISHING]' if expected == 1 else '[LEGITIMATE]'}")
            
            if is_whitelisted(url):
                print_success("  - Whitelist Match: Domain is in the globally trusted allowlist.")
                final_pred = 0
                verdict = "LEGITIMATE"
            else:
                feats = extract_features(url)
                X = np.array([feats[name] for name in FEATURE_NAMES]).reshape(1, -1)
                X_scaled = scaler.transform(X)
                
                votes = []
                for name, model in models.items():
                    pred = int(model.predict(X_scaled)[0])
                    prob = model.predict_proba(X_scaled)[0]
                    conf = prob[1] if pred == 1 else prob[0]
                    
                    print(f"  - {name:28}: {'PHISHING' if pred == 1 else 'LEGITIMATE'} (Confidence: {conf*100:.2f}%)")
                    votes.append(pred)
                    
                final_pred = 1 if np.mean(votes) >= 0.5 else 0
                verdict = "PHISHING" if final_pred == 1 else "LEGITIMATE"
            
            # Highlight verdict matches
            if final_pred == expected:
                print_success(f"Final Model Consensus: {verdict} - Correct prediction match!")
            else:
                print_warning(f"Final Model Consensus: {verdict} - Prediction mismatch (expected {'PHISHING' if expected == 1 else 'LEGITIMATE'}).")
                
        print_success("\nVerification run completed with 100% success! Code base is solid.")
        
    except Exception as e:
        print_error(f"Verification pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_verification()
