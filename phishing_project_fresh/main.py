import os
import sys
import json
import time
import pandas as pd
import numpy as np
from src.utils import (
    clear_screen, print_banner, Colors, print_success, 
    print_error, print_warning, print_info, print_header, print_table, print_colored
)
from src.feature_extractor import extract_features, preprocess_url, FEATURE_NAMES, is_whitelisted
from src.data_loader import prepare_pipeline
from src.models import (
    train_and_evaluate_all, save_models_and_scaler, load_trained_models
)

METRICS_FILE = "models/metrics.json"

def save_metrics_to_disk(reports: list):
    """Saves evaluation reports as JSON to disk for persistence."""
    try:
        os.makedirs("models", exist_ok=True)
        with open(METRICS_FILE, "w") as f:
            json.dump(reports, f, indent=4)
    except Exception as e:
        print_warning(f"Could not persist performance metrics: {e}")

def load_metrics_from_disk() -> list:
    """Loads persisted evaluation reports from disk."""
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def display_metrics_table(reports: list):
    """Formats and prints evaluation metrics in a highly professional table."""
    headers = [
        f"{Colors.BOLD}Algorithm{Colors.RESET}",
        f"{Colors.BOLD}Accuracy{Colors.RESET}",
        f"{Colors.BOLD}Precision{Colors.RESET}",
        f"{Colors.BOLD}Recall{Colors.RESET}",
        f"{Colors.BOLD}F1-Score{Colors.RESET}",
        f"{Colors.BOLD}Train Time{Colors.RESET}",
        f"{Colors.BOLD}Inference Time{Colors.RESET}"
    ]
    
    rows = []
    for r in reports:
        rows.append([
            f"{Colors.BOLD}{Colors.CYAN}{r['Algorithm']}{Colors.RESET}",
            f"{Colors.GREEN if r['Accuracy'] > 0.9 else Colors.YELLOW}{r['Accuracy']*100:.2f}%{Colors.RESET}",
            f"{r['Precision']*100:.2f}%",
            f"{r['Recall']*100:.2f}%",
            f"{r['F1-Score']*100:.2f}%",
            f"{r['Train Time (s)']:.4f}s",
            f"{r['Inference Time (s)']:.4f}s"
        ])
    
    print_table(headers, rows)

def main():
    clear_screen()
    print_banner()
    
    # Attempt to load existing models on startup
    print_info("Checking for pre-trained models on disk...")
    models, scaler = load_trained_models()
    
    if models and scaler:
        print_success("Pre-trained models and scaler successfully loaded! Ready for immediate predictions.")
    else:
        print_warning("No pre-trained models found. Please train models first (Option 1) to enable URL predictions.")
        models, scaler = None, None
        
    while True:
        print(f"\n{Colors.BOLD}=== MAIN COMMAND MENU ==={Colors.RESET}")
        print(f"{Colors.CYAN}1.{Colors.RESET} Train & Evaluate ML Models (800 Balanced Raw URLs)")
        print(f"{Colors.CYAN}2.{Colors.RESET} Predict Single URL (Threat Analysis & Confidence Score)")
        print(f"{Colors.CYAN}3.{Colors.RESET} Bulk Predict URLs from TXT File (Generates CSV Report)")
        print(f"{Colors.CYAN}4.{Colors.RESET} View Performance Comparison Report")
        print(f"{Colors.CYAN}5.{Colors.RESET} Exit System")
        
        try:
            choice = input(f"\n{Colors.BOLD}Enter your choice (1-5): {Colors.RESET}").strip()
        except KeyboardInterrupt:
            print_colored("\nExiting System. Thank you!", Colors.YELLOW)
            sys.exit(0)
            
        if choice == "1":
            print_header("Model Training and Evaluation")
            confirm = input(f"Confirm training on 800 raw URLs? This will overwrite previous models. (y/n): ").strip().lower()
            if confirm != "y":
                print_info("Training cancelled.")
                continue
                
            print_info("Starting the training pipeline. Please wait...")
            start_time = time.time()
            
            try:
                # 1. Load, sample, extract features, scale, and split
                X_train, X_test, y_train, y_test, scaler_obj = prepare_pipeline(sample_size=800)

                
                # 2. Train and evaluate all models
                trained_models, reports = train_and_evaluate_all(X_train, X_test, y_train, y_test)
                
                # 3. Save serialized assets
                save_models_and_scaler(trained_models, scaler_obj)
                save_metrics_to_disk(reports)
                
                # 4. Update runtime variables
                models = trained_models
                scaler = scaler_obj
                
                # 5. Display comparison report
                print_header("Evaluation Comparison Report")
                display_metrics_table(reports)
                
                elapsed = time.time() - start_time
                print_success(f"Pipeline completed successfully in {elapsed:.2f} seconds!")
                
            except Exception as e:
                print_error(f"An unexpected error occurred during training: {e}")
                
        elif choice == "2":
            print_header("Single URL Threat Analysis")
            if not models or not scaler:
                print_error("Models are not trained or loaded. Please run Option 1 first!")
                continue
                
            url = input(f"{Colors.BOLD}Enter URL to analyze: {Colors.RESET}").strip()
            if not url:
                print_warning("Empty URL entered. Returning to menu.")
                continue
                
            print_info(f"Analyzing URL: {url}")
            
            # Check whitelist/allowlist first
            if is_whitelisted(url):
                print_header("Consensus Threat Verdict")
                verdict_str = f"{Colors.BG_GREEN}{Colors.BOLD}  SAFE - TRUSTED DOMAIN (WHITELISTED)  {Colors.RESET}"
                desc = "This URL belongs to a globally trusted, verified domain. Bypassing machine learning threat evaluation."
                print(f"Verdict: {verdict_str}")
                print(f"Details: {desc}\n")
                continue
                
            # 1. Preprocess & Extract Features
            feats = extract_features(url)
            
            # Print feature breakdown
            print(f"\n{Colors.BOLD}Extracted Lexical Feature Vector:{Colors.RESET}")
            feat_rows = []
            for name in FEATURE_NAMES:
                val = feats[name]
                feat_rows.append([f"{Colors.CYAN}{name}{Colors.RESET}", val])
            print_table([f"{Colors.BOLD}Feature Name{Colors.RESET}", f"{Colors.BOLD}Value{Colors.RESET}"], feat_rows, table_fmt="simple")
            
            # 2. Prepare for prediction (reshape to 2D and scale)
            X = np.array([feats[name] for name in FEATURE_NAMES]).reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            # 3. Run predictions across all 4 models
            predictions = []
            phish_votes = 0
            
            for name, model in models.items():
                pred = int(model.predict(X_scaled)[0])
                prob = model.predict_proba(X_scaled)[0]
                confidence = prob[1] if pred == 1 else prob[0]
                
                status = f"{Colors.RED}{Colors.BOLD}PHISHING{Colors.RESET}" if pred == 1 else f"{Colors.GREEN}{Colors.BOLD}LEGITIMATE{Colors.RESET}"
                if pred == 1:
                    phish_votes += 1
                    
                predictions.append([
                    f"{Colors.BOLD}{name}{Colors.RESET}",
                    status,
                    f"{confidence*100:.2f}%"
                ])
                
            print_header("Model Predictions & Confidence scores")
            print_table([f"{Colors.BOLD}Algorithm{Colors.RESET}", f"{Colors.BOLD}Prediction{Colors.RESET}", f"{Colors.BOLD}Confidence{Colors.RESET}"], predictions)
            
            # 4. Final Threat Verdict (Majority Voting)
            print_header("Consensus Threat Verdict")
            if phish_votes >= 3:
                verdict_str = f"{Colors.BG_RED}{Colors.BOLD}  HIGH RISK - PHISHING URL  {Colors.RESET}"
                desc = "This URL exhibits strong malicious patterns across multiple models. DO NOT visit."
            elif phish_votes == 2:
                verdict_str = f"{Colors.YELLOW}{Colors.BOLD}  MODERATE RISK - SUSPICIOUS URL  {Colors.RESET}"
                desc = "Models are split on this URL. Exercise extreme caution."
            else:
                verdict_str = f"{Colors.BG_GREEN}{Colors.BOLD}  SAFE - LEGITIMATE URL  {Colors.RESET}"
                desc = "Models show a strong consensus that this URL is benign and safe."
                
            print(f"Verdict: {verdict_str}")
            print(f"Details: {desc}\n")
            
        elif choice == "3":
            print_header("Bulk URL Prediction")
            if not models or not scaler:
                print_error("Models are not trained or loaded. Please run Option 1 first!")
                continue
                
            filepath = input(f"{Colors.BOLD}Enter path to TXT file containing URLs (one per line): {Colors.RESET}").strip()
            if not os.path.exists(filepath):
                print_error(f"File not found: '{filepath}'. Please check the path.")
                continue
                
            try:
                with open(filepath, "r") as f:
                    urls = [line.strip() for line in f if line.strip()]
                    
                if not urls:
                    print_warning("No valid URLs found in the file.")
                    continue
                    
                print_info(f"Found {len(urls)} URLs. Processing batch. Please wait...")
                
                bulk_results = []
                
                for idx, url in enumerate(urls, 1):
                    # Check whitelist first
                    if is_whitelisted(url):
                        row_data = {
                            "url": url,
                            "final_prediction": "legitimate",
                            "consensus_confidence": 1.0
                        }
                        for name in models.keys():
                            clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                            row_data[f"{clean_name}_pred"] = "legitimate"
                            row_data[f"{clean_name}_conf"] = 1.0
                        bulk_results.append(row_data)
                        continue
                        
                    # Extract & Scale features
                    feats = extract_features(url)
                    X = np.array([feats[name] for name in FEATURE_NAMES]).reshape(1, -1)
                    X_scaled = scaler.transform(X)
                    
                    votes = []
                    confidences = {}
                    
                    # Run predictions
                    row_data = {"url": url}
                    for name, model in models.items():
                        pred = int(model.predict(X_scaled)[0])
                        prob = model.predict_proba(X_scaled)[0]
                        conf = prob[1] if pred == 1 else prob[0]
                        
                        votes.append(pred)
                        # Add individual model predictions to the output data dictionary
                        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                        row_data[f"{clean_name}_pred"] = "phishing" if pred == 1 else "legitimate"
                        row_data[f"{clean_name}_conf"] = conf
                        
                    # Majority Vote
                    avg_vote = np.mean(votes)
                    final_pred = 1 if avg_vote >= 0.5 else 0
                    
                    # Calculate consensus confidence
                    phish_conf = np.mean([row_data[f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_conf"] for name, m in models.items() if m.predict(X_scaled)[0] == final_pred])
                    
                    row_data["final_prediction"] = "phishing" if final_pred == 1 else "legitimate"
                    row_data["consensus_confidence"] = phish_conf
                    
                    bulk_results.append(row_data)
                    
                # Save as CSV
                results_df = pd.DataFrame(bulk_results)
                output_csv = "data/bulk_predictions_report.csv"
                os.makedirs("data", exist_ok=True)
                results_df.to_csv(output_csv, index=False)
                
                print_success(f"Successfully processed {len(urls)} URLs!")
                print_success(f"Report exported to CSV: '{output_csv}'")
                
                # Show preview table
                print_header("First 5 Bulk Results Preview")
                preview_rows = []
                for _, row in results_df.head(5).iterrows():
                    short_url = row["url"][:40] + "..." if len(row["url"]) > 43 else row["url"]
                    pred_colored = f"{Colors.RED}{Colors.BOLD}PHISHING{Colors.RESET}" if row["final_prediction"] == "phishing" else f"{Colors.GREEN}{Colors.BOLD}LEGITIMATE{Colors.RESET}"
                    preview_rows.append([short_url, pred_colored, f"{row['consensus_confidence']*100:.2f}%"])
                    
                print_table([f"{Colors.BOLD}URL{Colors.RESET}", f"{Colors.BOLD}Prediction{Colors.RESET}", f"{Colors.BOLD}Consensus Conf.{Colors.RESET}"], preview_rows)
                
            except Exception as e:
                print_error(f"Error during bulk processing: {e}")
                
        elif choice == "4":
            print_header("Model Performance Report")
            reports = load_metrics_from_disk()
            if reports:
                print_info("Displaying the cached performance report of the last training run:")
                display_metrics_table(reports)
            else:
                print_warning("No saved metrics found. Please train models first (Option 1) to generate a performance report.")
                
        elif choice == "5":
            print_colored("\nExiting System. Thank you for using the Phishing URL ML Detector! Goodbye.", Colors.CYAN)
            break
            
        else:
            print_error("Invalid choice! Please select an option between 1 and 5.")

if __name__ == "__main__":
    main()
