# Phishing URL Detection System using Machine Learning
**Master of Engineering (M.E.) Thesis / Capstone Project**

---

## 1. Project Abstract
Phishing remains one of the most persistent and damaging cyber threat vectors, exploiting human trust to harvest sensitive credentials, financial data, and personal identifiable information (PII). Traditional defense mechanisms rely heavily on static blacklisting, which fails to capture zero-day phishing campaigns. 

This project presents a robust, high-performance **Machine Learning-based Phishing URL Detection System** designed for automated, rapid, and network-independent URL classification. We implement and comparatively analyze four fundamental classifiers:
1. **Logistic Regression** (Linear Classifier)
2. **Decision Tree** (Non-linear Rule Classifier)
3. **Random Forest** (Ensemble Bagging Classifier)
4. **Support Vector Machine (SVM)** (Margin-maximizing Classifier)

Our system processes raw URLs to extract **15 distinct lexical features**, eliminating the latency and safety risks associated with active page crawling or external WHOIS lookup. The system operates through a user-friendly, high-fidelity command-line interface (CLI) that supports automated model training on balanced subsets, real-time single URL analysis with threat probability scoring, and bulk classification reports.

---

## 2. Theoretical Architecture & Feature Engineering

The detection pipeline consists of four main logical layers:
```
[ Raw URL Input ] ──> [ Lexical Feature Extractor ] ──> [ StandardScaler ] ──> [ ML Engine (4 Classifiers) ] ──> [ Threat Verdict ]
```

### 2.1 Lexical Feature Selection & Mathematical Rationale
A total of 15 lexical features are extracted dynamically using our parsing framework. Below is their academic formulation:

| No. | Feature Name | Analytical Description | Phishing Correlation / Rationale |
|---|---|---|---|
| **1** | `url_length` | Total length of the URL character string. | Phishing URLs are statistically longer due to nested subdomains or credential forwarding parameters. |
| **2** | `hostname_length` | Total length of the domain/netloc portion. | Attackers often spoof brand names by using extended domains (e.g., `paypal-update-login-verification.com`). |
| **3** | `path_length` | Total length of the directory and filename path. | Deep directory structures are often used to conceal malicious scripts or spoof directory paths. |
| **4** | `qty_dot` | Count of `.` characters in the full URL. | High dot counts occur when attackers use nested subdomains to mimic legitimate brand hierarchies. |
| **5** | `qty_hyphen` | Count of `-` characters in the full URL. | Hyphens are rarely used in standard benign domains but are heavily used to link words in phishing (e.g., `secure-login-chase`). |
| **6** | `qty_at` | Count of `@` characters. | The presence of `@` causes the browser to ignore all preceding characters and redirect to the host listed after the `@`. |
| **7** | `qty_question` | Count of `?` query boundaries. | Denotes parameter queries, which are highly prevalent in phishing URLs containing credential redirection links. |
| **8** | `qty_equal` | Count of `=` assignment operators. | Indicates variable passing in query strings, typical in active phishing credential harvesting scripts. |
| **9** | `qty_slash` | Count of `/` folder separators. | Reflects directory depth. Multiple slashes indicate complex nesting common in deep-link phishing pages. |
| **10**| `qty_digit` | Count of numerical characters `[0-9]`. | Malicious domains often contain random generated digits to bypass static signature matching rules. |
| **11**| `qty_letter` | Count of alphabetic characters `[a-zA-Z]`. | Used as a baseline denominator for character density analysis. |
| **12**| `has_ip` | Boolean (`0` or `1`) check for IP Address host. | Legitimate companies use named domains. The use of raw IP addresses (IPv4 or Hex) is a strong indicator of illegal hosting. |
| **13**| `has_shortening` | Boolean check for known URL shorteners. | Shorteners (like `bit.ly` or `tinyurl.com`) are used to mask the true destination domain from user inspection. |
| **14**| `has_http_in_path` | Boolean check for `http`/`https` inside the path. | Phishing paths often embed an entire destination URL inside parameters to redirect users (e.g., `http://malicious.com?url=https://paypal.com`). |
| **15**| `suspicious_words` | Count of keywords: `login`, `verify`, `banking`, etc. | Attackers use high-urgency keywords to coerce users into typing credentials. |

---

## 3. Mathematical Foundations of the ML Algorithms

### 3.1 Logistic Regression
Logistic Regression is a parametric linear classification model that maps the linear combination of inputs to a probability space between $0$ and $1$ using the **Sigmoid function**:

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Where:
*   $\mathbf{w}$ is the learned weight vector.
*   $\mathbf{x}$ is the standardized feature vector.
*   $b$ is the bias.
*   Optimization is completed via minimizing the **Binary Cross-Entropy Loss** with L2 regularization:

$$J(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(P(y_i=1 \mid \mathbf{x}_i)) + (1-y_i) \log(1 - P(y_i=1 \mid \mathbf{x}_i)) \right] + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$$

### 3.2 Decision Tree Classifier
The Decision Tree builds an inverted hierarchical tree by recursively partitioning the training samples based on feature thresholds that maximize the reduction in uncertainty. We measure uncertainty using the **Gini Impurity** index:

$$I_G(t) = 1 - \sum_{i=1}^{C} p_i^2$$

Where $p_i$ is the probability of a sample belonging to class $i$ at node $t$. The algorithm splits a node by maximizing the **Information Gain** ($\Delta I_G$):

$$\Delta I_G = I_G(parent) - \left( \frac{N_{left}}{N} I_G(left) + \frac{N_{right}}{N} I_G(right) \right)$$

### 3.3 Random Forest Classifier
Random Forest is a meta-estimator ensemble that constructs a multitude of uncorrelated Decision Trees during training. It reduces high variance (overfitting) inherent in single trees through **Bootstrap Aggregating (Bagging)** and **Feature Subspace Sampling**:
1.  **Bootstrapping**: Generates $B$ distinct datasets by sampling the training data with replacement.
2.  **Random Feature Splitting**: At each node split in a tree, only a random subset of features $m \approx \sqrt{p}$ is considered.
3.  **Aggregated Inference**: The final prediction is a consensus majority vote from all $B$ individual trees:

$$\hat{y} = \text{mode} \{ T_1(\mathbf{x}), T_2(\mathbf{x}), \dots, T_B(\mathbf{x}) \}$$

### 3.4 Support Vector Machine (SVM)
The Support Vector Classifier constructs an optimal, margin-maximizing hyperplane in a high-dimensional feature space that separates legitimate and phishing URLs. For non-linear relationships, the inputs are mapped into higher-dimensional space using the Radial Basis Function (RBF) Kernel:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

The optimization objective minimizes training classification errors while maximizing the geometric margin:

$$\min_{\mathbf{w}, b, \mathbf{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T \phi(\mathbf{x}_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

Where $C$ acts as a regularization parameter controling the penalty for misclassification, and $\xi_i$ represents slack variables allowing margin violations.

---

## 4. Evaluation Metrics Formulation

To provide a complete, academically rigorous evaluation, the system computes the following metrics based on the **Confusion Matrix**:

| | Predicted Legit (0) | Predicted Phishing (1) |
|---|---|---|
| **Actual Legit (0)** | True Negatives (TN) | False Positives (FP) |
| **Actual Phishing (1)** | False Negatives (FN) | True Positives (TP) |

### 4.1 Accuracy
The ratio of correctly classified URLs to the total URL set:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### 4.2 Precision
The proportion of URLs predicted as phishing that are truly malicious. A high precision minimizes false positives (blocking safe user traffic):

$$\text{Precision} = \frac{TP}{TP + FP}$$

### 4.3 Recall (Sensitivity)
The proportion of actual phishing URLs that the system successfully detects. A high recall minimizes false negatives (letting cyberthreats bypass security):

$$\text{Recall} = \frac{TP}{TP + FN}$$

### 4.4 F1-Score
The harmonic mean of Precision and Recall, representing a balanced performance score for classification models:

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 5. Usage & Execution Instructions

### 5.1 Project Setup
Ensure Python 3.8+ and pip are installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 5.2 Launch the Application
Run the interactive console menu from the root directory:
```bash
python main.py
```

### 5.3 Interactive Options Walkthrough
1.  **Option 1: Train & Evaluate ML Models**
    *   Instructs the system to load, pre-process, and scale a balanced sample of **1,000 raw URLs** (50% benign, 50% phishing).
    *   Trains Logistic Regression, Decision Tree, Random Forest, and SVM models.
    *   Generates and prints a comparison table of Accuracy, Precision, Recall, F1-Score, and Train/Inference times.
    *   Automatically serializes all trained classifiers and the `StandardScaler` to `models/` directory for persistence.
2.  **Option 2: Predict Single URL**
    *   Prompts you to enter any raw URL (e.g., `http://secure-login.paypal.com/signin`).
    *   Displays the extracted 15-dimensional numerical feature vector for transparency.
    *   Runs the scaled feature vector through the 4 independent models, returning individual class predictions and probability confidence ratings.
    *   Performs a majority consensus voting process to report a definitive **Threat Verdict** (Safe, Moderate Risk, High Risk).
3.  **Option 3: Bulk Predict from TXT File**
    *   Loads an external TXT file (containing one URL per line).
    *   Processes features in batch, performs model consensus evaluation, and outputs a complete prediction spreadsheet to `data/bulk_predictions_report.csv`.
4.  **Option 4: View Performance Comparison Report**
    *   Loads and displays the performance statistics from the latest training session.
5.  **Option 5: Exit**
    *   Exits the system gracefully.
