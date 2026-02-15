# Scalable & Resilient Symmetric Cipher (LS-RM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-MDPI%20Cryptography-blue)](https://doi.org/10.3390/cryptography9040070)

This repository contains the **Python implementation** of the cipher proposed in the paper:
**"A Scalable Symmetric Cryptographic Scheme Based on Latin Square, Permutations, and Reed-Muller Codes for Resilient Encryption"**.

The design integrates **Latin Squares** for non-linear substitution and **Reed-Muller (RM) codes** for diffusion. Unlike traditional designs, this scheme leverages the error-correcting properties of Reed-Muller codes to provide **resilience against fault attacks** and noise, ensuring data integrity even in unstable environments.

##  Project Structure

| File | Purpose | Description |
| :--- | :--- | :--- |
| `cipher_benchmark.py` | **Performance** | Measures throughput and execution time. Benchmarked against **AES-128**, **PRESENT-80**, and **PRESENT-128**. |
| `security_analysis.py` | **Security** | Analyzes the **Avalanche Effect** (SAC) and checks for **Differential Patterns** in the ciphertext. |
| `footprint_analysis.py` | **Memory** | Profiles the **RAM/Flash footprint** required to initialize the Latin Square and permutation structures. |

##  Technical Features

* **Fault-Resilient Diffusion:** Uses **Reed-Muller (RM) codes** as a diffusion layer. This approach provides high branch numbers and inherent resistance against fault injection attacks by leveraging error-correction capabilities.
* **Custom S-Boxes:** Utilizes dynamic $N \times N$ **Latin Squares** for robust non-linear substitution (confusion).
* **Scalable Architecture:** Supports variable block sizes and parameter flexibility ($n, k, d$). Verified configurations include:
    * **$N=32, K=26$** (High resilience/throughput)
    * **$N=16, K=15$** (Lightweight/constrained)
* **Efficient Deployment:** Includes **precomputed components** to optimize performance across multi-core or distributed systems.

##  Performance & Evaluation

The repository evaluates the cipher across three critical academic pillars:

1.  **Performance Analysis (`cipher_benchmark.py`):**
    * Compares encryption/decryption throughput against industry standards: **AES-128**, **PRESENT-80**, and **PRESENT-128**.
    * Validates execution time across $2\log_2 n$ rounds of transformation.

2.  **Security Analysis (`security_analysis.py`):**
    * Verifies the **Strict Avalanche Criterion (SAC)**, ensuring a ~50% bit-flip probability to resist statistical attacks.
    * Evaluates resistance against linear and differential cryptanalysis through pattern detection.

3.  **Memory Footprint (`footprint_analysis.py`):**
    * Measures the **storage and computational overhead** of the Latin Square mappings.
    * Profiles memory usage to demonstrate scalability for resource-constrained IoT or embedded environments.

##  Installation & Usage

**Prerequisites:**
* Python 3.8+

**Install Dependencies:**
```bash
pip install numpy reedmuller pycryptodome psutil

```

**Run Evaluation Scripts:**

```bash
# To run performance benchmarks
python cipher_benchmark.py

# To run security analysis
python security_analysis.py

# To run memory footprint analysis
python footprint_analysis.py

```

##  Citation

If you use this code in your research, please cite the original paper:

> **Ahmad, H.; Hannusch, C.** (2025). "A Scalable Symmetric Cryptographic Scheme Based on Latin Square, Permutations, and Reed-Muller Codes for Resilient Encryption." *Cryptography*, 9(4), 70.
> 🔗 **[Read Full Paper (MDPI)](https://doi.org/10.3390/cryptography9040070)**
