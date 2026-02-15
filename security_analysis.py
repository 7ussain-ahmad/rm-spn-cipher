import itertools
import numpy as np
import time
import math
from reedmuller.reedmuller import ReedMuller
import binascii # Keep for potential debugging if needed
from collections import Counter # For pattern consistency check

# --- Cipher System Implementation ---
class CipherSystem:
    def __init__(self, K_array, L, subsets, n=32, k=26, d=None, seed=42):
        # Basic Parameters
        self.n = n
        self.log_n = int(np.log2(n)) if n > 0 and (n & (n - 1)) == 0 else None
        if self.log_n is None: raise ValueError("n must be a power of 2")
        self.k = k
        self.alpha = self.log_n * self.k # Total bits in message/key/state
        self.num_rounds = 2 * self.log_n

        # Derived Parameters
        self.num_digits = self.k # Number of base-n digits
        self.bits_per_digit = self.log_n
        self.num_rm_portions = self.log_n
        self.codeword_length = self.n
        self.output_length = self.n * self.log_n # Ciphertext length in bits

        # Key and Latin Square Setup
        if not isinstance(K_array, np.ndarray) or K_array.dtype != np.uint8 or len(K_array) != self.alpha:
            raise ValueError(f"K must be a NumPy uint8 array of length {self.alpha}")
        self.K = K_array # Store K as NumPy array

        np.random.seed(seed)
        L_shuffled = L.copy()
        row_perm = np.random.permutation(self.n)
        col_perm = np.random.permutation(self.n)
        self.L = L_shuffled[row_perm][:, col_perm] # Use shuffled LS

        # Precompute conjugate Latin Square
        self.L_conj = np.zeros((self.n, self.n), dtype=int)
        for r_idx in range(self.n):
            for c_idx in range(self.n):
                symbol = self.L[r_idx, c_idx]
                self.L_conj[symbol, r_idx] = c_idx

        # Reed-Muller Setup
        self.r = self.determine_r_from_k()
        self.d = d if d is not None else 2**(self.log_n - self.r) if 0 <= self.r < self.log_n else 1
        try:
            self.rm = ReedMuller(self.r, self.log_n)
        except Exception as e:
            print(f"Error initializing ReedMuller(r={self.r}, m={self.log_n}): {e}")
            raise

        # Permutation Precomputation
        self.subsets = subsets
        np.random.seed(seed)
        # Store rho permutations as NumPy arrays internally
        self.rho_list_np = self._generate_rho_permutations_np()
        # Keep a list version if needed
        self.rho_list = [p.tolist() for p in self.rho_list_np]

        self.pi_list, self.key_digits_list = self._precompute_pi_and_keyschedule()

    def determine_r_from_k(self):
        m = self.log_n
        current_k_sum = 0
        for r_candidate in range(m + 1):
            current_k_sum += math.comb(m, r_candidate)
            if current_k_sum >= self.k:
                return r_candidate
        return m

    # NumPy-based Helper Functions
    def _binary_array_to_baseN(self, bin_array):
        if len(bin_array) != self.alpha:
            raise ValueError(f"Input binary array length must be {self.alpha}, got {len(bin_array)}")
        try:
            reshaped = bin_array.reshape((self.num_digits, self.bits_per_digit))
        except ValueError as e:
            raise ValueError(f"Cannot reshape array of length {len(bin_array)} into ({self.num_digits}, {self.bits_per_digit})") from e

        powers = 2**np.arange(self.bits_per_digit, dtype=np.uint64)
        digits = np.dot(reshaped.astype(np.uint64), powers)
        return digits.tolist()

    def _baseN_to_binary_array(self, digits):
        if len(digits) != self.num_digits:
            raise ValueError(f"Input digits list length must be {self.num_digits}, got {len(digits)}")
        digits_array = np.array(digits, dtype=int)
        powers = np.arange(self.bits_per_digit)
        bin_array_reshaped = ((digits_array[:, None] >> powers) & 1).astype(np.uint8)
        bin_array = bin_array_reshaped.flatten()
        if len(bin_array) != self.alpha:
            bin_array = np.pad(bin_array, (0, self.alpha - len(bin_array)), 'constant', constant_values=0).astype(np.uint8)[:self.alpha]
        return bin_array

    def _apply_permutation_np(self, vector_array, perm_indices):
        if len(vector_array) != len(perm_indices):
            raise ValueError(f"Vector length {len(vector_array)} and permutation length {len(perm_indices)} must match.")
        perm_indices_np = np.array(perm_indices, dtype=int)
        permuted_array = np.zeros_like(vector_array)
        permuted_array[perm_indices_np] = vector_array
        return permuted_array

    def _inverse_permutation_np(self, perm_indices):
        perm_indices_np = np.array(perm_indices, dtype=int)
        inverse = np.zeros_like(perm_indices_np, dtype=int)
        inverse[perm_indices_np] = np.arange(len(perm_indices_np), dtype=int)
        return inverse

    # Precomputation Methods
    def _generate_rho_permutations_np(self):
        perms = []
        num_rho_needed = self.num_rm_portions
        if len(self.subsets) < num_rho_needed:
            print(f"Warning: Need {num_rho_needed} (Ir, Ic) pairs for rho, got {len(self.subsets)}. Reusing subsets.")

        for z in range(num_rho_needed):
            subset_index = z % len(self.subsets) # Reuse subsets cyclically
            I_r, I_c = self.subsets[subset_index]
            current_perm_indices = np.arange(self.n, dtype=int)

            for i in I_r:
                sigma_i = np.argsort(self.L[i])
                current_perm_indices = current_perm_indices[sigma_i]

            for j in I_c:
                tau_j = np.argsort(self.L[:, j])
                current_perm_indices = current_perm_indices[tau_j]

            perms.append(current_perm_indices) # Store NumPy array
        return perms


    def _generate_pi_permutation(self, z, key_digits_round):
        rho_index = (z // 2) % self.num_rm_portions
        if rho_index >= len(self.rho_list_np):
            raise IndexError(f"rho_index {rho_index} out of bounds for rho_list_np (len {len(self.rho_list_np)})")
        rho_z = self.rho_list_np[rho_index]

        key_digits_round_arr = np.array(key_digits_round, dtype=int)

        v_values = np.zeros(self.alpha, dtype=int)
        indices_n = np.arange(self.n, dtype=int)
        key_indices_n = indices_n % self.num_digits
        rho_vals_n = rho_z[indices_n % self.n]

        v_values[:self.n] = (rho_vals_n + key_digits_round_arr[key_indices_n] * (indices_n**2 + 1) + z * (indices_n + 1)) % self.alpha

        if self.alpha > self.n:
            indices_alpha_minus_n = np.arange(self.alpha - self.n, dtype=int)
            key_indices_alpha = indices_alpha_minus_n % self.num_digits
            v_values[self.n:] = (key_digits_round_arr[key_indices_alpha] * (z + indices_alpha_minus_n + 1) * (indices_alpha_minus_n + 2)) % self.alpha

        sum_d = np.sum(key_digits_round_arr)
        indices_alpha = np.arange(self.alpha, dtype=int)
        key_indices_alpha_full = indices_alpha % self.num_digits
        sort_key_modifier = (indices_alpha + sum_d + z) % self.alpha

        dtype = [('v', int), ('d', int), ('mod', int), ('orig_idx', int)]
        tuples_to_sort = np.array(list(zip(v_values, key_digits_round_arr[key_indices_alpha_full], sort_key_modifier, indices_alpha)), dtype=dtype)

        sorted_array = np.sort(tuples_to_sort, order=['v', 'd', 'mod'])

        temp_pi = np.zeros(self.alpha, dtype=int)
        temp_pi[sorted_array['orig_idx']] = np.arange(self.alpha, dtype=int)

        shift_amount = (sum_d + z) % self.alpha
        final_pi = np.roll(temp_pi, -shift_amount)

        return final_pi # Return NumPy array


    def _precompute_pi_and_keyschedule(self):
        key_digits_list = []
        try:
            K_digits = self._binary_array_to_baseN(self.K)
            key_digits_list.append(K_digits)

            for z in range(1, self.num_rounds):
                if z == 1:
                    prev = np.array(key_digits_list[0], dtype=int)
                    indices = np.arange(self.num_digits, dtype=int)
                    next_indices = (indices + 1) % self.num_digits
                    D_digits_z = self.L[prev, prev[next_indices]].tolist()
                else:
                    prev = np.array(key_digits_list[z - 1], dtype=int)
                    prev_prev = np.array(key_digits_list[z - 2], dtype=int)
                    D_digits_z = self.L[prev, prev_prev].tolist()
                key_digits_list.append(D_digits_z)

        except Exception as e:
            print(f"Error during key schedule generation at round z={z if 'z' in locals() else 'unknown'}: {e}")
            raise

        pi_list = [self._generate_pi_permutation(z, key_digits_list[z]) for z in range(self.num_rounds)]

        return pi_list, key_digits_list


    # Core SPN Functions
    def multi_round_mapping(self, M_array):
        if len(M_array) != self.alpha: raise ValueError("Input M length mismatch")
        current_digits = self._binary_array_to_baseN(M_array)

        for z in range(self.num_rounds):
            key_round = self.key_digits_list[z]
            perm_round_indices = self.pi_list[z]

            subst_digits = [self.L[key_round[i]][current_digits[i]] for i in range(self.num_digits)]

            subst_array = self._baseN_to_binary_array(subst_digits)
            permuted_array = self._apply_permutation_np(subst_array, perm_round_indices)

            current_digits = self._binary_array_to_baseN(permuted_array)

        Y_array = self._baseN_to_binary_array(current_digits)
        return Y_array

    def inverse_multi_round_mapping(self, Y_array):
        if len(Y_array) != self.alpha: raise ValueError("Input Y length mismatch")
        current_digits = self._binary_array_to_baseN(Y_array)

        for z in range(self.num_rounds - 1, -1, -1):
            key_round = self.key_digits_list[z]
            perm_round_indices = self.pi_list[z]
            inv_perm_round_indices = self._inverse_permutation_np(perm_round_indices)

            current_array = self._baseN_to_binary_array(current_digits)
            inv_permuted_array = self._apply_permutation_np(current_array, inv_perm_round_indices)

            inv_permuted_digits = self._binary_array_to_baseN(inv_permuted_array)

            current_digits = [self.L_conj[inv_permuted_digits[i]][key_round[i]] for i in range(self.num_digits)]

        M_array = self._baseN_to_binary_array(current_digits)
        return M_array

    # Encryption/Decryption for Analysis (No Timing)
    def encrypt(self, M_array):
        Y_array = self.multi_round_mapping(M_array)

        Y_portions = np.split(Y_array, self.num_rm_portions)
        codewords = []
        for i, portion_np in enumerate(Y_portions):
            cw = self.rm.encode(portion_np)
            rho_perm_indices = self.rho_list_np[i]
            permuted_cw = self._apply_permutation_np(cw, rho_perm_indices)
            codewords.append(permuted_cw)

        C_array = np.concatenate(codewords)
        if len(C_array) != self.output_length:
            raise ValueError(f"Output C length error: {len(C_array)} vs {self.output_length}")
        return C_array

    def decrypt(self, C_array):
        if len(C_array) != self.output_length:
            raise ValueError(f"Input C length error: {len(C_array)} vs {self.output_length}")
        C_portions = np.split(C_array, self.num_rm_portions)
        inverse_rho_perms = [self._inverse_permutation_np(self.rho_list_np[i]) for i in range(self.num_rm_portions)]

        decoded_portions = []
        for i, c_portion_np in enumerate(C_portions):
            r_np = self._apply_permutation_np(c_portion_np, inverse_rho_perms[i])
            y = self.rm.decode(r_np)

            y_final = np.zeros(self.k, dtype=np.uint8)
            if y is not None and len(y) == self.k:
                y_final = y.astype(np.uint8)
            decoded_portions.append(y_final)

        Y_array = np.concatenate(decoded_portions)
        if len(Y_array) != self.alpha:
            Y_array = np.pad(Y_array, (0, self.alpha - len(Y_array)), 'constant', constant_values=0).astype(np.uint8)[:self.alpha]

        M_array = self.inverse_multi_round_mapping(Y_array)
        if len(M_array) != self.alpha:
            raise ValueError(f"Output M length error: {len(M_array)} vs {self.alpha}")
        return M_array


    # Avalanche and Differential Analysis Functions
    def analyze_avalanche_effect_all_bits(self, messages_np_list):
        """Analyze avalanche effect (HD and distribution) for every bit position."""
        if not messages_np_list: return {}
        bit_length = self.alpha
        if len(messages_np_list[0]) != bit_length:
            raise ValueError(f"Input message length {len(messages_np_list[0])} does not match cipher alpha {self.alpha}")

        avalanche_data = {}
        section_size = self.n
        num_sections = self.num_rm_portions
        section_boundaries = [(i * section_size, (i + 1) * section_size) for i in range(num_sections)]

        for bit_pos in range(bit_length):
            h_distances = []
            section_flip_counts_list = []

            for msg_np in messages_np_list:
                flipped_msg_np = msg_np.copy()
                flipped_msg_np[bit_pos] = 1 - flipped_msg_np[bit_pos]

                C_orig_np = self.encrypt(msg_np)
                C_flipped_np = self.encrypt(flipped_msg_np)

                diff_array = np.logical_xor(C_orig_np, C_flipped_np)
                diff_count = np.sum(diff_array)
                h_distances.append(diff_count)

                changed_indices = np.where(diff_array == 1)[0]

                counts_this_flip = np.zeros(num_sections, dtype=int)
                for idx in changed_indices:
                    for sec_idx, (start, end) in enumerate(section_boundaries):
                        if start <= idx < end:
                            counts_this_flip[sec_idx] += 1
                            break
                section_flip_counts_list.append(counts_this_flip)

            avg_h_distance = np.mean(h_distances) if h_distances else 0
            avg_avalanche_perc = (avg_h_distance / self.output_length) * 100 if self.output_length > 0 else 0
            individual_percentages = [(hd / self.output_length) * 100 for hd in h_distances] if self.output_length > 0 else [0]*len(h_distances)
            avg_section_counts = np.mean(section_flip_counts_list, axis=0) if section_flip_counts_list else np.zeros(num_sections)
            avg_section_perc = (avg_section_counts / avg_h_distance * 100) if avg_h_distance > 0 else np.zeros(num_sections)

            avalanche_data[bit_pos] = {
                'avg_h_distance': avg_h_distance,
                'avg_avalanche_perc': avg_avalanche_perc,
                'individual_h_distances': h_distances,
                'individual_percentages': individual_percentages,
                'avg_section_counts': avg_section_counts.tolist(),
                'avg_section_percentages': avg_section_perc.tolist()
            }
        return avalanche_data

    def analyze_all_bits_differential_pattern(self, messages_np_list):
        """Analyze differential patterns (NumPy arrays) when flipping all bits."""
        if not messages_np_list: return {}
        bit_length = self.alpha
        all_patterns = {}

        for bit_pos in range(bit_length):
            patterns = {}
            for i, msg_np in enumerate(messages_np_list):
                flipped_msg_np = msg_np.copy()
                flipped_msg_np[bit_pos] = 1 - flipped_msg_np[bit_pos]
                C_orig_np = self.encrypt(msg_np)
                C_flipped_np = self.encrypt(flipped_msg_np)
                diff_pattern_np = np.logical_xor(C_orig_np, C_flipped_np).astype(np.uint8)
                patterns[i] = diff_pattern_np
            all_patterns[bit_pos] = patterns
        return all_patterns

    def detect_pattern_consistency(self, differential_patterns_dict):
        """Detect if multiple messages yield the same output diff pattern for a given input bit flip."""
        consistency = {}
        for bit_pos, patterns in differential_patterns_dict.items():
            pattern_tuples = [tuple(p) for p in patterns.values()]
            num_patterns = len(pattern_tuples)
            if num_patterns == 0: continue

            unique_patterns = set(pattern_tuples)
            num_unique = len(unique_patterns)

            if num_unique < num_patterns:
                pattern_counts = Counter(pattern_tuples)
                most_common_pattern_tuple, common_freq = pattern_counts.most_common(1)[0]
                most_common_pattern = np.array(most_common_pattern_tuple, dtype=np.uint8)
                frequency = common_freq / num_patterns
                consistency[bit_pos] = {'unique_patterns': num_unique, 'total_patterns': num_patterns, 'most_common_pattern': most_common_pattern, 'frequency': frequency}
            else:
                consistency[bit_pos] = {'unique_patterns': num_unique, 'total_patterns': num_patterns, 'most_common_pattern': None, 'frequency': 0.0}
        return consistency


# MAIN EXECUTION FOR ANALYSIS
if __name__ == "__main__":

    # Configuration
    params_n32_k26 = {
        'a': 130, # k * log_n = 26 * 5
        'n': 32,
        'k': 26,
        'd': 4
    }
    params_n32_k16 = {
        'a': 80, # k * log_n = 16 * 5
        'n': 32,
        'k': 16,
        'd': 8
    }
    params_n16_k15 = {
        'a': 60, # k * log_n = 15 * 4
        'n': 16,
        'k': 15,
        'd': 2
    }

    # Select which parameter set to analyze
    # --- CHANGE PARAMETERS HERE ---
    params_to_analyze = params_n32_k26
    num_messages_to_test = 100 # Number of random messages
    # ------------------------------

    # Setup Cipher
    a = params_to_analyze['a']
    n = params_to_analyze['n']
    k = params_to_analyze['k']
    d = params_to_analyze['d']

    K_array = np.random.randint(0, 2, size=a, dtype=np.uint8)
    L = np.array([[((i + j) % n) for j in range(n)] for i in range(n)])
    subsets = []
    required_pairs = int(np.log2(n))
    if required_pairs <= 0: required_pairs = 1
    np.random.seed(42)
    for _ in range(required_pairs):
        I_r = set(np.random.choice(range(n), size=np.random.randint(1, n//2 + 1), replace=False))
        remaining = list(set(range(n)) - I_r)
        if not remaining: remaining = list(range(n))
        I_c = set(np.random.choice(remaining, size=np.random.randint(1, min(len(remaining), n//2 + 1)), replace=False))
        subsets.append((I_r, I_c))

    print(f"Initializing CipherSystem(n={n}, k={k})...")
    cipher = CipherSystem(K_array, L, subsets, n=n, k=k, d=d, seed=42)
    print("Initialization complete.")

    # Generate Messages
    messages_np_list = [np.random.randint(0, 2, size=a, dtype=np.uint8) for _ in range(num_messages_to_test)]

    # Run Avalanche Analysis
    print(f"\nRunning Avalanche Effect Analysis for n={n}, k={k} ({num_messages_to_test} messages)...")
    start_time = time.perf_counter()
    avalanche_data = cipher.analyze_avalanche_effect_all_bits(messages_np_list)
    end_time = time.perf_counter()
    print(f"Avalanche analysis took {(end_time - start_time):.2f} seconds.")

    print("\n--- Avalanche Effect Summary ---")
    all_avg_percentages = []
    overall_avg_section_counts = np.zeros(cipher.num_rm_portions)
    num_sections = cipher.num_rm_portions

    for bit_pos in range(cipher.alpha):
        if bit_pos in avalanche_data:
            data = avalanche_data[bit_pos]
            print(f"Input Bit {bit_pos}: Avg HD = {data['avg_h_distance']:.2f} ({data['avg_avalanche_perc']:.2f}%)")
            all_avg_percentages.append(data['avg_avalanche_perc'])

            section_counts_str = ", ".join([f"{c:.2f}" for c in data['avg_section_counts']])
            section_perc_str = ", ".join([f"{p:.1f}%" for p in data['avg_section_percentages']])
            print(f"  Avg Flips per Section ({num_sections} sections): [{section_counts_str}]")
            overall_avg_section_counts += np.array(data['avg_section_counts'])

    if all_avg_percentages:
        print(f"\nOverall Average Avalanche Percentage across all input bits: {np.mean(all_avg_percentages):.2f}%")

        overall_avg_section_counts /= cipher.alpha
        total_overall_avg_flips = np.sum(overall_avg_section_counts)
        overall_section_perc = (overall_avg_section_counts / total_overall_avg_flips * 100) if total_overall_avg_flips > 0 else np.zeros(num_sections)
        overall_counts_str = ", ".join([f"{c:.2f}" for c in overall_avg_section_counts])
        overall_perc_str = ", ".join([f"{p:.1f}%" for p in overall_section_perc])
        ideal_perc = 100 / num_sections
        print(f"\nOverall Average Distribution of Flips Across Sections (Ideal: {ideal_perc:.1f}% per section):")
        print(f"  Counts: [{overall_counts_str}]")
        print(f"  Percentages: [{overall_perc_str}]")

        ideal_hd = cipher.output_length / 2.0
        sac_met_counts = 0
        total_flips_tested = 0
        for bit_pos in range(cipher.alpha):
            if bit_pos in avalanche_data and 'individual_h_distances' in avalanche_data[bit_pos]:
                hds = avalanche_data[bit_pos]['individual_h_distances']
                if hds:
                    total_flips_tested += len(hds)
                    sac_met_counts += sum(1 for hd in hds if abs(hd - ideal_hd) < 1e-9)

        if total_flips_tested > 0:
            print(f"\nStrict Avalanche Criterion (SAC - exactly 50% change) met in {sac_met_counts}/{total_flips_tested} ({ (sac_met_counts/total_flips_tested)*100 :.2f}%) individual flips.")
        else:
            print("\nSAC check could not be performed.")

    # Run Differential Pattern Analysis
    print(f"\nRunning Differential Pattern Analysis for n={n}, k={k} ({num_messages_to_test} messages)...")
    start_time = time.perf_counter()
    all_diff_patterns = cipher.analyze_all_bits_differential_pattern(messages_np_list)
    pattern_consistency = cipher.detect_pattern_consistency(all_diff_patterns)
    end_time = time.perf_counter()
    print(f"Differential pattern analysis took {(end_time - start_time):.2f} seconds.")

    print("\n--- Differential Pattern Consistency Summary ---")
    consistent_bits = 0
    for bit_pos in range(cipher.alpha):
        if bit_pos in pattern_consistency:
            stats = pattern_consistency[bit_pos]
            if stats['most_common_pattern'] is not None:
                consistent_bits += 1
                print(f"Input Bit {bit_pos}: Found a common pattern!")
                print(f"  Unique Patterns: {stats['unique_patterns']} / {stats['total_patterns']}")
                pattern_str = "".join(map(str, stats['most_common_pattern']))
                print(f"  Most Common Pattern Snippet: {pattern_str[:30]}...")
                print(f"  Frequency: {stats['frequency']:.2f}")

    print(f"\nFound common differential patterns for {consistent_bits}/{cipher.alpha} input bit flips across {num_messages_to_test} messages.")