import itertools
import numpy as np
import time
import math
from reedmuller.reedmuller import ReedMuller
import binascii
from collections import Counter
import psutil
import os
import sys

# --- Helper function for popcount ---
def popcount(n):
    """Counts the number of set bits (1s) in a non-negative integer."""
    count = 0
    while n > 0:
        n &= (n - 1)
        count += 1
    return count

# --- Helper function to convert numpy array to int ---
def np_array_to_int(arr):
    """Converts a numpy array of 0s and 1s to a Python integer."""
    val = 0
    for bit in arr:
        val = (val << 1) | int(bit)
    return val

# --- Helper function to generate masks with specific weight ---
def generate_masks(length, weight):
    """Generates all integer masks of a given length and Hamming weight."""
    if weight > length or weight < 0:
        return
    indices = range(length)
    for positions in itertools.combinations(indices, weight):
        mask = 0
        for pos in positions:
            mask |= (1 << (length - 1 - pos))
        yield mask


# --- CipherSystem Class Definition ---
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
        self.num_digits = self.k
        self.bits_per_digit = self.log_n
        self.num_rm_portions = self.log_n
        self.codeword_length = self.n
        self.output_length = self.n * self.log_n

        # Key and Latin Square Setup
        if not isinstance(K_array, np.ndarray) or K_array.dtype != np.uint8 or len(K_array) != self.alpha:
            raise ValueError(f"K must be a NumPy uint8 array of length {self.alpha}")
        self.K = K_array

        np.random.seed(seed)
        L_shuffled = L.copy()
        row_perm = np.random.permutation(self.n)
        col_perm = np.random.permutation(self.n)
        self.L = L_shuffled[row_perm][:, col_perm]

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
            # sys.stderr.write(f"Error initializing ReedMuller(r={self.r}, m={self.log_n}): {e}\n")
            raise

        # Permutation Precomputation
        self.subsets = subsets
        np.random.seed(seed)
        self.rho_list_np = self._generate_rho_permutations_np()
        self.rho_list = [p.tolist() for p in self.rho_list_np] # List version for compatibility

        self.pi_list, self.key_digits_list = self._precompute_pi_and_keyschedule()


    def determine_r_from_k(self):
        m = self.log_n
        current_k_sum = 0
        for r_candidate in range(m + 1):
            current_k_sum += math.comb(m, r_candidate)
            if current_k_sum >= self.k:
                return r_candidate
        return m

    # NumPy-based Helper Functions (Simplified)
    def _binary_array_to_baseN(self, bin_array):
        reshaped = bin_array.reshape((self.num_digits, self.bits_per_digit))
        powers = 2**np.arange(self.bits_per_digit, dtype=np.uint64)
        digits = np.dot(reshaped.astype(np.uint64), powers)
        return digits.tolist()

    def _baseN_to_binary_array(self, digits):
        digits_array = np.array(digits, dtype=int)
        powers = np.arange(self.bits_per_digit)
        bin_array_reshaped = ((digits_array[:, None] >> powers) & 1).astype(np.uint8)
        bin_array = bin_array_reshaped.flatten()
        if len(bin_array) != self.alpha:
            bin_array = np.pad(bin_array, (0, self.alpha - len(bin_array)), 'constant', constant_values=0).astype(np.uint8)[:self.alpha]
        return bin_array

    def _apply_permutation_np(self, vector_array, perm_indices):
        perm_indices_np = np.array(perm_indices, dtype=int)
        permuted_array = np.zeros_like(vector_array)
        permuted_array[perm_indices_np] = vector_array
        return permuted_array

    def _inverse_permutation_np(self, perm_indices):
        perm_indices_np = np.array(perm_indices, dtype=int)
        inverse = np.zeros_like(perm_indices_np, dtype=int)
        inverse[perm_indices_np] = np.arange(len(perm_indices_np), dtype=int)
        return inverse

    def _generate_rho_permutations_np(self):
        perms = []
        num_rho_needed = self.num_rm_portions
        for z in range(num_rho_needed):
            subset_index = z % len(self.subsets)
            I_r, I_c = self.subsets[subset_index]
            current_perm_indices = np.arange(self.n, dtype=int)
            for i in I_r:
                sigma_i = np.argsort(self.L[i])
                current_perm_indices = current_perm_indices[sigma_i]
            for j in I_c:
                tau_j = np.argsort(self.L[:, j])
                current_perm_indices = current_perm_indices[tau_j]
            perms.append(current_perm_indices)
        return perms


    def _generate_pi_permutation(self, z, key_digits_round):
        rho_index = (z // 2) % self.num_rm_portions
        if rho_index >= len(self.rho_list_np): raise IndexError(f"rho_index {rho_index} out of bounds for rho_list_np")
        rho_z = self.rho_list_np[rho_index]

        key_digits_round_arr = np.array(key_digits_round, dtype=int)
        v_values = np.zeros(self.alpha, dtype=int)
        indices_n = np.arange(self.n, dtype=int)
        key_indices_n = indices_n % self.num_digits
        rho_vals_n = rho_z[indices_n % len(rho_z)]
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
        key_digits_for_sort = key_digits_round_arr[key_indices_alpha_full]
        tuples_to_sort = np.array(list(zip(v_values, key_digits_for_sort, sort_key_modifier, indices_alpha)), dtype=dtype)

        sorted_array = np.sort(tuples_to_sort, order=['v', 'd', 'mod'])

        temp_pi = np.zeros(self.alpha, dtype=int)
        temp_pi[sorted_array['orig_idx']] = np.arange(self.alpha, dtype=int)

        shift_amount = int(sum_d + z) % self.alpha
        final_pi = np.roll(temp_pi, -shift_amount)

        return final_pi


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
                    D_digits_z = self.L[prev % self.n, prev_prev % self.n].tolist()
                key_digits_list.append(D_digits_z)

        except Exception as e:
            # sys.stderr.write(f"Error during key schedule generation at round z={z if 'z' in locals() else 'unknown'}: {e}\n")
            raise

        pi_list = [self._generate_pi_permutation(z, key_digits_list[z]) for z in range(self.num_rounds)]

        return pi_list, key_digits_list


    # Core SPN Functions (Required for initialization)
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
        return C_array

    def decrypt(self, C_array):
        C_portions = np.split(C_array, self.num_rm_portions)
        inverse_rho_perms = [self._inverse_permutation_np(self.rho_list_np[i]) for i in range(self.num_rm_portions)]
        decoded_portions = []
        for i, c_portion_np in enumerate(C_portions):
            r_np = self._apply_permutation_np(c_portion_np, inverse_rho_perms[i])
            y = self.rm.decode(r_np)
            y_final = np.zeros(self.k, dtype=np.uint8)
            if y is not None and len(y) == self.k: y_final = y.astype(np.uint8)
            decoded_portions.append(y_final)
        Y_array = np.concatenate(decoded_portions)
        if len(Y_array) != self.alpha: Y_array = np.pad(Y_array, (0, self.alpha - len(Y_array)), 'constant', constant_values=0).astype(np.uint8)[:self.alpha]
        M_array = self.inverse_multi_round_mapping(Y_array)
        return M_array


def run_memory_test(scheme_params):
    # Setup Parameters
    a = scheme_params['a']
    n = scheme_params['n']
    k = scheme_params['k']
    d = scheme_params['d']

    # Generate static key components needed for initialization
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

    # Memory Measurement and Initialization
    process = psutil.Process(os.getpid())

    # Get baseline memory
    mem_before_kib = process.memory_info().rss / 1024

    print(f"Initializing CipherSystem(n={n}, k={k})...")
    start_init = time.perf_counter()

    # Initialize ALICE
    alice = CipherSystem(K_array, L, subsets, n=n, k=k, d=d, seed=42)

    # Initialize BOB
    bob = CipherSystem(K_array, L, subsets, n=n, k=k, d=d, seed=42)

    end_init = time.perf_counter()
    init_time = (end_init - start_init)

    # Get memory after init
    mem_after_kib = process.memory_info().rss / 1024
    mem_increment_kib = mem_after_kib - mem_before_kib

    # Output Results
    print("Initialization complete.")
    print(f"Memory before init: {mem_before_kib:.1f} KiB")
    print(f"Memory after init:  {mem_after_kib:.1f} KiB")
    print(f"Memory increment (Alice+Bob): {mem_increment_kib:.1f} KiB")
    print(f"Initialization time: {init_time:.2f} seconds.")

    return {'mem_kib': mem_increment_kib, 'params': scheme_params}


# MAIN EXECUTION
if __name__ == "__main__":

    # Define all parameter sets to test
    params_n32_k26 = {'a': 130, 'n': 32, 'k': 26, 'd': 4}
    params_n32_k16 = {'a': 80, 'n': 32, 'k': 16, 'd': 8}
    params_n16_k15 = {'a': 60, 'n': 16, 'k': 15, 'd': 2}
    params_n8_k4 = {'a': 12, 'n': 8, 'k': 4, 'd': 4}

    all_results = []

    print("--- STARTING MEMORY FOOTPRINT TEST ---")
    all_results.append(run_memory_test(params_n32_k26))
    print("\n" + "="*50 + "\n")

    all_results.append(run_memory_test(params_n32_k16))
    print("\n" + "="*50 + "\n")

    all_results.append(run_memory_test(params_n16_k15))
    print("\n" + "="*50 + "\n")

    all_results.append(run_memory_test(params_n8_k4))

    print("\n\n" + "="*50)
    print("FINAL MEMORY FOOTPRINT SUMMARY (ALICE + BOB)")
    print("="*50)

    for res in all_results:
        p = res['params']
        print(f"RM({p['n']},{p['k']}): (alpha={p['a']} bits)")
        print(f"  Memory Increment: {res['mem_kib']:.1f} KiB")