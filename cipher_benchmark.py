import itertools
import numpy as np
import time
import math
from reedmuller.reedmuller import ReedMuller
import binascii
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from collections import Counter # For pattern consistency check
import psutil # For memory measurement
import os     # For memory measurement
import sys    # For error messages (optional)
import tracemalloc # For memory allocation tracking


# Helper function for popcount
def popcount(n):
    """Counts the number of set bits (1s) in a non-negative integer."""
    count = 0
    while n > 0:
        n &= (n - 1)
        count += 1
    return count

# Helper function to convert numpy array to int
def np_array_to_int(arr):
    """Converts a numpy array of 0s and 1s to a Python integer."""
    val = 0
    for bit in arr:
        val = (val << 1) | int(bit)
    return val

# Helper function to generate masks with specific weight
def generate_masks(length, weight):
    """Generates all integer masks of a given length and Hamming weight."""
    if weight > length or weight < 0:
        return
    indices = range(length)
    for positions in itertools.combinations(indices, weight):
        mask = 0
        for pos in positions:
            mask |= (1 << (length - 1 - pos)) # Assumes MSB is index 0
        yield mask


# PRESENT CIPHER IMPLEMENTATION
# (Based on standard specification, e.g., https://asecuritysite.com/encryption/present)
class Present:
    def __init__(self, key, rounds=32):
        self.rounds = rounds
        if len(key) * 8 == 80:
            self.roundkeys = generateRoundkeys80(string2number(key), self.rounds)
        elif len(key) * 8 == 128:
            self.roundkeys = generateRoundkeys128(string2number(key), self.rounds)
        else:
            raise ValueError("Key must be a 128-bit or 80-bit rawstring")

    def encrypt(self, block):
        state = string2number(block)
        for i in range(self.rounds - 1):
            state = addRoundKey(state, self.roundkeys[i])
            state = sBoxLayer(state)
            state = pLayer(state)
        cipher = addRoundKey(state, self.roundkeys[-1])
        return number2string_N(cipher, 8) # PRESENT block size is 8 bytes (64 bits)

    def decrypt(self, block):
        state = string2number(block)
        for i in range(self.rounds - 1):
            state = addRoundKey(state, self.roundkeys[-i - 1])
            state = pLayer_dec(state)
            state = sBoxLayer_dec(state)
        decipher = addRoundKey(state, self.roundkeys[0])
        return number2string_N(decipher, 8) # PRESENT block size is 8 bytes (64 bits)

    def get_block_size(self):
        return 8 # 64 bits

# PRESENT helper functions
Sbox = [0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2]
Sbox_inv = [Sbox.index(x) for x in range(16)]
PBox = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
        4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
        8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
        12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]
PBox_inv = [PBox.index(x) for x in range(64)]

def generateRoundkeys80(key, rounds):
    roundkeys = []
    for i in range(1, rounds + 1):
        roundkeys.append(key >> 16)
        key = ((key & (2 ** 19 - 1)) << 61) + (key >> 19)
        key = (Sbox[key >> 76] << 76) + (key & (2 ** 76 - 1))
        key ^= i << 15
    return roundkeys

def generateRoundkeys128(key, rounds):
    roundkeys = []
    for i in range(1, rounds + 1):
        roundkeys.append(key >> 64)
        key = ((key & (2 ** 61 - 1)) << 67) + (key >> 61) # Corrected rotation
        key = (Sbox[key >> 124] << 124) + (Sbox[(key >> 120) & 0xF] << 120) + (key & ((1 << 120) - 1)) # Corrected SBox
        key ^= i << 62
    return roundkeys

def addRoundKey(state, roundkey):
    return state ^ roundkey

def sBoxLayer(state):
    output = 0
    for i in range(16):
        output += Sbox[(state >> (i * 4)) & 0xF] << (i * 4)
    return output

def sBoxLayer_dec(state):
    output = 0
    for i in range(16):
        output += Sbox_inv[(state >> (i * 4)) & 0xF] << (i * 4)
    return output

def pLayer(state):
    output = 0
    for i in range(64):
        output += ((state >> i) & 0x01) << PBox[i]
    return output

def pLayer_dec(state):
    output = 0
    for i in range(64):
        output += ((state >> i) & 0x01) << PBox_inv[i]
    return output

def string2number(i):
    if isinstance(i, str): i = i.encode()
    return int.from_bytes(i, byteorder='big')

def number2string_N(i, N):
    s = '%0*x' % (N * 2, i)
    if len(s) % 2: s = '0' + s
    return binascii.unhexlify(s)


class CipherSystem:
    def __init__(self, K_array, L, subsets, n=32, k=26, d=None, seed=42):
        self.n = n
        self.log_n = int(np.log2(n)) if n > 0 and (n & (n - 1)) == 0 else None
        if self.log_n is None: raise ValueError("n must be a power of 2")
        self.k = k
        self.alpha = self.log_n * self.k
        self.num_rounds = 2 * self.log_n

        self.num_digits = self.k
        self.bits_per_digit = self.log_n
        self.num_rm_portions = self.log_n
        self.codeword_length = self.n
        self.output_length = self.n * self.log_n

        if not isinstance(K_array, np.ndarray) or K_array.dtype != np.uint8 or len(K_array) != self.alpha:
            raise ValueError(f"K must be a NumPy uint8 array of length {self.alpha}")
        self.K = K_array

        np.random.seed(seed)
        L_shuffled = L.copy()
        row_perm = np.random.permutation(self.n)
        col_perm = np.random.permutation(self.n)
        self.L = L_shuffled[row_perm][:, col_perm]

        self.L_conj = np.zeros((self.n, self.n), dtype=int)
        for r_idx in range(self.n):
            for c_idx in range(self.n):
                symbol = self.L[r_idx, c_idx]
                self.L_conj[symbol, r_idx] = c_idx

        self.r = self.determine_r_from_k()
        self.d = d if d is not None else 2**(self.log_n - self.r) if 0 <= self.r < self.log_n else 1
        try:
            self.rm = ReedMuller(self.r, self.log_n)
        except Exception as e:
            raise

        self.subsets = subsets
        np.random.seed(seed)
        self.rho_list_np = self._generate_rho_permutations_np()
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

    def _binary_array_to_baseN(self, bin_array):
        if len(bin_array) != self.alpha: raise ValueError(f"Input binary array length must be {self.alpha}, got {len(bin_array)}")
        reshaped = bin_array.reshape((self.num_digits, self.bits_per_digit))
        powers = 2**np.arange(self.bits_per_digit, dtype=np.uint64)
        digits = np.dot(reshaped.astype(np.uint64), powers)
        return digits.tolist()

    def _baseN_to_binary_array(self, digits):
        if len(digits) != self.num_digits: raise ValueError(f"Input digits list length must be {self.num_digits}, got {len(digits)}")
        digits_array = np.array(digits, dtype=int)
        powers = np.arange(self.bits_per_digit)
        bin_array_reshaped = ((digits_array[:, None] >> powers) & 1).astype(np.uint8)
        bin_array = bin_array_reshaped.flatten()
        if len(bin_array) != self.alpha:
            bin_array = np.pad(bin_array, (0, self.alpha - len(bin_array)), 'constant', constant_values=0).astype(np.uint8)[:self.alpha]
        return bin_array

    def _apply_permutation_np(self, vector_array, perm_indices):
        if len(vector_array) != len(perm_indices): raise ValueError(f"Vector length {len(vector_array)} and permutation length {len(perm_indices)} must match.")
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
        if len(self.subsets) < num_rho_needed:
            pass

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
        except Exception:
            raise

        pi_list = [self._generate_pi_permutation(z, key_digits_list[z]) for z in range(self.num_rounds)]
        return pi_list, key_digits_list

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

    def encrypt_timed(self, M_array):
        start_spn = time.perf_counter_ns()
        Y_array = self.multi_round_mapping(M_array)
        end_spn = time.perf_counter_ns()
        spn_time_ms = (end_spn - start_spn) / 1_000_000

        start_rm_perm = time.perf_counter_ns()
        Y_portions = np.split(Y_array, self.num_rm_portions)

        codewords = []
        node_times = []
        for i, portion_np in enumerate(Y_portions):
            node_start = time.perf_counter_ns()
            cw = self.rm.encode(portion_np)
            rho_perm_indices = self.rho_list_np[i]
            permuted_cw = self._apply_permutation_np(cw, rho_perm_indices)
            node_end = time.perf_counter_ns()
            codewords.append(permuted_cw)
            node_times.append((node_end - node_start) / 1_000_000)

        C_array = np.concatenate(codewords)
        end_rm_perm = time.perf_counter_ns()

        rm_perm_time_ms = (end_rm_perm - start_rm_perm) / 1_000_000
        total_time_ms = spn_time_ms + rm_perm_time_ms
        max_node_time_ms = max(node_times) if node_times else 0

        return C_array, total_time_ms, spn_time_ms, max_node_time_ms

    def decrypt_timed(self, C_array):
        start_rm_perm_dec = time.perf_counter_ns()
        if len(C_array) != self.output_length: raise ValueError(f"Input ciphertext C length mismatch")
        C_portions = np.split(C_array, self.num_rm_portions)
        inverse_rho_perms = [self._inverse_permutation_np(self.rho_list_np[i]) for i in range(self.num_rm_portions)]

        decoded_portions = []
        node_times = []
        for i, c_portion_np in enumerate(C_portions):
            node_start = time.perf_counter_ns()
            r_np = self._apply_permutation_np(c_portion_np, inverse_rho_perms[i])
            y = self.rm.decode(r_np)

            y_final = np.zeros(self.k, dtype=np.uint8)
            if y is not None and isinstance(y, np.ndarray) and len(y) == self.k:
                y_final = y.astype(np.uint8)
            elif y is not None and isinstance(y, list) and len(y) == self.k:
                y_final = np.array(y, dtype=np.uint8)
            decoded_portions.append(y_final)
            node_end = time.perf_counter_ns()
            node_times.append((node_end - node_start) / 1_000_000)

        end_rm_perm_dec = time.perf_counter_ns()
        rm_perm_dec_time_ms = (end_rm_perm_dec - start_rm_perm_dec) / 1_000_000
        max_node_time_ms = max(node_times) if node_times else 0

        Y_array = np.concatenate(decoded_portions)
        if len(Y_array) != self.alpha:
            Y_array = np.pad(Y_array, (0, self.alpha - len(Y_array)), 'constant', constant_values=0).astype(np.uint8)[:self.alpha]

        start_spn_inv = time.perf_counter_ns()
        M_array = self.inverse_multi_round_mapping(Y_array)
        end_spn_inv = time.perf_counter_ns()
        spn_inv_time_ms = (end_spn_inv - start_spn_inv) / 1_000_000

        total_time_ms = rm_perm_dec_time_ms + spn_inv_time_ms

        return M_array, total_time_ms, spn_inv_time_ms, max_node_time_ms

    def encrypt_no_rm_timed(self, M_array):
        start_spn = time.perf_counter_ns()
        Y_array = self.multi_round_mapping(M_array)
        end_spn = time.perf_counter_ns()
        spn_time_ms = (end_spn - start_spn) / 1_000_000
        return Y_array, spn_time_ms

    def decrypt_no_rm_timed(self, Y_array):
        start_spn_inv = time.perf_counter_ns()
        M_array = self.inverse_multi_round_mapping(Y_array)
        end_spn_inv = time.perf_counter_ns()
        spn_inv_time_ms = (end_spn_inv - start_spn_inv) / 1_000_000
        return M_array, spn_inv_time_ms


# AES HELPER FUNCTIONS
def aes_encrypt(mode, plaintext_bytes, key_bytes, iv_bytes=None):
    cipher = None
    if mode == 'ECB':
        cipher = AES.new(key_bytes, AES.MODE_ECB)
        if len(plaintext_bytes) == AES.block_size:
            start_time_ns = time.perf_counter_ns()
            ciphertext = cipher.encrypt(plaintext_bytes)
            end_time_ns = time.perf_counter_ns()
            return ciphertext, (end_time_ns - start_time_ns) / 1_000_000
        else:
            padded_plaintext = pad(plaintext_bytes, AES.block_size)
            start_time_ns = time.perf_counter_ns()
            ciphertext = cipher.encrypt(padded_plaintext)
            end_time_ns = time.perf_counter_ns()
            return ciphertext, (end_time_ns - start_time_ns) / 1_000_000
    elif mode == 'CBC':
        if iv_bytes is None: raise ValueError("IV required for CBC mode")
        cipher = AES.new(key_bytes, AES.MODE_CBC, iv_bytes)
    elif mode == 'CTR':
        if iv_bytes is None: raise ValueError("Nonce (from IV) required for CTR mode")
        cipher = AES.new(key_bytes, AES.MODE_CTR, nonce=iv_bytes[:8])
    else: raise ValueError("Unsupported AES mode")

    if len(plaintext_bytes) % AES.block_size != 0:
        plaintext_bytes = pad(plaintext_bytes, AES.block_size)

    start_time_ns = time.perf_counter_ns()
    ciphertext = cipher.encrypt(plaintext_bytes)
    end_time_ns = time.perf_counter_ns()
    return ciphertext, (end_time_ns - start_time_ns) / 1_000_000

def aes_decrypt(mode, ciphertext_bytes, key_bytes, iv_bytes=None):
    cipher = None
    plaintext_bytes = b''
    if mode == 'ECB':
        cipher = AES.new(key_bytes, AES.MODE_ECB)
        start_time_ns = time.perf_counter_ns()
        padded_plaintext = cipher.decrypt(ciphertext_bytes)
        end_time_ns = time.perf_counter_ns()
        try:
            if len(padded_plaintext) > 0 and padded_plaintext[-1] <= AES.block_size:
                plaintext_bytes = unpad(padded_plaintext, AES.block_size)
            else:
                plaintext_bytes = padded_plaintext
        except ValueError:
            plaintext_bytes = padded_plaintext
        return plaintext_bytes, (end_time_ns - start_time_ns) / 1_000_000
    elif mode == 'CBC':
        if iv_bytes is None: raise ValueError("IV required for CBC mode")
        cipher = AES.new(key_bytes, AES.MODE_CBC, iv_bytes)
    elif mode == 'CTR':
        if iv_bytes is None: raise ValueError("Nonce (from IV) required for CTR mode")
        cipher = AES.new(key_bytes, AES.MODE_CTR, nonce=iv_bytes[:8])
    else: raise ValueError("Unsupported AES mode")

    start_time_ns = time.perf_counter_ns()
    plaintext_bytes = cipher.decrypt(ciphertext_bytes)
    end_time_ns = time.perf_counter_ns()

    if mode == 'CBC':
        try:
            if len(plaintext_bytes) > 0 and plaintext_bytes[-1] <= AES.block_size:
                plaintext_bytes = unpad(plaintext_bytes, AES.block_size)
        except ValueError:
            pass

    return plaintext_bytes, (end_time_ns - start_time_ns) / 1_000_000


# BENCHMARKING HARNESS
def run_benchmarks(num_messages, scheme_params, include_aes=True):
    a = scheme_params['a']
    n = scheme_params['n']
    k = scheme_params['k']
    d = scheme_params['d']

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
    alice = CipherSystem(K_array, L, subsets, n=n, k=k, d=d, seed=42)
    bob = CipherSystem(K_array, L, subsets, n=n, k=k, d=d, seed=42)
    print("Initialization complete.")

    K_128_binary_str = "".join(map(str, K_array[:128])).ljust(128, '0')
    present_key_bytes = int(K_128_binary_str, 2).to_bytes(16, 'big')
    try:
        present_cipher = Present(present_key_bytes)
    except ValueError as e:
        print(f"Error initializing PRESENT: {e}")
        present_cipher = None

    aes_key_bytes = present_key_bytes
    aes_iv_bytes = get_random_bytes(16)

    messages_cs = [np.random.randint(0, 2, size=a, dtype=np.uint8) for _ in range(num_messages)]
    messages_present_np = [m[:64] for m in messages_cs]
    messages_aes_np = [m[:128] for m in messages_cs]

    print(f"\nRunning benchmarks for {num_messages} messages...")

    cs_enc_times, cs_dec_times = [], []
    cs_enc_parallel_times, cs_dec_parallel_times = [], []
    cs_no_rm_enc_times, cs_no_rm_dec_times = [], []
    present_enc_times, present_dec_times = [], []
    aes_ecb_enc_times, aes_ecb_dec_times = [], []
    aes_cbc_enc_times, aes_cbc_dec_times = [], []
    aes_ctr_enc_times, aes_ctr_dec_times = [], []

    correct_full, correct_no_rm = True, True
    correct_present, correct_aes_ecb, correct_aes_cbc, correct_aes_ctr = True, True, True, True

    for i in range(num_messages):
        M_cs = messages_cs[i]
        C_cs, enc_total, enc_spn, enc_max_node = alice.encrypt_timed(M_cs)
        M_dec_cs, dec_total, dec_spn, dec_max_node = bob.decrypt_timed(C_cs)
        cs_enc_times.append(enc_total); cs_dec_times.append(dec_total)
        cs_enc_parallel_times.append(enc_spn + enc_max_node); cs_dec_parallel_times.append(dec_spn + dec_max_node)
        if not np.array_equal(M_cs, M_dec_cs): correct_full = False

        Y_cs, no_rm_enc_time = alice.encrypt_no_rm_timed(M_cs)
        M_no_rm_cs, no_rm_dec_time = bob.decrypt_no_rm_timed(Y_cs)
        cs_no_rm_enc_times.append(no_rm_enc_time); cs_no_rm_dec_times.append(no_rm_dec_time)
        if not np.array_equal(M_cs, M_no_rm_cs): correct_no_rm = False

        if present_cipher:
            M_present_np = messages_present_np[i]
            M_present_str = "".join(map(str, M_present_np))
            plain_bytes_present = int(M_present_str, 2).to_bytes(8, 'big')
            start_time_ns = time.perf_counter_ns(); C_bytes_present = present_cipher.encrypt(plain_bytes_present); present_enc_times.append((time.perf_counter_ns() - start_time_ns) / 1_000_000)
            start_time_ns = time.perf_counter_ns(); M_dec_bytes_present = present_cipher.decrypt(C_bytes_present); present_dec_times.append((time.perf_counter_ns() - start_time_ns) / 1_000_000)
            if plain_bytes_present != M_dec_bytes_present: correct_present = False

        if include_aes:
            M_aes_np = messages_aes_np[i]
            M_aes_str = "".join(map(str, M_aes_np))
            plain_bytes_aes = int(M_aes_str, 2).to_bytes(16, 'big')

            C_aes_ecb, t_enc_ecb = aes_encrypt('ECB', plain_bytes_aes, aes_key_bytes)
            M_dec_aes_ecb_bytes, t_dec_ecb = aes_decrypt('ECB', C_aes_ecb, aes_key_bytes)
            aes_ecb_enc_times.append(t_enc_ecb); aes_ecb_dec_times.append(t_dec_ecb)
            if plain_bytes_aes != M_dec_aes_ecb_bytes: correct_aes_ecb = False

            C_aes_cbc, t_enc_cbc = aes_encrypt('CBC', plain_bytes_aes, aes_key_bytes, aes_iv_bytes)
            M_dec_aes_cbc_bytes, t_dec_cbc = aes_decrypt('CBC', C_aes_cbc, aes_key_bytes, aes_iv_bytes)
            aes_cbc_enc_times.append(t_enc_cbc); aes_cbc_dec_times.append(t_dec_cbc)
            if plain_bytes_aes != M_dec_aes_cbc_bytes: correct_aes_cbc = False

            C_aes_ctr, t_enc_ctr = aes_encrypt('CTR', plain_bytes_aes, aes_key_bytes, aes_iv_bytes)
            M_dec_aes_ctr_bytes, t_dec_ctr = aes_decrypt('CTR', C_aes_ctr, aes_key_bytes, aes_iv_bytes)
            aes_ctr_enc_times.append(t_enc_ctr); aes_ctr_dec_times.append(t_dec_ctr)
            if plain_bytes_aes != M_dec_aes_ctr_bytes: correct_aes_ctr = False

        print(f"Processed message {i+1}/{num_messages}")

    print("\n--- Average Benchmark Results ---")
    print(f"Scheme: Proposed RM({n},{k}) (Block: {a}-bit)")
    print(f"  Correctness (Full Scheme): {correct_full}")
    print(f"  Correctness (SPN-Only): {correct_no_rm}")
    print(f"  Avg. SPN-Only Encrypt (Baseline): {np.mean(cs_no_rm_enc_times):.3f} ms")
    print(f"  Avg. SPN-Only Decrypt (Baseline): {np.mean(cs_no_rm_dec_times):.3f} ms")
    print(f"  Avg. Full Scheme Encrypt (Single Node): {np.mean(cs_enc_times):.3f} ms")
    print(f"  Avg. Full Scheme Decrypt (Single Node): {np.mean(cs_dec_times):.3f} ms")
    print(f"  Avg. Full Scheme Encrypt (Parallel Est.): {np.mean(cs_enc_parallel_times):.3f} ms")
    print(f"  Avg. Full Scheme Decrypt (Parallel Est.): {np.mean(cs_dec_parallel_times):.3f} ms")

    if present_cipher:
        print("\nScheme: PRESENT-128 (Block: 64-bit)")
        print(f"  Correctness: {correct_present}")
        print(f"  Avg. Encryption: {np.mean(present_enc_times):.3f} ms")
        print(f"  Avg. Decryption: {np.mean(present_dec_times):.3f} ms")

    if include_aes:
        print("\nScheme: AES-128 (Block: 128-bit)")
        print(f"  Correctness (ECB): {correct_aes_ecb}")
        print(f"  Avg. Encryption (ECB): {np.mean(aes_ecb_enc_times):.3f} ms")
        print(f"  Avg. Decryption (ECB): {np.mean(aes_ecb_dec_times):.3f} ms")
        print(f"  Correctness (CBC): {correct_aes_cbc}")
        print(f"  Avg. Encryption (CBC): {np.mean(aes_cbc_enc_times):.3f} ms")
        print(f"  Avg. Decryption (CBC): {np.mean(aes_cbc_dec_times):.3f} ms")
        print(f"  Correctness (CTR): {correct_aes_ctr}")
        print(f"  Avg. Encryption (CTR): {np.mean(aes_ctr_enc_times):.3f} ms")
        print(f"  Avg. Decryption (CTR): {np.mean(aes_ctr_dec_times):.3f} ms")


# MAIN EXECUTION
if __name__ == "__main__":

    params_130_bit = { 'a': 130, 'n': 32, 'k': 26, 'd': 4 }
    run_benchmarks(num_messages=100, scheme_params=params_130_bit, include_aes=True)

    print("\n" + "="*50 + "\n")

    params_60_bit = { 'a': 60, 'n': 16, 'k': 15, 'd': 2 }
    run_benchmarks(num_messages=100, scheme_params=params_60_bit, include_aes=True)