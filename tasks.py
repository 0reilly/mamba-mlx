"""
Task generators for the Mamba experiment.

Three synthetic tasks:
1. Retrieval — find the token following the first occurrence of a query token.
2. Addition   — given pairs (a_i, b_i, =, c_i), predict the sum digit(s).
3. Dyck-1     — predict the next character in a Dyck-1 (parentheses) string.

All functions return MLX arrays (mlx.core, imported as mx). No numpy.
"""

import random
import mlx.core as mx

# ═══════════════════════════════════════════════════════════════════════════
# Vocabulary definition
# ═══════════════════════════════════════════════════════════════════════════
# Digits 0–9 occupy indices 0–9.
PLUS       = 10   # '+'
EQUALS     = 11   # '='
SPACE      = 12   # ' ' (unused in generation but reserved)
OPEN_PAREN = 13   # '('
CLOSE_PAREN = 14  # ')'
# Indices 15+ are free for retrieval token values.

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _rng(seed=None):
    """Return a seeded `random.Random` instance for reproducible generation."""
    return random.Random(seed)


def _generate_balanced_dyck(rng, length):
    """Generate a uniformly random balanced Dyck-1 word of the given *even* length.

    Uses the ballot / Catalan rejection-free algorithm: at each step choose
    '(' if it does not make completion impossible, else ')'.
    """
    n = length // 2
    remaining_open = n
    remaining_close = n
    seq = []
    for _ in range(length):
        if remaining_open > 0 and (remaining_close == 0 or
                                   rng.random() < remaining_open / (remaining_open + remaining_close)):
            seq.append(OPEN_PAREN)
            remaining_open -= 1
        else:
            seq.append(CLOSE_PAREN)
            remaining_close -= 1
    return seq


# ═══════════════════════════════════════════════════════════════════════════
# Task 1: Retrieval
# ═══════════════════════════════════════════════════════════════════════════

def generate_retrieval_batch(batch_size, seq_len, vocab_size):
    """Generate a batch of retrieval (needle-in-a-haystack) tasks.

    Each example consists of ``seq_len`` tokens:
      - positions 0 … seq_len-2 : random ints in [0, vocab_size-1]
      - position   seq_len-1     : a *query token* (also in [0, vocab_size-1])

    The model must output the token that immediately follows the **first**
    occurrence of the query token within positions 0 … seq_len-2.  If the
    query token is not found (or is the very last random token with no
    successor), the answer is 0.

    Returns
    -------
    input_ids  : mx.array, shape (batch_size, seq_len)
    target_ids : mx.array, shape (batch_size, seq_len)
        For positions 0 … seq_len-2 the target is the standard next-token
        (input[i+1]); at the final position it is the retrieval answer.
    """
    # -- Generate all random tokens at once (B, seq_len) --------------------
    # We generate seq_len tokens: the last column *is* the query token.
    tokens = mx.random.randint(0, vocab_size, (batch_size, seq_len))

    # -- Per-example answer lookup (Python loop — small batch, on CPU) ------
    answers = []
    for b in range(batch_size):
        # Extract as Python ints for the search
        row = tokens[b].tolist()
        query = row[-1]                     # last token is the query
        answer = 0                          # default if not found
        for i in range(seq_len - 2):        # search positions 0 … seq_len-3
            if row[i] == query:
                answer = row[i + 1]         # token that follows the match
                break
        answers.append(answer)

    answer_col = mx.array(answers, dtype=mx.int32)   # (batch_size,)

    # -- Build targets: shifted left by 1, last column = answer -------------
    # target[:, :-1] = input[:, 1:]
    # target[:,  -1] = answer_col
    target_ids = mx.concatenate(
        [tokens[:, 1:], answer_col[:, None]],
        axis=1,
    )

    return tokens, target_ids


# ═══════════════════════════════════════════════════════════════════════════
# Task 2: Addition
# ═══════════════════════════════════════════════════════════════════════════

def generate_addition_batch(batch_size, seq_len):
    """Generate a batch of decimal addition tasks.

    Each example is a compact encoding of repeated *pairs*::

        a₁  b₁  =  c₁  a₂  b₂  =  c₂  …

    where ``a_i`` and ``b_i`` are single decimal digits (tokens 0–9),
    ``=`` is the ``EQUALS`` token, and ``c_i`` is the digit(s) of the
    sum ``a_i + b_i`` (one or two tokens for sums 0–18).

    Example (logical values, not token IDs)::

        [4, 2, =, 6,   3, 7, =, 1, 0]   → 4+2=6, 3+7=10

    Returns
    -------
    input_ids  : mx.array, shape (batch_size, seq_len)
    target_ids : mx.array, shape (batch_size, seq_len)
        Standard next-token targets (shifted left by 1); the final position
        is ``-100`` (ignored in loss).
    """
    pad_token = -100

    inputs  = mx.full((batch_size, seq_len), pad_token, dtype=mx.int32)
    targets = mx.full((batch_size, seq_len), pad_token, dtype=mx.int32)

    for b in range(batch_size):
        seq = []
        pos = 0
        while pos < seq_len:
            # Enough room for at least a, b, =, and one sum digit?
            if pos + 4 > seq_len:
                break

            a = random.randint(0, 9)
            b_digit = random.randint(0, 9)
            s = a + b_digit                     # 0 … 18
            s_tokens = [int(d) for d in str(s)]  # one or two digits

            # Check if the full tuple fits
            needed = 3 + len(s_tokens)          # a, b, =, sum_digits
            if pos + needed > seq_len:
                break

            seq.extend([a, b_digit, EQUALS] + s_tokens)
            pos += needed

        # Pad the remaining positions with 0 (silent pad)
        final_len = len(seq)
        if final_len < seq_len:
            seq.extend([0] * (seq_len - final_len))

        # Standard next-token targets: target[i] = seq[i+1], last = -100
        tgt = seq[1:] + [pad_token]

        inputs[b]  = mx.array(seq, dtype=mx.int32)
        targets[b] = mx.array(tgt, dtype=mx.int32)

    return inputs, targets


# ═══════════════════════════════════════════════════════════════════════════
# Task 3: Dyck-1
# ═══════════════════════════════════════════════════════════════════════════

def generate_dyck_batch(batch_size, seq_len):
    """Generate a batch of Dyck-1 (balanced-parentheses) sequences.

    Each sequence is a uniformly random balanced Dyck-1 word of length
    ``seq_len`` (which should be even).  '(' is token ``OPEN_PAREN``,
    ')' is token ``CLOSE_PAREN``.

    Returns
    -------
    input_ids  : mx.array, shape (batch_size, seq_len)
        The parentheses tokens.
    target_ids : mx.array, shape (batch_size, seq_len)
        Standard next-token targets (shifted left by 1); the final position
        is ``-100`` (ignored in loss).
    """
    pad_token = -100
    rng = _rng()

    inputs  = mx.zeros((batch_size, seq_len), dtype=mx.int32)
    targets = mx.full((batch_size, seq_len), pad_token, dtype=mx.int32)

    for b in range(batch_size):
        seq = _generate_balanced_dyck(rng, seq_len)
        tgt = seq[1:] + [pad_token]

        inputs[b]  = mx.array(seq, dtype=mx.int32)
        targets[b] = mx.array(tgt, dtype=mx.int32)

    return inputs, targets
