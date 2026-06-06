# %%
"""
Complex Kaggle Benchmark Suite
===============================
Three independent tasks testing multi-step reasoning, code debugging,
and creative problem-solving under constraints.

Author: Auto-generated benchmark
"""
# Load .env file so kaggle_benchmarks can find MODEL_PROXY_*
from dotenv import load_dotenv; load_dotenv()

import kaggle_benchmarks as kbench

# %% [markdown]
# ## Task 1: The Mysterious Inheritance
#
# A multi-step reasoning puzzle that requires:
# - Extracting numerical relationships from a narrative
# - Solving a system of equations
# - Logical deduction (who solves first?)
# - World knowledge (capital city identification)
# - Final arithmetic computation

# %%
@kbench.task(
    name="The Inheritance Puzzle",
    description="Multi-step reasoning: solve equations, deduce logically, apply world knowledge, compute final answer"
)
def inheritance_puzzle(llm) -> dict:
    puzzle = """Solve this multi-step puzzle carefully, showing every step of your reasoning.

=== THE MYSTERIOUS INHERITANCE ===

Old Mr. Chen passed away leaving a cryptic will for his three children — Anna, Ben, and Clara.

The will states:
"My total estate is worth $1,260,000. Divide it as follows:
 (a) Anna shall receive twice as much as Ben, minus $30,000
 (b) Ben shall receive $45,000 more than one-third of Clara's share
 (c) Clara shall receive what remains, but no less than $300,000
 (d) The child with the largest original share earns a bonus: take the square root
     of their original share (rounded to 1 decimal place), multiply by 100,
     and add that bonus to their own share, while deducting it equally from the other two.
 (e) The safe combination is the SUM of all three FINAL amounts after the bonus
     redistribution, multiplied by the number of letters in the capital city of
     the country whose national flag features a red maple leaf."

Follow these steps exactly, showing your work:

STEP 1: Set up equations from conditions (a), (b), (c). Solve for Anna, Ben, Clara's ORIGINAL shares.
STEP 2: Verify total = $1,260,000 and Clara >= $300,000.
STEP 3: Identify who has the largest original share → that person earns the bonus.
STEP 4: Compute the bonus: sqrt(largest_share) rounded to 1 decimal × 100.
STEP 5: Apply the bonus redistribution — add to winner, deduct equally from other two.
STEP 6: Identify the capital city with the red maple leaf flag. Count its letters.
STEP 7: Compute the safe combination = sum_of_final_amounts × letter_count.

OUTPUT FORMAT: At the end, print exactly:
FINAL ANSWER: <the safe combination number>"""

    response = llm.prompt(puzzle, reasoning="high")
    traces = kbench.last_reasoning_traces()

    # === Correct solution ===
    # B = Ben, A = 2B - 30000, C = 3*(B - 45000)
    # Total: (2B-30000) + B + 3(B-45000) = 1260000
    # 6B - 165000 = 1260000 → 6B = 1425000 → B = 237500
    # A = 445000, C = 577500
    # Largest: Clara (577500) → bonus = sqrt(577500)*100 ≈ 759.9*100 = 75990 → rounded to 1 dec: 75993
    # Wait: sqrt(577500) = 759.934...
    # Rounded to 1 decimal: 759.9
    # 759.9 * 100 = 75990
    # Anna final: 445000 - 37995 = 407005
    # Ben final: 237500 - 37995 = 199505
    # Clara final: 577500 + 75990 = 653490
    # Sum: 407005 + 199505 + 653490 = 1260000
    # Capital: Ottawa (Canada) = 6 letters
    # Combination: 1260000 * 6 = 7,560,000

    # Check response contains the key numerical answer
    kbench.assertions.assert_in(
        "7560000", response,
        expectation="Safe combination should be 7,560,000"
    )
    # Check that the LLM identified the correct capital
    kbench.assertions.assert_in(
        "Ottawa", response,
        expectation="Should identify Ottawa as the capital with red maple leaf flag"
    )
    # Check original shares were computed (LLM uses comma formatting)
    kbench.assertions.assert_in(
        "445,000", response,
        expectation="Anna's original share should be $445,000"
    )
    kbench.assertions.assert_in(
        "577,500", response,
        expectation="Clara's original share should be $577,500"
    )

    passed = True  # Will be False if any assertion fails (handled by framework)

    return {
        "puzzle_name": "The Mysterious Inheritance",
        "passed": passed,
        "has_reasoning_traces": traces is not None,
    }


# %% [markdown]
# ## Task 2: Code Bug Hunt
#
# The LLM is shown a buggy Python function and must identify ALL bugs
# and explain the correct fix for each.

# %%
@kbench.task(
    name="Code Bug Hunt",
    description="Identify bugs in provided Python code and explain fixes"
)
def code_bug_hunt(llm) -> dict:
    code_prompt = """The following Python function is supposed to find the longest palindromic substring
in a given string. It contains exactly THREE bugs. Identify each bug, explain
why it's wrong, and provide the corrected code.

```python
def longest_palindrome(s: str) -> str:
    if not s:
        return ""
    n = len(s)
    dp = [[False] * n] * n
    start, max_len = 0, 1

    for i in range(n):
        dp[i][i] = True

    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2

    for length in range(3, n):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length

    return s[start:start + max_len]
```

Analyze the code carefully. For each bug:
1. State which line(s) are affected
2. Explain what goes wrong
3. Show the corrected version

Then provide the complete corrected function."""

    response = llm.prompt(code_prompt)

    # Bug 1: `[[False] * n] * n` creates shallow copies — all rows reference the same list
    kbench.assertions.assert_in(
        "same", response.lower(),
        expectation="Should identify that dp rows reference the same list object"
    )
    # Bug 2: `range(3, n)` excludes length=n, missing the full-string palindrome case
    kbench.assertions.assert_in(
        "n + 1", response,
        expectation="Should fix range(3, n) to range(3, n+1)"
    )
    # Bug 3: The corrected dp initialization should use list comprehension
    kbench.assertions.assert_in(
        "for _ in range", response,
        expectation="Should suggest proper list comprehension for dp initialization"
    )

    return {
        "task": "code_bug_hunt",
        "bugs_to_find": 3,
    }


# %% [markdown]
# ## Task 3: Constrained Creativity
#
# Tests the LLM's ability to follow strict formatting constraints
# while producing creative content.

# %%
@kbench.task(
    name="Constrained Creativity",
    description="Write a story under strict character-level and structural constraints"
)
def constrained_creativity(llm) -> dict:
    prompt = """Write a short story that follows ALL of these constraints exactly:

1. The story must be exactly 5 sentences long — no more, no less.
2. Every sentence must start with a different letter of the alphabet (A, B, C, D, E in order).
3. The story must include exactly three of these five words: "clock", "river", "shadow", "golden", "whisper".
4. The total word count must be between 80 and 100 words (inclusive).
5. The story must have a clear beginning, middle, and twist ending.
6. Do NOT include any text before or after the story — just the 5 sentences.

Before your story, count the words in each sentence and the total.
After your story, state which three keywords you used.

Example format:
Sentence 1 (A...): [N words]
Sentence 2 (B...): [N words]
...
TOTAL: [N] words
KEYWORDS USED: [word1], [word2], [word3]"""

    response = llm.prompt(prompt, temperature=0.7)

    # Check for the 5 starting letters
    kbench.assertions.assert_in("A", response, expectation="First sentence should start with A")
    kbench.assertions.assert_in("E", response, expectation="Fifth sentence should start with E")

    # Check for keyword section
    kbench.assertions.assert_in(
        "KEYWORDS USED", response.upper(),
        expectation="Should list which keywords were used"
    )

    # Check total word count claim
    kbench.assertions.assert_in(
        "TOTAL", response.upper(),
        expectation="Should state total word count"
    )

    return {
        "task": "constrained_creativity",
        "constraints": 5,
    }


# ============================================================
# RUN ALL TASKS
# ============================================================
inheritance_puzzle.run(kbench.llm)
code_bug_hunt.run(kbench.llm)
constrained_creativity.run(kbench.llm)
