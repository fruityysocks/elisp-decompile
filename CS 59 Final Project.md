# COSC 59 Final Project: Emacs Lisp Decompiler

**GitHub Link:** https://github.com/fruityysocks/elisp-decompile.git

**Credit Statement:** This project enhanced the elisp-decompiler (https://github.com/rocky/elisp-decompile) originally created by GitHub user @rocky. I also used Claude Code to help figure out the logic and debug the codebase. 

---

## ELISP Decompiler: 

I found this decompiler on github and thought it would be fun to contribute. The original plan was to build on top of the existing decompiler by handing more complex macros, but I quickly realised that the decompiler failed to handle simple `if-else` programs (more on that below). The original decompiler relied on pattern matching to an extensive grammar with many, many (~100) rules and that felt like an incredibly in-elegant solution to me. It is also not easily scalable. So, I thought it would be interesting to improve the logic by implementing a Control Flow Graph (CFG) based approach that recognizes patterns through program structure rather than text matching,

The original grammar-based decompiler had several bugs for simple if-else cases. For example: 
```
byte code for simple-if:
  doc:   ...
  args: (arg1)
0	dup	  
1	constant  0
2	gtr	  
3	goto-if-nil 1
6	constant  true
7	return	  
8:1	constant  false
9	return	  

```

this byte-code got reconstructed to the following: 
```
(defun simple-test(arg1)
"   ..."
  (if (> DUP-empty-stack 0)
      (cond 
            ( 'true)ru
            (t 'false)als))
)
```

Which, obviously, is wrong. So I implemented the following fixes. 

---

###  Bug Fixes in Grammar-Based Approach
There were 3 bugs: 
1. Reconstructing `DUP-empty-stack` instead of `arg1`: The `SourceWalker` class (responsible for code generation) wasn't initializing the evaluation stack with function arguments. When the decompiler encountered a `DUP` instruction (which duplicates the top stack value), it had an empty stack and showed a placeholder `DUP-empty-stack`.

2. Wrapping `cond` in an `if` block: The parser recognized the bytecode pattern as an `cond_form` and the `if` as a expr. So, adding `n_if_form` fixed it.  

3. Weird constant reconstruction: 
The `n_clause()` method in [lapdecompile/semantics.py](lapdecompile/semantics.py) was writing values twice:
1. Once during template processing
2. Again in manual code at line 381
And the manual code did `val[1:-1]` which removed the first and last characters from "true" → "ru" and "false" → "als".



## CFG-Based Decompilation Implementation

The main issue with the grammar-based implementation is that it is not scalable. Every new pattern needs new rules, (and although there are only finitely many rules -- this is a tedious and vexing process). It essentially is brittle and hard to maintain. 
CFG helps because it recognizes patterns using flow structure and not text matching, and hence is a lot more scalable. 

### Architecture

```
Old Grammar-Based Approach:
Bytecode → Tokens → Parser (grammar rules) → AST → Transform (fixes) → Code

New CFG-Based Approach:
Bytecode → Tokens → Basic Blocks → CFG → Structure Analysis → AST → Code
```
The following diagram shows the current pipeline for the project. 
 

```
┌─────────────┐
│  LAP File   │  Emacs bytecode disassembly (text)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Scanner (scanner.py)                              │
│  • Parses LAP text format                                   │
│  • Extracts function metadata (name, args, docstring)       │
│  • Creates Token objects for each instruction               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  Token Stream  │
                  └────────┬───────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Basic Blocks (bb.py)                              │
│  • Identifies jump targets                                  │
│  • Splits at branch instructions                            │
│  • Computes stack effects                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  Basic Blocks  │
                  └────────┬───────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: CFG (cfg.py) + Dominators (dominators.py)        │
│  • Builds control flow graph                                │
│  • Connects blocks with edges                               │
│  • Computes dominator tree                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  CFG + DomTree │
                  └────────┬───────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
┌───────────────────────┐    ┌───────────────────────┐
│  GRAMMAR APPROACH     │    │  CFG APPROACH         │
│  (parser.py)          │    │  (structure.py)       │
│                       │    │                       │
│  • Earley parsing     │    │  • Pattern detection  │
│  • Token patterns     │    │  • Topology analysis  │
│  • Grammar rules      │    │  • Region structuring │
└───────┬───────────────┘    └───────┬───────────────┘
        │                            │
        │                            │
        │                            │
        │                            │
        ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: AST (Abstract Syntax Tree)                        │
│  • SyntaxTree nodes                                         │
│  • Represents code structure                                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: Transform (transform.py)                          │
│  • Optimizes AST                                            │
│  • Simplifies expressions                                   │
│  • Normalizes patterns                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 6: Code Generation (semantics.py)                    │
│  • Template engine                                          │
│  • Indentation management                                   │
│  • Format to Elisp                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  Elisp Source  │
                  └────────────────┘
```

Unfortunately, due to time constraints, neither of the implementations of the parser are robust, and the grammar-based has higher accuracy because it handles more cases than the CFG-based parser (eg: the latter does not handle do-list and complex conditionals ). I tested the code on the test files provided by @rocky and also on some of the functions I wrote for HW3; namely, ` max-list`, `sum-evens`, `mygcd`, `check-all`, `fact`, `is-relative-prime`, `binom-coeffs`, and `catalan.` 


## Test Results 


```bash
python run_tests.py
```
[run_tests.py](run_tests.py) tests 48 LAP files from `test/lap/` and `test-prish/` directories with both approaches. 
- Grammar-based decompiler (--use-parser flag)
- CFG-based decompiler (--use-cfg flag)

Then it compares outputs to the original `.el` files by normalizing whitespace, and calculating a similarity score using Python's `difflib.SequenceMatcher`, which albeit is a crude metric for analysing success but since the dissimilarities are using so glaring. It does bring to light the files and patterns that need to be addressed. 

Once run, the results of the comparison are stored in `comprehensive_test_results.txt` with full outputs and errors for each test case. 


## How to Use This? 

### Installation
```bash
git clone https://github.com/fruityysocks/elisp-decompile.git
cd elisp-decompile
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Basic Usage
```bash
# CFG approach (new)
python -m lapdecompile <directory/test-file>.lap --use-cfg

# Grammar approach (original)
python -m lapdecompile <directory/test-file>.lap --use-parser

# Compare both approaches
python -m lapdecompile <directory/test-file>.lap --use-parser > grammar.el
python -m lapdecompile <directory/test-file>.lap --use-cfg > cfg.el
diff -u grammar.el cfg.el
```

### Testing
```bash
# Run comprehensive test suite
python3 run_tests.py

# View detailed results
cat comprehensive_test_results.txt
```

### Visualization
```bash
# Generate CFG diagrams
python3 -m lapdecompile --graphs test/lap/test-if.lap
# Creates /tmp/flow-*.png files
```

---