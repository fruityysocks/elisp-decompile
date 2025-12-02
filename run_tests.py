#!/usr/bin/env python3
"""
Comprehensive test comparing grammar-based vs CFG-based decompiler.
Tests all .lap files and compares outputs to original .el files.
"""
import os
import sys
import subprocess
import difflib
from pathlib import Path
from collections import defaultdict

def run_decompiler(lap_file, use_cfg=False):
    """Run decompiler and return output"""
    # Use bash to activate venv and run decompiler
    flag = '--use-cfg' if use_cfg else '--use-parser'
    cmd = f'source .venv/bin/activate && python3 -m lapdecompile {flag} {lap_file}'

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, shell=True, executable='/bin/bash')
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT", -1
    except Exception as e:
        return None, str(e), -1

def normalize_whitespace(text):
    """Normalize whitespace for comparison"""
    lines = text.strip().split('\n')
    return '\n'.join(line.rstrip() for line in lines if line.strip())

def compare_outputs(original, decompiled):
    """Compare original and decompiled code, return similarity score"""
    if original is None or decompiled is None:
        return 0.0

    orig_norm = normalize_whitespace(original)
    decomp_norm = normalize_whitespace(decompiled)

    if orig_norm == decomp_norm:
        return 100.0

    # Use difflib to calculate similarity
    matcher = difflib.SequenceMatcher(None, orig_norm, decomp_norm)
    return matcher.ratio() * 100

def find_el_file(lap_file):
    """Find corresponding .el file"""
    base = lap_file.stem
    el_file = lap_file.with_suffix('.el')

    if el_file.exists():
        return el_file

    # Check for alternate names (binom-coeff -> binomial-coeff)
    parent = lap_file.parent
    for el_candidate in parent.glob('*.el'):
        if 'decompile' not in el_candidate.stem and base.replace('-', '') in el_candidate.stem.replace('-', ''):
            return el_candidate

    return None

def test_all_files(test_dirs):
    """Test all .lap files in given directories"""
    results = {
        'grammar': defaultdict(dict),
        'cfg': defaultdict(dict),
    }

    all_lap_files = []
    for test_dir in test_dirs:
        all_lap_files.extend(sorted(Path(test_dir).glob('*.lap')))

    print(f"Testing {len(all_lap_files)} .lap files...")
    print("=" * 80)

    for lap_file in all_lap_files:
        name = f"{lap_file.parent.name}/{lap_file.name}"
        print(f"\nTesting: {name}")

        # Find original .el file
        el_file = find_el_file(lap_file)
        original = None
        if el_file and el_file.exists():
            original = el_file.read_text()

        # Test grammar-based
        print("  Grammar-based...", end=' ')
        gram_out, gram_err, gram_rc = run_decompiler(lap_file, use_cfg=False)
        if gram_rc == 0 and gram_out:
            print("Success")
            results['grammar'][name]['output'] = gram_out
            results['grammar'][name]['error'] = None
            results['grammar'][name]['rc'] = 0
            if original:
                similarity = compare_outputs(original, gram_out)
                results['grammar'][name]['similarity'] = similarity
        else:
            error_msg = gram_err.split('\n')[0] if gram_err else "Unknown error"
            print(f"Failed: {error_msg[:60]}")
            results['grammar'][name]['output'] = gram_out
            results['grammar'][name]['error'] = gram_err
            results['grammar'][name]['rc'] = gram_rc
            results['grammar'][name]['similarity'] = 0.0

        # Test CFG-based
        print("  CFG-based.......", end=' ')
        cfg_out, cfg_err, cfg_rc = run_decompiler(lap_file, use_cfg=True)
        if cfg_rc == 0 and cfg_out:
            print("Success")
            results['cfg'][name]['output'] = cfg_out
            results['cfg'][name]['error'] = None
            results['cfg'][name]['rc'] = 0
            if original:
                similarity = compare_outputs(original, cfg_out)
                results['cfg'][name]['similarity'] = similarity
        else:
            error_msg = cfg_err.split('\n')[0] if cfg_err else "Unknown error"
            print(f"Failed: {error_msg[:60]}")
            results['cfg'][name]['output'] = cfg_out
            results['cfg'][name]['error'] = cfg_err
            results['cfg'][name]['rc'] = cfg_rc
            results['cfg'][name]['similarity'] = 0.0

    return results

def analyze_results(results):
    """Analyze results and generate report"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # Success rates
    gram_success = sum(1 for r in results['grammar'].values() if r['rc'] == 0)
    cfg_success = sum(1 for r in results['cfg'].values() if r['rc'] == 0)
    total = len(results['grammar'])

    print(f"\nSuccess Rates:")
    print(f"  Grammar-based: {gram_success}/{total} ({gram_success/total*100:.1f}%)")
    print(f"  CFG-based:     {cfg_success}/{total} ({cfg_success/total*100:.1f}%)")

    # Similarity scores (for successful decompilations)
    gram_similarities = [r['similarity'] for r in results['grammar'].values()
                        if r['rc'] == 0 and 'similarity' in r]
    cfg_similarities = [r['similarity'] for r in results['cfg'].values()
                       if r['rc'] == 0 and 'similarity' in r]

    if gram_similarities:
        gram_avg = sum(gram_similarities) / len(gram_similarities)
        gram_perfect = sum(1 for s in gram_similarities if s == 100.0)
        print(f"\nGrammar-based similarity:")
        print(f"  Average: {gram_avg:.1f}%")
        print(f"  Perfect matches: {gram_perfect}/{len(gram_similarities)}")

    if cfg_similarities:
        cfg_avg = sum(cfg_similarities) / len(cfg_similarities)
        cfg_perfect = sum(1 for s in cfg_similarities if s == 100.0)
        print(f"\nCFG-based similarity:")
        print(f"  Average: {cfg_avg:.1f}%")
        print(f"  Perfect matches: {cfg_perfect}/{len(cfg_similarities)}")

    # Files where both succeed
    both_succeed = []
    only_grammar = []
    only_cfg = []
    both_fail = []

    for name in results['grammar'].keys():
        gram_ok = results['grammar'][name]['rc'] == 0
        cfg_ok = results['cfg'][name]['rc'] == 0

        if gram_ok and cfg_ok:
            both_succeed.append(name)
        elif gram_ok:
            only_grammar.append(name)
        elif cfg_ok:
            only_cfg.append(name)
        else:
            both_fail.append(name)

    print(f"\nFile Categories:")
    print(f"  Both succeed: {len(both_succeed)}")
    print(f"  Only grammar: {len(only_grammar)}")
    print(f"  Only CFG:     {len(only_cfg)}")
    print(f"  Both fail:    {len(both_fail)}")

    # Show which files fall into each category
    if both_succeed:
        print(f"\nFiles where BOTH succeed ({len(both_succeed)}):")
        for name in sorted(both_succeed):
            gram_sim = results['grammar'][name].get('similarity', 0)
            cfg_sim = results['cfg'][name].get('similarity', 0)
            print(f"  {name:40} Grammar: {gram_sim:5.1f}% | CFG: {cfg_sim:5.1f}%")

    if only_grammar:
        print(f"\n⚠ Files where only GRAMMAR succeeds ({len(only_grammar)}):")
        for name in sorted(only_grammar):
            error = results['cfg'][name]['error']
            error_line = error.split('\n')[0] if error else "Unknown"
            print(f"  {name:40} CFG error: {error_line[:35]}")

    if only_cfg:
        print(f"\n⚠ Files where only CFG succeeds ({len(only_cfg)}):")
        for name in sorted(only_cfg):
            error = results['grammar'][name]['error']
            error_line = error.split('\n')[0] if error else "Unknown"
            print(f"  {name:40} Grammar error: {error_line[:35]}")

    if both_fail:
        print(f"\nFiles where BOTH fail ({len(both_fail)}):")
        for name in sorted(both_fail):
            gram_err = results['grammar'][name]['error']
            cfg_err = results['cfg'][name]['error']
            gram_line = gram_err.split('\n')[0] if gram_err else "Unknown"
            cfg_line = cfg_err.split('\n')[0] if cfg_err else "Unknown"
            print(f"  {name}")
            print(f"    Grammar: {gram_line[:70]}")
            print(f"    CFG:     {cfg_line[:70]}")

    # Common error patterns
    print(f"\n" + "=" * 80)
    print("COMMON ERROR PATTERNS")
    print("=" * 80)

    gram_errors = defaultdict(list)
    cfg_errors = defaultdict(list)

    for name, result in results['grammar'].items():
        if result['rc'] != 0 and result['error']:
            error_type = result['error'].split('\n')[0]
            gram_errors[error_type].append(name)

    for name, result in results['cfg'].items():
        if result['rc'] != 0 and result['error']:
            error_type = result['error'].split('\n')[0]
            cfg_errors[error_type].append(name)

    print("\nGrammar-based errors:")
    for error, files in sorted(gram_errors.items(), key=lambda x: -len(x[1])):
        print(f"  [{len(files)} files] {error[:70]}")

    print("\nCFG-based errors:")
    for error, files in sorted(cfg_errors.items(), key=lambda x: -len(x[1])):
        print(f"  [{len(files)} files] {error[:70]}")

def main():
    test_dirs = ['test-prish', 'test/lap']
    results = test_all_files(test_dirs)
    analyze_results(results)

    # Save detailed results
    print("\n" + "=" * 80)
    print("Saving detailed results to comprehensive_test_results.txt")

    with open('comprehensive_test_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for name in sorted(results['grammar'].keys()):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"FILE: {name}\n")
            f.write('=' * 80 + "\n\n")

            f.write("GRAMMAR-BASED:\n")
            f.write("-" * 80 + "\n")
            gram = results['grammar'][name]
            f.write(f"Return code: {gram['rc']}\n")
            if 'similarity' in gram:
                f.write(f"Similarity: {gram['similarity']:.1f}%\n")
            if gram['error']:
                f.write(f"Error:\n{gram['error']}\n")
            if gram['output']:
                f.write(f"Output:\n{gram['output']}\n")

            f.write("\nCFG-BASED:\n")
            f.write("-" * 80 + "\n")
            cfg = results['cfg'][name]
            f.write(f"Return code: {cfg['rc']}\n")
            if 'similarity' in cfg:
                f.write(f"Similarity: {cfg['similarity']:.1f}%\n")
            if cfg['error']:
                f.write(f"Error:\n{cfg['error']}\n")
            if cfg['output']:
                f.write(f"Output:\n{cfg['output']}\n")

if __name__ == '__main__':
    main()
