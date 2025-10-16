"""
Parser e extração estruturada de campos da coluna 'commit'.

Este módulo implementa a extração de:
- issue_ids (ex: IKSWQ-787)
- actions (verbos canônicos)
- apis (CamelCase/snake_case methods)
- modules (componentes snake_case)
- packages (Java/Android packages)
- build_flags (USE_*, *TOF*)
- errors/events (SYSTEM_RESTART, Exception, etc.)
- contagens (n_msgs, n_apis, etc.)

Autor: Filo-Priori V5
Data: 2025-10-15
"""

import ast
import re
from typing import Dict, List, Set
import pandas as pd
import numpy as np


# ============================================================================
# CONSTANTS & PATTERNS
# ============================================================================

# Canonical action verbs (stem forms)
ACTIONS = {
    'support', 'implement', 'update', 'enable', 'add', 'fix', 'deprecate',
    'disable', 'remove', 'refactor', 'clean', 'revert', 'optimize', 'rename',
    'bump', 'merge', 'improve', 'resolve', 'correct', 'enhance', 'introduce'
}

# Regex patterns
RE_ISSUE = re.compile(r'\b[A-Z][A-Z0-9]+-\d+\b')
RE_API = re.compile(r'\b(get|set|enable|disable|start|stop|add|remove|update|create|delete|init|load|save|read|write|open|close|send|receive|handle|process|check|validate|parse|build|destroy)[A-Za-z0-9_]+\b')
RE_CONST = re.compile(r'\b[A-Z][A-Z0-9_]{3,}\b')
RE_MODULE = re.compile(r'\b[a-z0-9]+(?:_[a-z0-9]+){1,}\b')
RE_PACKAGE = re.compile(r'\b(?:[a-zA-Z_][\w\-]*\.)+[a-zA-Z_][\w\-]*\b')
RE_FLAG = re.compile(r'\bUSE_[A-Z0-9_]+\b|[A-Z][A-Z0-9_]*TOF[A-Z0-9_]*|FEATURE_[A-Z0-9_]+|CONFIG_[A-Z0-9_]+')
RE_ERROR = re.compile(r'\b(null pointer|npe|nullpointerexception|exception|error|crash|system_restart|terminated|timeout|failure|failed|abort|hang)\b', re.I)

# Error normalization map
ERROR_SYNONYMS = {
    'null pointer': 'NullPointerException',
    'npe': 'NullPointerException',
    'system_restart': 'SYSTEM_RESTART',
    'terminated': 'TERMINATED',
    'timeout': 'TIMEOUT',
    'crash': 'CRASH',
    'hang': 'HANG',
    'abort': 'ABORT',
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_commit_list(cell) -> List[str]:
    """
    Parse commit column string to list of messages.

    Args:
        cell: Raw cell value (string representation of list)

    Returns:
        List of commit message strings
    """
    if pd.isna(cell) or not cell or str(cell).strip() in ['[]', '{}', '']:
        return []

    try:
        if isinstance(cell, str):
            parsed = ast.literal_eval(cell)
            if isinstance(parsed, list):
                return [str(msg) for msg in parsed if msg]
        elif isinstance(cell, list):
            return [str(msg) for msg in cell if msg]
    except (ValueError, SyntaxError):
        # Fallback: treat as single message
        return [str(cell)]

    return []


def normalize_error(error_str: str) -> str:
    """Normalize error string to canonical form."""
    lower = error_str.lower().strip()
    return ERROR_SYNONYMS.get(lower, error_str.capitalize())


def split_camel_snake(token: str) -> List[str]:
    """
    Split CamelCase or snake_case tokens.

    Examples:
        'getCdmaSidNidPairs' -> ['getCdmaSidNidPairs']
        'a/b/c' -> ['a', 'b', 'c']
    """
    parts = []

    # Split by /
    for part in re.split(r'[/\\]', token):
        if part:
            parts.append(part.strip())

    return parts


# ============================================================================
# CORE EXTRACTION FUNCTION
# ============================================================================

def extract_commit_fields(messages: List[str]) -> Dict:
    """
    Extract structured fields from list of commit messages.

    Args:
        messages: List of commit message strings

    Returns:
        Dictionary with extracted fields:
        - commit_issue_ids
        - commit_actions
        - commit_apis
        - commit_modules
        - commit_packages
        - commit_build_flags
        - commit_errors
        - commit_n_msgs
        - commit_n_apis
        - commit_n_issues
        - commit_n_modules
        - commit_n_packages
        - commit_n_flags
        - commit_n_errors
    """
    issues: Set[str] = set()
    actions: Set[str] = set()
    apis: Set[str] = set()
    modules: Set[str] = set()
    packages: Set[str] = set()
    flags: Set[str] = set()
    errors: Set[str] = set()

    full_text = ' '.join(messages)

    for msg in messages:
        # Issue IDs
        issues.update(RE_ISSUE.findall(msg))

        # Actions (verbs)
        for verb in ACTIONS:
            # Match verb with common suffixes: ed, es, ing, s
            if re.search(rf'\b{verb}(?:ed|es|ing|s)?\b', msg, re.I):
                actions.add(verb)

        # APIs
        api_matches = RE_API.findall(msg)
        for api in api_matches:
            # Split tokens by /
            for token in split_camel_snake(api):
                if len(token) >= 6:  # Filter short matches like 'get'
                    apis.add(token)

        # Modules (snake_case)
        module_matches = RE_MODULE.findall(msg)
        for mod in module_matches:
            if len(mod) > 4 and '_' in mod:  # Must have underscore and reasonable length
                modules.add(mod)

        # Packages
        pkg_matches = RE_PACKAGE.findall(msg)
        for pkg in pkg_matches:
            if pkg.count('.') >= 2:  # At least 2 dots (e.g., com.android.chrome)
                packages.add(pkg)

        # Flags
        flag_matches = RE_FLAG.findall(msg)
        flags.update(flag_matches)

        # Errors
        error_matches = RE_ERROR.findall(msg)
        for err in error_matches:
            normalized = normalize_error(err)
            errors.add(normalized)

    # Move UPPERCASE constants with underscores from modules to flags
    constants = {c for c in RE_CONST.findall(full_text) if '_' in c and len(c) > 4}
    flags.update(constants)
    modules -= constants  # Remove from modules if they ended up there

    # Clean and sort
    issues = sorted(issues)
    actions = sorted(actions)
    apis = sorted(apis)
    modules = sorted(modules)
    packages = sorted(packages)
    flags = sorted(flags)
    errors = sorted(errors)

    return {
        'commit_issue_ids': issues,
        'commit_actions': actions,
        'commit_apis': apis,
        'commit_modules': modules,
        'commit_packages': packages,
        'commit_build_flags': flags,
        'commit_errors': errors,
        'commit_n_msgs': len(messages),
        'commit_n_apis': len(apis),
        'commit_n_issues': len(issues),
        'commit_n_modules': len(modules),
        'commit_n_packages': len(packages),
        'commit_n_flags': len(flags),
        'commit_n_errors': len(errors),
    }


def make_commit_text(fields: Dict, max_items_per_field: int = 20) -> str:
    """
    Create canonical commit text from extracted fields.

    Args:
        fields: Dictionary from extract_commit_fields()
        max_items_per_field: Maximum items to include per field

    Returns:
        Formatted commit text string
    """
    parts = []

    def add_section(tag: str, items: List[str], max_items: int = max_items_per_field):
        if items:
            # Limit items and join
            limited = items[:max_items]
            joined = "; ".join(limited)
            parts.append(f"[{tag}] {joined}.")

    # Priority order: Errors > Actions > APIs > Modules > Flags > Packages > Issues
    add_section("Errors", fields['commit_errors'], max_items=10)
    add_section("Actions", fields['commit_actions'], max_items=15)
    add_section("APIs", fields['commit_apis'], max_items=20)
    add_section("Modules", fields['commit_modules'], max_items=15)
    add_section("Flags", fields['commit_build_flags'], max_items=15)
    add_section("Packages", fields['commit_packages'], max_items=10)
    add_section("Commit Issues", fields['commit_issue_ids'], max_items=10)

    return "\n".join(parts) if parts else "No commit info."


# ============================================================================
# DATAFRAME PROCESSING
# ============================================================================

def process_commits(df: pd.DataFrame, commit_column: str = 'commit') -> pd.DataFrame:
    """
    Process commit column in dataframe and add structured fields.

    Args:
        df: Input dataframe
        commit_column: Name of commit column

    Returns:
        Dataframe with new commit_* columns added
    """
    print(f"Processing {len(df)} rows...")

    # Parse commit lists
    print("Parsing commit messages...")
    commit_lists = df[commit_column].apply(parse_commit_list)

    # Extract fields
    print("Extracting structured fields...")
    extracted = commit_lists.apply(extract_commit_fields)

    # Convert to dataframe columns
    commit_df = pd.DataFrame(extracted.tolist())

    # Add string versions for categorical features
    for col in ['commit_issue_ids', 'commit_actions', 'commit_apis',
                'commit_modules', 'commit_packages', 'commit_build_flags', 'commit_errors']:
        commit_df[f'{col}_str'] = commit_df[col].apply(lambda x: ';'.join(x) if x else '')

    # Generate commit_text
    print("Generating commit_text...")
    commit_df['commit_text'] = extracted.apply(make_commit_text)

    # Concatenate with original dataframe
    result = pd.concat([df.reset_index(drop=True), commit_df], axis=1)

    # Print statistics
    print("\n" + "="*70)
    print("EXTRACTION STATISTICS")
    print("="*70)
    print(f"Total rows: {len(result)}")
    print(f"Rows with ≥1 issue_id: {(commit_df['commit_n_issues'] > 0).sum()} ({(commit_df['commit_n_issues'] > 0).mean()*100:.1f}%)")
    print(f"Rows with ≥1 API: {(commit_df['commit_n_apis'] > 0).sum()} ({(commit_df['commit_n_apis'] > 0).mean()*100:.1f}%)")
    print(f"Rows with ≥1 error: {(commit_df['commit_n_errors'] > 0).sum()} ({(commit_df['commit_n_errors'] > 0).mean()*100:.1f}%)")
    print(f"\nAverage items per row:")
    print(f"  Issues: {commit_df['commit_n_issues'].mean():.2f}")
    print(f"  APIs: {commit_df['commit_n_apis'].mean():.2f}")
    print(f"  Modules: {commit_df['commit_n_modules'].mean():.2f}")
    print(f"  Errors: {commit_df['commit_n_errors'].mean():.2f}")
    print(f"  Flags: {commit_df['commit_n_flags'].mean():.2f}")

    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 01_parse_commit.py <input_csv> [output_csv]")
        print("Example: python 01_parse_commit.py ../datasets/train.csv ../artifacts/train_parsed.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.csv', '_parsed.csv')

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Process
    result = process_commits(df, commit_column='commit')

    # Save
    print(f"\nSaving to {output_path}...")
    result.to_csv(output_path, index=False)

    print(f"\nDone! Output shape: {result.shape}")
    print(f"New columns added: {[c for c in result.columns if c not in df.columns]}")
