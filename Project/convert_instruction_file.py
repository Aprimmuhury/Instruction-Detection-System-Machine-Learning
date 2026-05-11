import ast
import random
import pandas as pd
from pathlib import Path
import argparse


def extract_lists_from_code(code: str):
    # Remove fenced code markers if present
    lines = [l for l in code.splitlines() if not l.strip().startswith('```')]
    code_clean = '\n'.join(lines)
    tree = ast.parse(code_clean)
    lists = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, (ast.List, ast.Tuple)):
                    try:
                        val = ast.literal_eval(node.value)
                        lists[target.id] = val
                    except Exception:
                        pass
    return lists


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default='instruction_detection_large.csv', help='Path to the template file')
    p.add_argument('--per-class', type=int, default=500, help='Examples per class')
    p.add_argument('--overwrite', action='store_true', help='Overwrite the input file with CSV output')
    args = p.parse_args()

    path = Path(args.file)
    code = path.read_text(encoding='utf-8', errors='ignore')
    lists = extract_lists_from_code(code)

    instr_tpl = lists.get('instruction_templates')
    not_instr_tpl = lists.get('not_instruction_templates')
    actions = lists.get('actions')

    if not instr_tpl or not not_instr_tpl or not actions:
        raise SystemExit('Could not extract required lists from the file')

    data = []
    for _ in range(args.per_class):
        t = random.choice(instr_tpl)
        a = random.choice(actions)
        data.append((t.format(a), 'instruction'))
    for _ in range(args.per_class):
        t = random.choice(not_instr_tpl)
        a = random.choice(actions)
        data.append((t.format(a), 'not instruction'))

    random.shuffle(data)
    df = pd.DataFrame(data, columns=['sentence', 'label'])

    out_path = path if args.overwrite else path.with_name(path.stem + '_converted.csv')
    df.to_csv(out_path, index=False)
    print(f'Wrote {len(df)} rows to {out_path}')


if __name__ == '__main__':
    main()
