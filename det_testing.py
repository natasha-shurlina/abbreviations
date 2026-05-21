import json
import argparse
from pathlib import Path

from det import (
    Rules, DeterministicConfig, DeterministicChecker,
    strip_markdown, enrich_with_llm
)


def run_on_file(input_path: Path, checker: DeterministicChecker) -> dict:
    raw_text = input_path.read_text(encoding="utf-8")
    clean_text = strip_markdown(raw_text)
    findings = checker.check_text(clean_text)
    findings_dicts = enrich_with_llm(findings, clean_text, None)

    errors = [f for f in findings_dicts if f["verdict"] == "ОШИБКА"]
    skipped = [f for f in findings_dicts if f["verdict"] == "ПРОПУСК"]
    ok_list = [f for f in findings_dicts if f["verdict"] == "OK"]
    allowed_used = sorted({f["fragment"] for f in ok_list})

    return {
        "mode": "det",
        "input": str(input_path),
        "errors": errors,
        "skipped_count": len(skipped),
        "allowed_found": allowed_used,
    }


def load_gt(gt_path: Path) -> list:
    lines = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def compare(found_errors: list, expected: list) -> dict:
    found_fragments = [e["fragment"].strip().lower() for e in found_errors]
    expected_lower = [e.strip().lower() for e in expected]

    tp = sum(1 for f in found_fragments if f in expected_lower)
    fp = sum(1 for f in found_fragments if f not in expected_lower)
    fn = sum(1 for e in expected_lower if e not in found_fragments)

    total = tp + fp + fn if (tp + fp + fn) > 0 else 1

    return {
        "true_positive": round(tp / total, 3),
        "false_negative": round(fn / total, 3),
        "false_positive": round(fp / total, 3),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Пакетная проверка сокращений по папке с файлами"
    )
    parser.add_argument("--input-dir", "-i", required=True,
                        help="Папка с входными .txt файлами")
    parser.add_argument("--rules", "-r", required=True,
                        help="Путь к rules.yaml")
    parser.add_argument("--gt", action="store_true",
                        help="Сравнивать с эталоном из подпапки gt/")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Ошибка: {input_dir} не является папкой")
        return

    results_dir = input_dir / "results"
    results_dir.mkdir(exist_ok=True)

    gt_dir = input_dir / "gt" if args.gt else None

    rules = Rules.from_yaml(args.rules)
    det_cfg = DeterministicConfig(threshold=2, short_word_min_len=3)
    checker = DeterministicChecker(rules, det_cfg)

    input_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".txt", ".md")
    )

    if not input_files:
        print(f"Файлы .txt в папке {input_dir} не найдены")
        return

    for input_path in input_files:
        print(f"Обрабатываю: {input_path.name}")

        result = run_on_file(input_path, checker)

        if gt_dir is not None:
            gt_path = gt_dir / input_path.name
            if gt_path.exists():
                expected = load_gt(gt_path)
                cmp = compare(result["errors"], expected)
                result["comparison"] = cmp
                print(f"ошибок найдено: {len(result['errors'])}, "
                      f"ожидалось: {len(expected)}, "
                      f"ttrue_positive: {cmp['true_positive']}, "
                      f"false_negative: {cmp['false_negative']}, "
                      f"false_positive: {cmp['false_positive']}")
            else:
                print(f"Эталон не найден: {gt_path}")
        else:
            print(f"Ошибок найдено: {len(result['errors'])}")

        out_path = results_dir / (input_path.stem + ".json")
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    print(f"\nГотово. Результаты в: {results_dir}")


if __name__ == "__main__":
    main()
