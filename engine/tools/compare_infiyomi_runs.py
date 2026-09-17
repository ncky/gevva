#!/usr/bin/env python3
"""Compare local Infiyomi runs, including paired page-cluster quality intervals.

Does not judge, upload, modify the source runs, or decide promotion. Missing
judge coverage is reported explicitly; paired scores only use common regions.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import unicodedata


def read(path):
    return json.loads(path.read_text())


def run_metrics(root):
    summary = read(root / "summary.json")
    pages = read(root / "page_results.json")
    total_tokens = sum((a.get("usage") or {}).get("completion_tokens", 0)
                       for page in pages for a in page["attempts"])
    fields = ("pages", "ok_pages", "failed_pages", "incomplete_pages",
              "total_seconds", "pages_per_second", "time_to_first_page_seconds",
              "expected_regions", "returned_regions", "translated_regions",
              "region_return_rate", "translation_return_rate", "attempts")
    result = {key: summary.get(key) for key in fields}
    result.update(run_dir=str(root), attempted_completion_tokens=total_tokens,
                  attempted_completion_tps=total_tokens / summary["total_seconds"])
    quality_path = root / "quality/summary.json"
    if quality_path.exists():
        result["quality"] = read(quality_path)
        score_path = root / "quality/page_scores.json"
        if score_path.exists():
            scores = [score for page in read(score_path) for score in page["scores"]]
            expected = summary["expected_regions"]
            missing = max(0, expected - len(scores))
            total_score = sum(s["average"] for s in scores)
            result["quality_coverage_bounds"] = {
                "unscored_gold_regions": missing,
                "full_dataset_mean_bounds":
                    [total_score / expected, (total_score + 10 * missing) / expected]
                    if expected else None,
                "note": "Assign every unscored gold region 0 or 10. Bounds cover missing judge coverage, not judge bias or uncertainty in scored regions.",
            }
            def normalized(text):
                return "".join(unicodedata.normalize("NFKC", text).split())
            matches = sum(normalized(s.get("gold_source_text", "")) ==
                          normalized(s.get("model_source_text", "")) for s in scores)
            result["source_text_agreement"] = {
                "evaluated_regions": len(scores),
                "exact_normalized_matches": matches,
                "rate": matches / len(scores) if scores else None,
                "note": "Exact OCR source-text match after NFKC and whitespace removal, on judged regions only. Not a translation-quality score.",
            }
    return result


def paired_quality(baseline, candidate):
    if not all((p / "quality/page_scores.json").exists() for p in (baseline, candidate)):
        return {"available": False}
    config_paths = [p / "quality/judge_model_config.json" for p in (baseline, candidate)]
    same_judge_config = (all(p.exists() for p in config_paths) and
                         read(config_paths[0]) == read(config_paths[1]))
    def scores(root):
        return {p["page_id"]: {s["id"]: s for s in p["scores"]}
                for p in read(root / "quality/page_scores.json")}
    base, cand = scores(baseline), scores(candidate)
    pairs = []
    worst = []
    for page in sorted(base.keys() & cand.keys()):
        ids = base[page].keys() & cand[page].keys()
        if not ids:
            continue
        differences = [cand[page][i]["average"] - base[page][i]["average"] for i in ids]
        pairs.append((sum(differences), len(differences)))
        worst.append({"page_id": page, "matched_regions": len(ids),
                      "mean_score_delta": statistics.mean(differences)})
    if not pairs:
        return {"available": False, "reason": "no common scored regions"}
    # Resample whole pages, retaining all their regions and the benchmark's
    # region-weighted mean. Individual regions on one page are not independent.
    rng = random.Random(109)
    samples = []
    for _ in range(5000):
        draw = rng.choices(pairs, k=len(pairs))
        samples.append(sum(p[0] for p in draw) / sum(p[1] for p in draw))
    samples.sort()
    return {
        "available": True,
        "identical_judge_config_files": same_judge_config,
        "matched_pages": len(pairs),
        "matched_scored_regions": sum(p[1] for p in pairs),
        "baseline_scored_regions": sum(map(len, base.values())),
        "candidate_scored_regions": sum(map(len, cand.values())),
        "candidate_minus_baseline_mean": sum(p[0] for p in pairs) / sum(p[1] for p in pairs),
        "page_cluster_bootstrap_95_percent_interval": [samples[125], samples[4874]],
        "note": "Common scored regions only; inspect coverage and failures separately. Same fixed judge configuration required; this interval excludes judge-model bias.",
        "lowest_page_deltas": sorted(worst, key=lambda x: x["mean_score_delta"])[:10],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline, candidate = run_metrics(args.baseline), run_metrics(args.candidate)
    result = {"baseline": baseline, "candidate": candidate,
              "successful_pages_per_second_ratio": candidate["pages_per_second"] / baseline["pages_per_second"],
              "paired_quality": paired_quality(args.baseline, args.candidate)}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
