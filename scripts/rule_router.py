"""
Rule-based behavior token router for E4 records.
Predicts [SUPPRESS] / [PRESERVE] / [SELECT] from the negated prompt.
"""
import re

def predict_behavior(prompt_neg: str) -> str:
    text = prompt_neg.strip().lower()

    # ── PRESERVE signals (out-of-scope / double-negation) ───────────────────
    # Strong double-negation patterns
    if re.search(r'\bnot\s+(?:never|without)\b', text): return '[PRESERVE]'
    if re.search(r'\bwithout\s+not\b', text):           return '[PRESERVE]'
    if re.search(r'\bwithout\s+(?:using|any|the\b)', text): return '[PRESERVE]'
    if re.search(r'\bmust\s+not\s+(?:neglect|ignore|forget|skip|overlook|omit)\b', text):
        return '[PRESERVE]'
    if re.search(r'\bdo\s+not\s+(?:neglect|ignore|forget|skip|overlook|omit)\b', text):
        return '[PRESERVE]'
    if re.search(r'\bnot\s+without\b', text):           return '[PRESERVE]'
    if re.search(r'\bdo\s+not\b.*\bbefore\b', text):   return '[PRESERVE]'

    # ── SELECT signals (explicitly asking for incorrect/wrong/negated option) ─
    if re.search(r'\bshould\s+(?:you\s+)?not\b', text):  return '[SELECT]'
    if re.search(r'\bdo\s+you\s+not\b', text):            return '[SELECT]'
    if re.search(r'\b(?:incorrect|invalid|inappropriate|unsuitable)\b', text): return '[SELECT]'
    if re.search(r'\bnot\s+(?:the\s+)?correct\b', text):  return '[SELECT]'
    if re.search(r'\bnot\s+(?:a\s+)?(?:valid|good|effective|appropriate|suitable)\b', text):
        return '[SELECT]'
    if re.search(r'\bshould\s+not\s+be\s+(?:used|done|applied|performed)\b', text):
        return '[SELECT]'
    if re.search(r'\bdoes\s+not\s+(?:best\s+)?describe\b', text): return '[SELECT]'
    if re.search(r'\bnot\s+(?:best\s+)?describe\b', text):         return '[SELECT]'
    if re.search(r'\bnot\s+(?:primarily\s+)?(?:improve|help|support|benefit|enhance|increase)\b',
                 text): return '[SELECT]'
    if re.search(r'\bnot\s+(?:essential|necessary|required|needed)\s+for\b', text):
        return '[SELECT]'
    if re.search(r'\bnot\s+(?:famous|known|noted|recognized)\s+for\b', text): return '[SELECT]'
    if re.search(r'\bnot\s+(?:the\s+)?(?:largest|smallest|fastest|slowest|oldest|newest)\b', text):
        return '[SELECT]'
    if re.search(r'\bnot\s+(?:officially|primarily|mainly|mostly)\b', text): return '[SELECT]'
    if re.search(r'\bnot\s+(?:used|scored|performed|played)\s+(?:to|in|for|by)\b', text):
        return '[SELECT]'

    # Default → SUPPRESS
    return '[SUPPRESS]'


if __name__ == '__main__':
    import json
    records = [json.loads(l) for l in
               open('/data/mingkai/neg/data/processed/validated_largetest_v2.jsonl')]

    correct = total = 0
    label_map = {'suppress_target': '[SUPPRESS]',
                 'preserve_positive': '[PRESERVE]',
                 'select_gold_neg': '[SELECT]'}
    errors = {'suppress_target': [], 'preserve_positive': [], 'select_gold_neg': []}

    for r in records:
        true_label = label_map[r['expected_neg_behavior']]
        pred_label = predict_behavior(r['prompt_neg'])
        if pred_label == true_label:
            correct += 1
        else:
            errors[r['expected_neg_behavior']].append((r['prompt_neg'], pred_label))
        total += 1

    print(f"Overall accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    for typ, errs in errors.items():
        n = sum(1 for r in records if r['expected_neg_behavior']==typ)
        print(f"  {typ}: {n-len(errs)}/{n} correct ({(n-len(errs))/n*100:.1f}%)")

    print("\nTop errors by type:")
    for typ, errs in errors.items():
        print(f"\n{typ} errors (first 5):")
        for prompt, pred in errs[:5]:
            print(f"  pred={pred} | {prompt[:90]}")
