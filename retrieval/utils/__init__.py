def none_or_str(value):
    if value == "None":
        return None
    return value


def parse_rouge_score(result, t="f1", digits=4):
    if t == "precision":
        return {k: round(v.mid.precision, digits) for k, v in result.items()}
    elif t == "recall":
        return {k: round(v.mid.recall, digits) for k, v in result.items()}
    else:
        return {k: round(v.mid.fmeasure, digits) for k, v in result.items()}
