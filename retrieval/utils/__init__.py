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


def compute_rouge(metric, predictions, references, mode="precision"):
    # Choose precision, recall, or f1
    if mode == "precision":
        parse = lambda x: x.precision
    elif mode == "recall":
        parse = lambda x: x.recall
    else:
        parse = lambda x: x.f1

    # Compute
    result = {
        rouge_type: [parse(x) for x in results]
        for rouge_type, results in metric.compute(
            predictions=predictions,
            references=references,
            use_aggregator=False,
        ).items()
    }

    # Return
    return result.values()
