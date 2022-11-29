from utils.lcs import lcs_similarity


def get_labels(dialogue):
    """Get all different labels used in annotations"""

    labels = set()
    for dialogue in [x, y]:
        for turn in dialogue:
            for annotation in turn:
                labels.update(annotation.keys())

    return sorted(list(labels))


def get_annotations_per_type(dialogue, labels=None):
    """Get sequences of annotations organized by granularity and type"""

    if labels is None:
        labels = get_labels(dialogue)

    new_dialogue = {l: [] for l in labels}

    for turn in dialogue:
        turn_annotations = {l: [] for l in labels}

        for annotation in turn:
            for k, v in annotation.items():
                turn_annotations[k].append(v)

        for k, v in turn_annotations.items():
            new_dialogue[k].append(v)

    return new_dialogue


def compare(x, y, labels=None):
    # Make up sequences of annotations per label
    x = get_annotations_per_type(x, labels)
    y = get_annotations_per_type(y, labels)

    return None


if __name__ == "__main__":
    x = [
        [
            {
                "domain": "Restaurant",
                "act": "Inform",
                "slot": "area",
                "value": "centre",
            },
            {
                "domain": "Restaurant",
                "act": "Inform",
                "slot": "food",
                "value": "chinese",
            },
        ],
        [
            {
                "domain": "Restaurant",
                "act": "Inform",
                "slot": "pricerange",
                "value": "all price ranges",
            },
            {
                "domain": "Restaurant",
                "act": "Request",
                "slot": "pricerange",
                "value": "?",
            },
        ],
    ]
    y = [
        [{"domain": "Hotel", "act": "Inform"}],
        [
            {"domain": "Hotel", "act": "Inform", "slot": "area", "value": "Cambridge"},
            {
                "domain": "Hotel",
                "act": "Inform",
                "slot": "choice",
                "value": "a variety",
            },
            {"domain": "Hotel", "act": "Request", "slot": "pricerange", "value": "?"},
            {"domain": "general", "act": "greet"},
        ],
    ]

    result = compare(x, y)
    print(f"Similarity: {result}")
