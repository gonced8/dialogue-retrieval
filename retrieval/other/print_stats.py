"""Prints the number of dialogue acts, slots, and values for both, user, and agent.
"""

from utils import get_sequences

if __name__ == "__main__":
    filename_data = "../../data/multiwoz/processed/dev.json"

    # Both
    print("#########\n# {: <5} #\n#########\n".format("Both"))
    sequences = get_sequences(filename_data, speaker="both", debug=True)

    # User
    print("#########\n# {: <5} #\n#########\n".format("User"))
    sequences = get_sequences(filename_data, speaker="user", debug=True)

    # Agent
    print("#########\n# {: <5} #\n#########\n".format("Agent"))
    sequences = get_sequences(filename_data, speaker="agent", debug=True)
