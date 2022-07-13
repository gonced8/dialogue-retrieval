import math


def lcs_aux(X, Y, m, n):
    if m == 0 or n == 0:
        return 0
    elif X[m - 1] == Y[n - 1]:
        return 1 + lcs_aux(X, Y, m - 1, n - 1)
    else:
        return max(lcs_aux(X, Y, m, n - 1), lcs_aux(X, Y, m - 1, n))


def lcs_distance(X, Y):
    m = len(X)
    n = len(Y)
    return m + n - 2 * lcs_aux(X, Y, m, n)


def lcs_similarity(X, Y):
    m = len(X)
    n = len(Y)
    return lcs_aux(X, Y, m, n) / math.sqrt(m * n)


if __name__ == "__main__":
    import streamlit as st

    X = st.text_input("X", "AAXXXBBCCCD")
    Y = st.text_input("Y", "AABBCCC")

    st.metric("LCS similarity", lcs_similarity(X, Y))
