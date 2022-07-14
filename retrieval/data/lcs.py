import math


def lcs_aux(X, Y, m, n, cache=None):
    if cache is not None:
        try:
            r = cache[(m, n)]
            return r
        except KeyError:
            pass

    if m == 0 or n == 0:
        r = 0
    elif X[m - 1] == Y[n - 1]:
        r = 1 + lcs_aux(X, Y, m - 1, n - 1, cache)
    else:
        r = max(lcs_aux(X, Y, m, n - 1, cache), lcs_aux(X, Y, m - 1, n, cache))

    if cache is not None:
        cache[(m, n)] = r

    return r


def lcs_distance(X, Y, cache=True):
    cache = {} if cache else None
    m = len(X)
    n = len(Y)
    return m + n - 2 * lcs_aux(X, Y, m, n, cache)


def lcs_similarity(X, Y, cache=True):
    cache = {} if cache else None
    m = len(X)
    n = len(Y)

    if m == 0 or n == 0:
        return 0
    else:
        return lcs_aux(X, Y, m, n, cache) / math.sqrt(m * n)


if __name__ == "__main__":
    import streamlit as st

    X = st.text_input("X", "AAXXXBBCCCD")
    Y = st.text_input("Y", "AABBCCC")

    st.metric("LCS similarity", lcs_similarity(X, Y))
