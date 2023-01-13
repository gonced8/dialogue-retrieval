from sentence_transformers.util import cos_sim, pairwise_dot_score
import torch


def heuristic_score(heuristic_fn, answers, device):
    scores = []

    for i, reference in enumerate(answers[:-1]):
        for j, hypothesis in enumerate(answers[i + 1 :]):
            scores.append(heuristic_fn(hypothesis, reference))

    return torch.tensor(scores, device=device).view(-1)


def model_score(embeddings):
    n, d = embeddings.shape

    idx = torch.triu_indices(n, n, 1, device=embeddings.device)

    references_index = idx[0].view(-1, 1).expand(-1, d)
    hypothesis_index = idx[1].view(-1, 1).expand(-1, d)

    references = torch.gather(embeddings, dim=0, index=references_index)
    hypothesis = torch.gather(embeddings, dim=0, index=hypothesis_index)

    return pairwise_dot_score(references, hypothesis)


def compare_rank_scores(scores_h, scores_m):
    # Get scores to compare
    n = scores_m.size(0)
    idx = torch.triu_indices(n, n, 1, device=scores_m.device)

    scores_h_i = torch.gather(scores_h, dim=0, index=idx[0])
    scores_h_j = torch.gather(scores_h, dim=0, index=idx[1])

    scores_m_i = torch.gather(scores_m, dim=0, index=idx[0])
    scores_m_j = torch.gather(scores_m, dim=0, index=idx[1])

    return ((scores_h_i - scores_h_j) - (scores_m_i - scores_m_j)) ** 2


def compare_rank_scores_neighbors(scores_h, scores_m, b):
    idx = torch.triu_indices(b, b, 1, device=scores_m.device)

    # Put scores of each candidate pair in a matrix where each row represents a different reference
    scores_h_neighbors = torch.empty(b, b, dtype=torch.float, device=scores_h.device)
    scores_h_neighbors[idx[0], idx[1]] = scores_h
    scores_h_neighbors[idx[1], idx[0]] = scores_h

    scores_m_neighbors = torch.empty(b, b, dtype=torch.float, device=scores_m.device)
    scores_m_neighbors[idx[0], idx[1]] = scores_m
    scores_m_neighbors[idx[1], idx[0]] = scores_m

    # Remove diagonals
    scores_h_neighbors = (
        scores_h_neighbors.flatten()[1:].view(b - 1, b + 1)[:, :-1].reshape(b, b - 1)
    )
    scores_m_neighbors = (
        scores_m_neighbors.flatten()[1:].view(b - 1, b + 1)[:, :-1].reshape(b, b - 1)
    )

    # Sort candidates according to heuristic
    scores_h_neighbors, indices = torch.sort(scores_h_neighbors, dim=1)
    scores_m_neighbors = scores_m_neighbors.gather(1, indices)

    # Compute margins between neighbors
    scores_h_margins = torch.diff(scores_h_neighbors)
    scores_m_margins = torch.diff(scores_m_neighbors)

    return (scores_h_margins - scores_m_margins) ** 2


def compare_rank_scores_heuristic(scores_h, scores_m):
    # Get scores to compare
    n = scores_m.size(0)
    idx = torch.triu_indices(n, n, 1, device=scores_m.device)

    scores_h_i = torch.gather(scores_h, dim=0, index=idx[0])
    scores_h_j = torch.gather(scores_h, dim=0, index=idx[1])

    scores_m_i = torch.gather(scores_m, dim=0, index=idx[0])
    scores_m_j = torch.gather(scores_m, dim=0, index=idx[1])

    # Compute difference in scores
    diff_scores_h = scores_h_i - scores_h_j
    diff_scores_m = scores_m_i - scores_m_j

    torch.sign(diff_scores_h, out=diff_scores_h)

    return torch.nn.functional.relu(-diff_scores_h * diff_scores_m)


class M3SE(torch.nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct=cos_sim):
        """From Luís Borges
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(M3SE, self).__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct

    def forward(
        self,
        query_vectors,
        pos_vectors,
        neg_vectors,
        teacher_query,
        teacher_pos,
        teacher_neg,
    ):

        b_size = query_vectors.size(0)

        # --------------- STUDENT --------------- #
        scores = self.similarity_fct(query_vectors, pos_vectors) * self.scale

        pos_scores = torch.diagonal(scores).unsqueeze(1)  # Scores pos: (batch_size, 1)
        neg_scores_inbatch = scores.masked_select(
            ~torch.eye(b_size, dtype=bool).cuda()
        ).view(
            b_size, b_size - 1
        )  # (batch_size, batch_size-1)

        scores = torch.cat((pos_scores, neg_scores_inbatch), dim=1)

        scores_neg = self.similarity_fct(query_vectors, neg_vectors) * self.scale
        scores_neg = torch.diagonal(scores_neg).unsqueeze(1)
        scores = torch.cat((scores, scores_neg), dim=1)
        # --------------------------------------- #

        # --------------- TEACHER --------------- #
        teacher_scores = self.similarity_fct(teacher_query, teacher_pos) * self.scale

        teacher_pos_scores = torch.diagonal(teacher_scores).unsqueeze(
            1
        )  # Scores pos: (batch_size, 1)
        teacher_neg_scores_inbatch = teacher_scores.masked_select(
            ~torch.eye(b_size, dtype=bool).cuda()
        ).view(
            b_size, b_size - 1
        )  # (batch_size, batch_size-1)

        teacher_scores = torch.cat(
            (teacher_pos_scores, teacher_neg_scores_inbatch), dim=1
        )

        teacher_scores_neg = (
            self.similarity_fct(teacher_query, teacher_neg) * self.scale
        )
        teacher_scores_neg = torch.diagonal(teacher_scores_neg).unsqueeze(1)
        teacher_scores = torch.cat((teacher_scores, teacher_scores_neg), dim=1)
        # --------------------------------------- #

        j = torch.argmax(teacher_scores[:, 1:], dim=1).reshape(-1, 1)
        t_j_star = torch.gather(teacher_scores[:, 1:], 1, j)
        s_j_star = torch.gather(scores[:, 1:], 1, j)

        P_teacher = teacher_scores[:, 0]
        P_student = scores[:, 0]
        N_teacher = teacher_scores[:, 1:]
        N_student = scores[:, 1:]

        aux1 = P_teacher - t_j_star
        aux2 = P_student - s_j_star
        aux3 = (aux1 - aux2) ** 2
        aux4 = torch.sum(aux3, dim=1)

        aux5 = N_student - s_j_star
        aux6 = torch.relu(aux5) ** 2
        aux7 = torch.sum(aux6, dim=1)

        final = aux4 + aux7
        return torch.mean(final)
