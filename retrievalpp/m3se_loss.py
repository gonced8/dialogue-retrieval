"""From Luís Borges"""
import torch
from sentence_transformers.util import cos_sim


class M3SE(torch.nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct=cos_sim):
        """
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
