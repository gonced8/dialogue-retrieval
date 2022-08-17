# Not very correct evaluation (assumed that there is always a correct candidate)

| Description         | MRR @ 10 | MRR @ 100 | Min-Max Normalized DCG @ 10 | Min-Max Normalized DCG @ 100 |
|---------------------|:--------:|:---------:|:---------------------------:|:----------------------------:|
| Pretrained baseline |  55.01%  |   31.42%  |            89.84%           |            92.30%            |
| Fine-tuned model    |  75.48%  |   50.61%  |            95.92%           |            96.51%            |


# Probably has errors

| Description | Learning Rate | Quantile Transformation | ROUGE1-P | ROUGE2-P | ROUGEL-P | LCS Similarity |
|:-----------:|:-------------:|:-----------------------:|:--------:|:--------:|:--------:|:--------------:|
|   Baseline  |       -       |            -            |  15.03%  |   3.39%  |  12.88%  |     26.32%     |
|  Fine-tuned |      1e-5     |            N            |  14.59%  |   2.70%  |  12.40%  |     25.43%     |
|  Fine-tuned |      1e-5     |            Y            |  15.01%  |   2.91%  |  12.91%  |     25.76%     |
|  Fine-tuned |      1e-6     |            Y            |  15.51%  |   3.14%  |  13.16%  |     26.47%     |


# Correct

| Description | Learning Rate | Quantile Transformation | ROUGE1-P | ROUGE2-P | ROUGEL-P | LCS Similarity |
|:-----------:|:-------------:|:-----------------------:|:--------:|:--------:|:--------:|:--------------:|
|   Baseline  |       -       |            -            |  14.11%  |   2.53%  |  12.01%  |     24.90%     |
|  Fine-tuned |      1e-5     |            N            |  14.87%  |   2.89%  |  12.82%  |     25.83%     |
|  Fine-tuned |      1e-5     |            Y            |  14.89%  |   2.68%  |  12.58%  |     27.22%     |
|  Fine-tuned |      1e-6     |            Y            |  15.83%  |   3.24%  |  13.72%  |     26.92%     |
