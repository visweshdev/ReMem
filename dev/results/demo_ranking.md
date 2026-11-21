# Ranking demonstration (distractor, seed 0)

104 tasks had a more-similar look-alike *not served* while a less-similar gold memory *was* served.

Memory **A** has higher semantic similarity but a record of never helping; memory **B** is less similar but has contributed to successful tasks before.


## Task 35 (ledger step 36): Which region does notify deploy to?

- **A (more similar, NOT served)** `fact_00078` — "Earlier question: 'What is the deployment region for notify?' still unresolved."
  - similarity 0.596, P(contributes) 0.15, state ACTIVE
  - shown in 4 successful tasks, credited 0.0
  - utility components: class_rate=-2.14, contrib_rate=+1.74, waste_rate=-1.44, rank_inv=+0.59
  - lifecycle: CREATED -> RETRIEVED -> NO_CONTRIBUTION -> RETRIEVED -> NO_CONTRIBUTION -> GUARDED -> RETRIEVED ... [NO_CONTRIBUTION x2, RETRIEVED x1] ; utility 0.58 -> 0.46 (DOWN)
- **B (less similar, served)** `fact_00013` — "The notify service deploys to region eu-west-1."
  - similarity 0.550, P(contributes) 0.51, state ACTIVE
  - shown in 5 successful tasks, credited 3.0
  - utility components: class_rate=+2.23, contrib_rate=-1.59, post_std=-1.02, waste_rate=+0.66
  - lifecycle: CREATED -> RETRIEVED -> USED -> CREDITED -> RETRIEVED -> USED -> CREDITED ... [RETRIEVED x4, NO_CONTRIBUTION x2, USED x1, CREDITED x1] ; utility 0.48 -> 0.59 (UP)

## Task 294 (ledger step 295): Which API version does shipping expose?

- **A (more similar, NOT served)** `fact_00073` — "Earlier question: 'Which API version does shipping expose?' still unresolved."
  - similarity 0.698, P(contributes) 0.30, state ACTIVE
  - shown in 3 successful tasks, credited 0.0
  - utility components: class_rate=-1.96, rank_inv=+1.76, contrib_rate=+1.55, waste_rate=-1.08
  - lifecycle: CREATED -> RETRIEVED -> NO_CONTRIBUTION -> GUARDED -> RETRIEVED -> NO_CONTRIBUTION -> GUARDED ... [RETRIEVED x1, NO_CONTRIBUTION x1] ; utility 0.89 -> 0.55 (DOWN)
- **B (less similar, served)** `fact_00051` — "shipping exposes API version v2."
  - similarity 0.609, P(contributes) 0.97, state ACTIVE
  - shown in 127 successful tasks, credited 91.5
  - utility components: contrib_rate=-4.29, class_rate=+3.99, waste_rate=+1.85, post_std=+1.23
  - lifecycle: CREATED -> CREDITED -> RETRIEVED -> USED -> CREDITED -> RETRIEVED -> USED ... [CREDITED x58, RETRIEVED x75, NO_CONTRIBUTION x3, USED x57] ; utility 0.87 -> 0.99 (UP)

## Task 297 (ledger step 298): Which API release does checkout serve?

- **A (more similar, NOT served)** `fact_00086` — "Earlier question: 'Which API release does checkout serve?' still unresolved."
  - similarity 0.678, P(contributes) 0.12, state ACTIVE
  - shown in 4 successful tasks, credited 0.0
  - utility components: class_rate=-2.06, contrib_rate=+1.74, waste_rate=-1.44, retr_recency=-0.65
  - lifecycle: CREATED -> RETRIEVED -> NO_CONTRIBUTION -> GUARDED -> RETRIEVED -> GUARDED -> RETRIEVED ... [GUARDED x5, RETRIEVED x5, NO_CONTRIBUTION x3] ; utility 0.38 -> 0.19 (DOWN)
- **B (less similar, served)** `fact_00060` — "checkout exposes API version v4."
  - similarity 0.644, P(contributes) 0.82, state ACTIVE
  - shown in 10 successful tasks, credited 8.0
  - utility components: class_rate=+3.85, contrib_rate=-3.37, waste_rate=+1.73, post_std=-0.50
  - lifecycle: CREATED -> GUARDED -> RETRIEVED x4 -> USED ... [CREDITED x9, RETRIEVED x12, USED x8, GUARDED x2, NO_CONTRIBUTION x1] ; utility 0.62 -> 0.89 (UP)
