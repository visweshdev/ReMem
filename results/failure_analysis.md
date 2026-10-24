# Failure analysis (auto-generated)

Counts are per-task tags over all test seeds, reader mode `first`; a task may carry several tags. Rates are per task. Examples are the first ones encountered, not cherry-picked successes.

| Category | A  flat vector (similarity only) | B  flat vector + static score (recency, confidence) | C  ReMem (full) |
|---|---:|---:|---:|
| `gold_not_in_pool` | 311 (2.5%) | 311 (2.5%) | 311 (2.5%) |
| `gold_pooled_not_selected` | 1947 (15.5%) | 1935 (15.4%) | 1190 (9.5%) |
| `utility_demoted_gold` | 30 (0.2%) | 40 (0.3%) | 71 (0.6%) |
| `useful_filtered` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `irrelevant_retrieved` | 2204 (17.6%) | 2204 (17.6%) | 1931 (15.4%) |
| `outdated_preferred` | 1226 (9.8%) | 867 (6.9%) | 0 (0.0%) |
| `success_without_memory` | 409 (3.3%) | 410 (3.3%) | 280 (2.2%) |
| `fail_despite_gold` | 2712 (21.6%) | 2372 (18.9%) | 1651 (13.2%) |

## Per family (D ReMem)

| Category | budget_medium | budget_tight | contradiction | distractor | distractor_norecur | long_horizon | rare_critical | state_change | useless_frequent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gold_not_in_pool` | 30 | 129 | 0 | 93 | 35 | 0 | 0 | 5 | 19 |
| `gold_pooled_not_selected` | 296 | 464 | 82 | 34 | 46 | 47 | 63 | 78 | 80 |
| `utility_demoted_gold` | 1 | 17 | 6 | 3 | 0 | 5 | 0 | 15 | 24 |
| `useful_filtered` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `irrelevant_retrieved` | 231 | 213 | 233 | 237 | 56 | 264 | 249 | 204 | 244 |
| `outdated_preferred` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `success_without_memory` | 69 | 78 | 16 | 35 | 16 | 13 | 13 | 17 | 23 |
| `fail_despite_gold` | 160 | 113 | 219 | 213 | 37 | 256 | 238 | 190 | 225 |

## Examples: C  ReMem (full)


### `gold_not_in_pool`

- **distractor seed 1 task 8** (current, success=False): _Which API release does orders serve?_
  - gold `fact_00072` pool_rank=-1 ctx_pos=-1 state=ACTIVE: The orders service serves API v2.
  - served [trap/ACTIVE] Earlier question: 'Which API release does orders serve?' still unresolved.
  - served [gold/ACTIVE] API release for storage: v4.
  - served [trap/ACTIVE] Earlier question: 'Which API version does orders expose?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does auth serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - agent distracted=False guessed=['orders.api_version']
- **distractor seed 1 task 17** (current, success=False): _Which API release does pricing serve?_
  - gold `fact_00030` pool_rank=-1 ctx_pos=-1 state=ACTIVE: The pricing service serves API v2.
  - served [gold/ACTIVE] API release for storage: v4.
  - served [trap/ACTIVE] Earlier question: 'Which API version does pricing expose?' still unresolved.
  - served [gold/ACTIVE] analytics exposes API version v4.
  - served [gold/ACTIVE] inventory exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - agent distracted=False guessed=['pricing.api_version']
- **distractor seed 1 task 18** (current, success=False): _Which API release does pricing serve?_
  - gold `fact_00030` pool_rank=-1 ctx_pos=-1 state=ACTIVE: The pricing service serves API v2.
  - served [gold/ACTIVE] API release for storage: v4.
  - served [trap/ACTIVE] Earlier question: 'Which API version does pricing expose?' still unresolved.
  - served [gold/ACTIVE] inventory exposes API version v4.
  - served [gold/ACTIVE] analytics exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - agent distracted=True guessed=['pricing.api_version']

### `gold_pooled_not_selected`

- **distractor seed 0 task 21** (current, success=False): _What version of the API is auth running and who owns auth?_
  - gold `fact_00054` pool_rank=3 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [trap/ACTIVE] Earlier question: 'Who is responsible for auth?' still unresolved.
  - served [gold/ACTIVE] shipping exposes API version v2.
  - served [trap/ACTIVE] Earlier question: 'Who owns auth?' still unresolved.
  - agent distracted=True guessed=['auth.api_version', 'auth.owner']
- **distractor seed 0 task 46** (current, success=False): _Which API release does auth serve?_
  - gold `fact_00054` pool_rank=18 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] API release for support: v2.
  - served [gold/ACTIVE] API release for audit: v2.
  - served [gold/ACTIVE] recs exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - agent distracted=False guessed=['auth.api_version']
- **distractor seed 0 task 49** (current, success=False): _Which API version does auth expose and which person is the owner of auth?_
  - gold `fact_00054` pool_rank=4 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [gold/ACTIVE] reports exposes API version v2.
  - served [gold/ACTIVE] shipping exposes API version v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] search exposes API version v2.
  - agent distracted=False guessed=['auth.api_version']

### `utility_demoted_gold`

- **distractor seed 0 task 21** (current, success=False): _What version of the API is auth running and who owns auth?_
  - gold `fact_00054` pool_rank=3 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [trap/ACTIVE] Earlier question: 'Who is responsible for auth?' still unresolved.
  - served [gold/ACTIVE] shipping exposes API version v2.
  - served [trap/ACTIVE] Earlier question: 'Who owns auth?' still unresolved.
  - agent distracted=True guessed=['auth.api_version', 'auth.owner']
- **distractor seed 0 task 49** (current, success=False): _Which API version does auth expose and which person is the owner of auth?_
  - gold `fact_00054` pool_rank=4 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [gold/ACTIVE] reports exposes API version v2.
  - served [gold/ACTIVE] shipping exposes API version v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] search exposes API version v2.
  - agent distracted=False guessed=['auth.api_version']
- **distractor seed 0 task 90** (current, success=True): _Which API version does auth expose and which person is the owner of auth?_
  - gold `fact_00054` pool_rank=4 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [gold/ACTIVE] reports exposes API version v2.
  - served [gold/ACTIVE] shipping exposes API version v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] search exposes API version v2.
  - agent distracted=False guessed=['auth.api_version']

### `irrelevant_retrieved`

- **distractor seed 0 task 1** (current, success=False): _Who owns auth?_
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [trap/ACTIVE] Earlier question: 'Who is responsible for auth?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Who owns auth?' still unresolved.
  - served [gold/ACTIVE] payments is owned by grace.
  - served [gold/ACTIVE] alice is responsible for recs.
  - agent distracted=True guessed=['auth.owner']
- **distractor seed 0 task 17** (current, success=False): _Which API version does checkout expose?_
  - gold `fact_00060` pool_rank=1 ctx_pos=3 state=ACTIVE: checkout exposes API version v4.
  - served [gold/ACTIVE] shipping exposes API version v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [trap/ACTIVE] Earlier question: 'Which API version does checkout expose?' still unresolved.
  - served [gold/ACTIVE] checkout exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - agent distracted=True guessed=['checkout.api_version']
- **distractor seed 0 task 21** (current, success=False): _What version of the API is auth running and who owns auth?_
  - gold `fact_00054` pool_rank=3 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [trap/ACTIVE] Earlier question: 'Who is responsible for auth?' still unresolved.
  - served [gold/ACTIVE] shipping exposes API version v2.
  - served [trap/ACTIVE] Earlier question: 'Who owns auth?' still unresolved.
  - agent distracted=True guessed=['auth.api_version', 'auth.owner']

### `success_without_memory`

- **distractor seed 0 task 90** (current, success=True): _Which API version does auth expose and which person is the owner of auth?_
  - gold `fact_00054` pool_rank=4 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [gold/ACTIVE] reports exposes API version v2.
  - served [gold/ACTIVE] shipping exposes API version v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] search exposes API version v2.
  - agent distracted=False guessed=['auth.api_version']
- **distractor seed 0 task 95** (current, success=True): _Which API release does auth serve?_
  - gold `fact_00054` pool_rank=18 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] API release for support: v2.
  - served [gold/ACTIVE] API release for audit: v2.
  - served [gold/ACTIVE] recs exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - agent distracted=False guessed=['auth.api_version']
- **distractor seed 0 task 121** (current, success=True): _Which API release does auth serve?_
  - gold `fact_00054` pool_rank=18 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] API release for support: v2.
  - served [gold/ACTIVE] API release for audit: v2.
  - served [gold/ACTIVE] recs exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - agent distracted=False guessed=['auth.api_version']

### `fail_despite_gold`

- **distractor seed 0 task 1** (current, success=False): _Who owns auth?_
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [trap/ACTIVE] Earlier question: 'Who is responsible for auth?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Who owns auth?' still unresolved.
  - served [gold/ACTIVE] payments is owned by grace.
  - served [gold/ACTIVE] alice is responsible for recs.
  - agent distracted=True guessed=['auth.owner']
- **distractor seed 0 task 17** (current, success=False): _Which API version does checkout expose?_
  - gold `fact_00060` pool_rank=1 ctx_pos=3 state=ACTIVE: checkout exposes API version v4.
  - served [gold/ACTIVE] shipping exposes API version v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [trap/ACTIVE] Earlier question: 'Which API version does checkout expose?' still unresolved.
  - served [gold/ACTIVE] checkout exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - agent distracted=True guessed=['checkout.api_version']
- **distractor seed 0 task 22** (current, success=False): _Which API version does checkout expose?_
  - gold `fact_00060` pool_rank=1 ctx_pos=2 state=ACTIVE: checkout exposes API version v4.
  - served [gold/ACTIVE] shipping exposes API version v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] checkout exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API version does checkout expose?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - agent distracted=True guessed=['checkout.api_version']

## Examples: A  flat vector (similarity only)


### `gold_not_in_pool`

- **distractor seed 1 task 8** (current, success=False): _Which API release does orders serve?_
  - gold `fact_00072` pool_rank=-1 ctx_pos=-1 state=ACTIVE: The orders service serves API v2.
  - served [trap/ACTIVE] Earlier question: 'Which API release does orders serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API version does orders expose?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does storage serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does auth serve?' still unresolved.
  - agent distracted=False guessed=['orders.api_version']
- **distractor seed 1 task 17** (current, success=False): _Which API release does pricing serve?_
  - gold `fact_00030` pool_rank=-1 ctx_pos=-1 state=ACTIVE: The pricing service serves API v2.
  - served [trap/ACTIVE] Earlier question: 'Which API version does pricing expose?' still unresolved.
  - served [gold/ACTIVE] analytics exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does storage serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does auth serve?' still unresolved.
  - agent distracted=False guessed=['pricing.api_version']
- **distractor seed 1 task 18** (current, success=False): _Which API release does pricing serve?_
  - gold `fact_00030` pool_rank=-1 ctx_pos=-1 state=ACTIVE: The pricing service serves API v2.
  - served [trap/ACTIVE] Earlier question: 'Which API version does pricing expose?' still unresolved.
  - served [gold/ACTIVE] analytics exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does storage serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does auth serve?' still unresolved.
  - agent distracted=True guessed=['pricing.api_version']

### `gold_pooled_not_selected`

- **distractor seed 0 task 46** (current, success=False): _Which API release does auth serve?_
  - gold `fact_00054` pool_rank=18 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] recs exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does shipping serve?' still unresolved.
  - served [gold/ACTIVE] API release for ranking: v3.
  - agent distracted=False guessed=['auth.api_version']
- **distractor seed 0 task 95** (current, success=True): _Which API release does auth serve?_
  - gold `fact_00054` pool_rank=18 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] recs exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does shipping serve?' still unresolved.
  - served [gold/ACTIVE] API release for ranking: v3.
  - agent distracted=False guessed=['auth.api_version']
- **distractor seed 0 task 109** (current, success=False): _Which API version does auth expose?_
  - gold `fact_00054` pool_rank=19 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - served [trap/ACTIVE] Earlier question: 'Which API version does shipping expose?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API version does analytics expose?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API version does checkout expose?' still unresolved.
  - served [gold/ACTIVE] API release for quota: v1.
  - served [gold/ACTIVE] API release for ledger: v2.
  - agent distracted=False guessed=['auth.api_version']

### `utility_demoted_gold`

- **budget_tight seed 0 task 3** (current, success=False): _Tell me the region where checkout is hosted._
  - gold `fact_00058` pool_rank=4 ctx_pos=-1 state=ACTIVE: checkout service is hosted in region eu-west-1.
  - served [trap/ACTIVE] Earlier question: 'Tell me the region where checkout is hosted.' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Tell me the region where gateway is hosted.' still unresolved.
  - served [gold/ACTIVE] Region for pricing deployment: us-east-1.
  - agent distracted=False guessed=['checkout.region']
- **budget_tight seed 0 task 22** (current, success=False): _Tell me the region where gateway is hosted._
  - gold `fact_00064` pool_rank=4 ctx_pos=-1 state=ACTIVE: The gateway service deploys to region eu-west-1.
  - served [trap/ACTIVE] Earlier question: 'Tell me the region where gateway is hosted.' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Tell me the region where pricing is hosted.' still unresolved.
  - served [gold/ACTIVE] Region for pricing deployment: us-east-1.
  - agent distracted=True guessed=['gateway.region']
- **budget_tight seed 0 task 27** (current, success=False): _Tell me the region where checkout is hosted._
  - gold `fact_00058` pool_rank=4 ctx_pos=-1 state=ACTIVE: checkout service is hosted in region eu-west-1.
  - served [trap/ACTIVE] Earlier question: 'Tell me the region where checkout is hosted.' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Tell me the region where gateway is hosted.' still unresolved.
  - served [gold/ACTIVE] Region for pricing deployment: us-east-1.
  - agent distracted=False guessed=['checkout.region']

### `irrelevant_retrieved`

- **distractor seed 0 task 1** (current, success=False): _Who owns auth?_
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [trap/ACTIVE] Earlier question: 'Who is responsible for auth?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Who owns auth?' still unresolved.
  - served [gold/ACTIVE] payments is owned by grace.
  - served [gold/ACTIVE] alice is responsible for recs.
  - agent distracted=True guessed=['auth.owner']
- **distractor seed 0 task 17** (current, success=False): _Which API version does checkout expose?_
  - gold `fact_00060` pool_rank=1 ctx_pos=1 state=ACTIVE: checkout exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API version does checkout expose?' still unresolved.
  - served [gold/ACTIVE] checkout exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API version does shipping expose?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API version does analytics expose?' still unresolved.
  - agent distracted=True guessed=['checkout.api_version']
- **distractor seed 0 task 21** (current, success=False): _What version of the API is auth running and who owns auth?_
  - gold `fact_00054` pool_rank=3 ctx_pos=3 state=ACTIVE: The auth service serves API v2.
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [trap/ACTIVE] Earlier question: 'Who is responsible for auth?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Who owns auth?' still unresolved.
  - served [gold/ACTIVE] The auth service serves API v2.
  - served [gold/ACTIVE] Region for auth deployment: us-west-2.
  - agent distracted=True guessed=['auth.api_version', 'auth.owner']

### `outdated_preferred`

- **contradiction seed 0 task 49** (current, success=False): _Which API version does billing expose?_
  - gold `fact_00084` pool_rank=8 ctx_pos=-1 state=ACTIVE: The billing service serves API v3.
  - served [trap/ACTIVE] Earlier question: 'Which API version does billing expose?' still unresolved.
  - served [gold/ACTIVE] billing exposes API version v1.
  - served [gold/ACTIVE] API release for returns: v1.
  - served [gold/ACTIVE] API release for media: v2.
  - served [gold/ACTIVE] API release for reports: v3.
  - agent distracted=False guessed=[]
- **contradiction seed 0 task 80** (current, success=False): _What version of the API is support running and who owns support?_
  - gold `fact_00083` pool_rank=2 ctx_pos=2 state=ACTIVE: support switched; the API version is now v3.
  - gold `fact_00050` pool_rank=1 ctx_pos=1 state=ACTIVE: grace is responsible for support.
  - served [gold/ACTIVE] API release for support: v2.
  - served [gold/ACTIVE] grace is responsible for support.
  - served [update/ACTIVE] support switched; the API version is now v3.
  - served [gold/ACTIVE] API release for returns: v1.
  - served [gold/ACTIVE] API release for media: v2.
  - agent distracted=False guessed=[]
- **contradiction seed 0 task 104** (current, success=False): _Which API version does notify expose?_
  - gold `fact_00082` pool_rank=1 ctx_pos=1 state=ACTIVE: notify switched; the API version is now v1.
  - served [gold/ACTIVE] notify exposes API version v2.
  - served [update/ACTIVE] notify switched; the API version is now v1.
  - served [gold/ACTIVE] API release for returns: v1.
  - served [trap/ACTIVE] Earlier question: 'Which API version does billing expose?' still unresolved.
  - served [gold/ACTIVE] API release for media: v2.
  - agent distracted=False guessed=[]

### `success_without_memory`

- **distractor seed 0 task 95** (current, success=True): _Which API release does auth serve?_
  - gold `fact_00054` pool_rank=18 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] recs exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does shipping serve?' still unresolved.
  - served [gold/ACTIVE] API release for ranking: v3.
  - agent distracted=False guessed=['auth.api_version']
- **distractor seed 0 task 121** (current, success=True): _Which API release does auth serve?_
  - gold `fact_00054` pool_rank=18 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] recs exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does shipping serve?' still unresolved.
  - served [gold/ACTIVE] API release for ranking: v3.
  - agent distracted=False guessed=['auth.api_version']
- **distractor seed 0 task 172** (current, success=True): _Which API release does auth serve?_
  - gold `fact_00054` pool_rank=18 ctx_pos=-1 state=ACTIVE: The auth service serves API v2.
  - served [gold/ACTIVE] API release for ledger: v2.
  - served [gold/ACTIVE] recs exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API release does shipping serve?' still unresolved.
  - served [gold/ACTIVE] API release for ranking: v3.
  - agent distracted=False guessed=['auth.api_version']

### `fail_despite_gold`

- **distractor seed 0 task 1** (current, success=False): _Who owns auth?_
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [trap/ACTIVE] Earlier question: 'Who is responsible for auth?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Who owns auth?' still unresolved.
  - served [gold/ACTIVE] payments is owned by grace.
  - served [gold/ACTIVE] alice is responsible for recs.
  - agent distracted=True guessed=['auth.owner']
- **distractor seed 0 task 17** (current, success=False): _Which API version does checkout expose?_
  - gold `fact_00060` pool_rank=1 ctx_pos=1 state=ACTIVE: checkout exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API version does checkout expose?' still unresolved.
  - served [gold/ACTIVE] checkout exposes API version v4.
  - served [trap/ACTIVE] Earlier question: 'Which API release does checkout serve?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API version does shipping expose?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Which API version does analytics expose?' still unresolved.
  - agent distracted=True guessed=['checkout.api_version']
- **distractor seed 0 task 21** (current, success=False): _What version of the API is auth running and who owns auth?_
  - gold `fact_00054` pool_rank=3 ctx_pos=3 state=ACTIVE: The auth service serves API v2.
  - gold `fact_00053` pool_rank=0 ctx_pos=0 state=ACTIVE: The owner of auth is ivan.
  - served [gold/ACTIVE] The owner of auth is ivan.
  - served [trap/ACTIVE] Earlier question: 'Who is responsible for auth?' still unresolved.
  - served [trap/ACTIVE] Earlier question: 'Who owns auth?' still unresolved.
  - served [gold/ACTIVE] The auth service serves API v2.
  - served [gold/ACTIVE] Region for auth deployment: us-west-2.
  - agent distracted=True guessed=['auth.api_version', 'auth.owner']
