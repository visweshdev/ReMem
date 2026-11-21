# Failure analysis (auto-generated)

Counts are per-task tags over all test seeds, reader mode `first`; a task may carry several tags. Rates are per task. Examples are the first ones encountered, not cherry-picked successes.

| Category | A  flat vector (similarity only) | B  flat vector + static score (recency, confidence) | C  ReMem (full, learns which attributes change) |
|---|---:|---:|---:|
| `gold_not_in_pool` | 1184 (3.1%) | 1184 (3.1%) | 1184 (3.1%) |
| `gold_pooled_not_selected` | 7445 (19.8%) | 7388 (19.7%) | 4416 (11.7%) |
| `utility_demoted_gold` | 240 (0.6%) | 275 (0.7%) | 308 (0.8%) |
| `useful_filtered` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `irrelevant_retrieved` | 6952 (18.5%) | 6947 (18.5%) | 6100 (16.2%) |
| `outdated_preferred` | 3244 (8.6%) | 2211 (5.9%) | 120 (0.3%) |
| `success_without_memory` | 1643 (4.4%) | 1641 (4.4%) | 1020 (2.7%) |
| `fail_despite_gold` | 7686 (20.4%) | 6688 (17.8%) | 5069 (13.5%) |

## Per family (D ReMem)

| Category | budget_medium | budget_tight | contradiction | distractor | distractor_norecur | long_horizon | rare_critical | state_change | useless_frequent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gold_not_in_pool` | 251 | 469 | 18 | 203 | 98 | 4 | 45 | 60 | 36 |
| `gold_pooled_not_selected` | 953 | 1972 | 183 | 208 | 140 | 328 | 166 | 261 | 205 |
| `utility_demoted_gold` | 47 | 71 | 24 | 35 | 5 | 38 | 14 | 38 | 36 |
| `useful_filtered` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `irrelevant_retrieved` | 814 | 696 | 690 | 734 | 185 | 793 | 759 | 656 | 773 |
| `outdated_preferred` | 0 | 0 | 48 | 0 | 0 | 0 | 0 | 72 | 0 |
| `success_without_memory` | 195 | 371 | 41 | 108 | 45 | 93 | 37 | 69 | 61 |
| `fail_despite_gold` | 526 | 255 | 694 | 646 | 135 | 728 | 711 | 647 | 727 |

## Examples: C  ReMem (full, learns which attributes change)


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

### `outdated_preferred`

- **contradiction seed 0 task 185** (current, success=False): _Which region does reports deploy to?_
  - gold `fact_00085` pool_rank=0 ctx_pos=3 state=ACTIVE: Region for reports deployment: eu-west-1.
  - served [gold/ACTIVE] reports service is hosted in region us-west-2.
  - served [gold/ACTIVE] The returns service deploys to region us-west-2.
  - served [gold/ACTIVE] Region for audit deployment: us-west-2.
  - served [update/ACTIVE] Region for reports deployment: eu-west-1.
  - served [gold/ACTIVE] Region for auth deployment: us-west-2.
  - agent distracted=False guessed=[]
- **contradiction seed 1 task 105** (current, success=False): _Which API version does messaging expose?_
  - gold `fact_00078` pool_rank=0 ctx_pos=2 state=ACTIVE: API release for messaging: v2.
  - served [gold/ACTIVE] messaging exposes API version v4.
  - served [gold/ACTIVE] API release for sessions: v4.
  - served [update/ACTIVE] API release for messaging: v2.
  - served [gold/ACTIVE] API release for ledger: v4.
  - served [gold/ACTIVE] API release for fraud: v4.
  - agent distracted=False guessed=[]
- **contradiction seed 1 task 120** (current, success=False): _Which person is the owner of analytics and tell me the region where analytics is hosted._
  - gold `fact_00008` pool_rank=0 ctx_pos=0 state=ACTIVE: bob is responsible for analytics.
  - gold `fact_00080` pool_rank=2 ctx_pos=2 state=ACTIVE: analytics migrated; it is now hosted in region us-west-2.
  - served [gold/ACTIVE] bob is responsible for analytics.
  - served [gold/ACTIVE] Region for analytics deployment: sa-east-1.
  - served [update/ACTIVE] analytics migrated; it is now hosted in region us-west-2.
  - served [gold/ACTIVE] auth service is hosted in region eu-central-1.
  - served [gold/ACTIVE] Region for sessions deployment: us-east-1.
  - agent distracted=True guessed=['analytics.owner', 'analytics.region']

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
