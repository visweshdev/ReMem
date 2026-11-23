# Security

The default ReMem configuration runs locally. A persistent store writes unencrypted facts and ledger data to `facts.jsonl`
and `ledger.json`. Keep passwords and API keys out of memory stores.

Optional model and tokenizer dependencies can download their assets. If you configure a remote embedding or LLM endpoint,
text sent for embedding, extraction, or an update check is shared with that provider.

To report a security problem, please use GitHub's private
[security advisory form](https://github.com/visweshdev/ReMem/security/advisories/new) rather than a public issue.
