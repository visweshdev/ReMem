# Security

ReMem stores memories as plain JSON files on your own disk and makes no network calls unless you configure an
external embedding or LLM provider. Anything you `add` is stored unencrypted in the memory folder, so do not store
secrets (passwords, API keys) in it.

To report a security problem, please use GitHub's private
[security advisory form](https://github.com/visweshdev/ReMem/security/advisories/new) rather than a public issue.
