# DeepSeek V4 Flash — five-tier development smoke

**Status:** preliminary capability smoke, not a benchmark result  
**Split:** development, seed index 0 only  
**Role:** retailer (T1–T4), retailer A (T5)  
**Sampling:** temperature 0, strict required tool call  
**Protocol:** 180/180 decisions clean

| Tier | Local cost | Base-stock cost | Reward | Immediate fill | Bullwhip |
|---:|---:|---:|---:|---:|---:|
| 1 | 42.0 | 186.0 | 0.816 | 1.000 | undefined (constant demand) |
| 2 | 340.5 | 211.5 | 0.383 | 0.781 | 1.789 |
| 3 | 693.0 | 201.0 | 0.225 | 0.903 | 0.408 |
| 4 | 697.5 | 201.0 | 0.224 | 0.923 | 0.308 |
| 5 | 1,754.5 | 362.0 | 0.171 | 0.717 | 9.576 |

Recorded successful-trace usage was 184,727 input tokens and 7,920 output
tokens, approximately $0.028 at the account-visible Akash price. This excludes
the direct API probe and aborted integration attempts and is not an account
billing total.

## Interpretation limits

- One seed cannot estimate variance or support model-ranking claims.
- The initial integration attempt is excluded: the harness allowed prose, hit a
  64-token limit, and correctly failed the protocol gate. Requiring a tool call
  fixed the issue.
- Akash returned transient HTTP 502 responses. Two request-level retries occur
  before environment mutation and allowed clean completion.
- Tier 1 exposed a likely baseline problem: constant ordering at demand 8 costs
  42, while the current adaptive base-stock reference costs 186. The reference's
  `(lead time + 1) * forecast` target must be checked against the implemented
  weekly event order before larger paid evaluations or a v1 freeze.
