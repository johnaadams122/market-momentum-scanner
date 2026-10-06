# Security

## Reporting a vulnerability

Use GitHub's private vulnerability reporting (the "Report a vulnerability" button
in this repository's Security tab) when it is shown there. If it is not shown, do
not send sensitive details: open a minimal public issue that only asks for a
private way to report, without the exploit, credentials or personal data, and
wait for a reply.

Reports should identify the affected commit, the expected and observed behavior,
and a minimal reproduction using synthetic data. Never include credentials,
real account data, private files or machine-specific paths.

## Scope and maintenance

This is a portfolio project supplied under the MIT license. There is no support
response-time commitment or claim that older commits receive security fixes.
Review configuration and permissions before running it with your own data.
Model output and external service responses must be treated as untrusted input.

If a public disclosure exposes a credential, revoke or rotate it immediately.
Removing a file or making a repository private does not remove existing clones.

## Trust boundaries to know before running it

- **Filing and news text steers a language model.** The text of SEC filings and news
  articles is untrusted input, and a local language model reads it and returns the label
  (`real`, `pump` or `no-data`), a confidence and a fixed-price-buyout flag. The prompts tell the
  model to ignore instructions inside that text, which is a mitigation, not a guarantee, so crafted
  text can push the model's answer in either direction. Enforced in code, outside the model: instrument
  eligibility, missing or non-finite float and relative-volume checks, the SEC dilution classification
  (offering, shelf and reverse-split filings and dilution wording in filing text), item-code routing in
  the 8-K mode, the confidence floor (in the 8-K mode and in the SEC-filing fallback checks of the
  other modes) and the check that the model's answer is well formed. In the
  one-shot 8-K mode these run before or after the model and the model cannot override a code-side
  rejection. In the one-shot `--source finnhub` mode the model's label and confidence do not reject a
  candidate and are only recorded: the candidate list is ordered by gap and `catalyst_score` is always
  0.0, so crafted news there can only cause a REJECT (an unusable answer or a fixed-price-buyout flag) or
  a PROCEED with whatever label and confidence the model gave, and cannot change any score or order.
  The movers daemon has the same reject conditions, and in addition the label, confidence and news
  source feed the score that orders its watchlist (a bonus for a `real` label only), so crafted news
  can raise a candidate's rank there. In both news modes dilution wording in the news text still sets a
  dilution flag in code. Everything the model
  decides, namely whether a catalyst counts as real, its confidence and the buyout flag, is only as
  reliable as the model.
- **Adapter modules are trusted code.** `MARKET_MOMENTUM_TOKEN_PROVIDER`,
  `MARKET_MOMENTUM_RATE_LIMITER`, `MARKET_MOMENTUM_CALENDAR` and `MARKET_MOMENTUM_COST_MODEL` name
  Python modules that are imported with `importlib` and run with the scanner's own permissions when
  they are loaded. Only point them at modules you wrote or reviewed, and keep those environment
  variables out of reach of untrusted users.
- **Text can leave the machine when the shadow judge is on.** With `SCANNER_JUDGE_SHADOW=1` and an
  `ANTHROPIC_API_KEY`, each judged candidate's news headline and summary are sent to Anthropic's API,
  and the Alpaca client can fill the summary with the full article body. Leave the shadow judge off if
  that text must not leave the machine. Shadow log files contain headlines and model rationales; do not
  commit them.
- **Credentials.** The Finnhub key, the Alpaca key and secret and the SEC contact string come from the
  environment when a request is made; the Anthropic key is read from the environment by the Anthropic
  SDK; the Schwab bearer token is whatever the token-provider module's `get_valid_token()` returns.
  This code sends the Finnhub, Alpaca, Schwab and SEC values only as request headers (the SEC contact
  string as the `User-Agent` header). Error messages for failed Schwab, SEC EDGAR, Finnhub and Alpaca
  requests, and for a failed token-provider call, report only the exception type (plus the HTTP status
  for EDGAR), not the underlying library text, so a malformed value for those requests does not appear
  in output, logs or the written watchlist. The Anthropic shadow path is handled differently: its error
  text is not reduced to the type but masked before it is logged, by pattern (text starting `sk-ant-`)
  and by exact match against the configured key. That masking does not cover other secret formats.
