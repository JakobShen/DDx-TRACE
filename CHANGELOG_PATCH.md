# Revised patch changelog

## Protocol / evidence unit
- Added `--reveal-unit {eurorad,figure}`; default is now `eurorad`.
- `eurorad` keeps each source EuroRad `imaging_examination` / figure-protocol entry as one requestable evidence bundle. Routine T1/T2/FLAIR/STIR/DWI sequence entries are not merged.
- Initial prompt no longer passes the EuroRad stem such as "Based on the imaging figures provided..."; it uses a benchmark-native hidden-evidence task statement.
- Agent is told the count of hidden available evidence bundles, not the inventory list.

## Follow-up / temporal policy
- `time_past != null` no longer excludes exams.
- Follow-up, comparison, postoperative, and post-treatment bundles remain requestable when local images exist.
- Generic requests preferentially match initial/current studies; explicit follow-up/delayed requests can match delayed studies.
- Numeric time metadata without unit is marked as `time_specified_numeric_without_unit` and `time_past_interpretable=false`.

## Matcher / resolver
- Expanded modality, region, sequence, contrast, and nuclear medicine normalization.
- Added hidden caption text to matcher-only resolver tokens; captions are not revealed to the agent.
- Ambiguous/broad requests are no longer penalized as invalid when an eligible above-threshold candidate exists. The resolver reveals the best-scoring EuroRad-style bundle and logs the resolution as `matched_ambiguous_*_best_effort`.
- Adjusted already-revealed logic so a revealed broad/optional match does not automatically block a more acquisition-specific unrevealed bundle.

## Route metrics
- Added `route_evaluable` and `evidence_modality_class` to exam bundles.
- Histopathology, clinical photos, and illustrations are retained for provenance but excluded from imaging-route denominators (`S_ER`, `S_order`, `Opt_burden`).
- Renamed the optional-label proxy to `nonessential_request_rate_proxy`; it should not be described as unnecessary imaging.

## Judge / deterministic scorer
- Judge prompt schema now explicitly includes `trajectory_scores`.
- Deterministic scorer default version updated to `rule_v8_rubric_extracted_terms`.
- Rule scorer extracts case-specific accepted terms from rubric prose patterns; no global hand-built diagnosis synonym dictionary was added.

## Run artifacts
- `run_config` now records `reveal_unit`, input file name/SHA256, and clinical-history redaction summary.
- `code_version` now records schema, judge, scorer, evidence-unit, follow-up-policy, and route-denominator-policy versions.
- Request outcome summary now includes per-case request outcome counts.

## Input data helper
- Added `fix_eurorad_inputdata.py` and a generated `eurorad_neuro_01_fixed.json`.
- Standardizes `difficulty` / `rarity` labels.
- Redacts targeted direct clinical-history leakage while preserving `original_clinical_history`.

## Round 3 EuroRad-unit update

- Changed default reveal unit to `eurorad` to stay close to the EuroRad figure/protocol inventory.
- Updated prompt, README, metric notes, run config, and manuscript wording accordingly.
- Added a simple spine-level region penalty so explicit cervical/thoracic/lumbar/lumbosacral requests do not match the wrong spine level. Generic spine requests are resolved best-effort when possible rather than penalized as ambiguous.

## Round 4 ambiguity policy update

- Ambiguous requests are treated as EuroRad-inventory resolver artifacts, not agent errors.
- `ambiguous_request` is no longer emitted as an invalid outcome when a matchable candidate exists.
- Matched ambiguous cases are counted as normal matched requests for `B_inv`, `S_ER`, and `S_order`, while `num_ambiguous_resolved_requests` and per-request `ambiguity_resolved=true` preserve the audit trail.
- Documentation now defines the EuroRad-style evidence unit explicitly as the source `imaging_examination` / figure-protocol item, not a hospital study-level order.

## Ablation extension

- Added `ablation_main.py` as the separate entrypoint for non-official ablation studies.
- Implemented five ablation settings:
  - `history-only`: clinical history only, no images and no requests.
  - `all-images-at-once`: all requestable EuroRad-style evidence units are attached in one call.
  - `random-order-reveal`: the model cannot request; evidence units are passively revealed one at a time in deterministic random order, up to the request budget, after a history-only baseline turn.
  - `gold-order-reveal`: the model cannot request; evidence units are passively revealed one at a time by `preferred_order` / gold order, up to the request budget, after a history-only baseline turn.
  - `oracle-findings`: request-based sequential setting; when a request matches, the model receives image + minimal metadata + oracle imaging findings for the matched evidence unit.
- Passive ablations mark route/request metrics (`S_ER`, `Opt_burden`, `Inv_req_rate`, `S_order`) as not applicable.
- `oracle-findings` keeps route/request metrics applicable because the model still actively requests evidence.
- Official `run_main.py` and `ablation_main.py` now default to fixed judge settings: `--judge-provider vertex --judge-model gemini-3-flash-preview`.
