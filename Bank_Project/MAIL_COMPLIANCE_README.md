# Mail-compliant forecasting pipeline

The numbered Phase 1–6 scripts are the governed implementation of the emailed
use case. They predict full-day branch withdrawals and deposits directly, then
allocate those totals into AM and PM using training-only historical shares.

It includes:

- direct daily modeling without AM-to-PM recursive dependence;
- training-only branch/calendar AM/PM allocation profiles;
- exact conservation: AM + PM equals the daily forecast;
- configurable salary-cycle indicators;
- approved public-holiday, religious-event and bank-event indicators;
- historical-spike indicators and spike-weighted robust training;
- chronological train, validation and untouched test periods;
- a same-half-day/previous-week baseline;
- nonnegative cash-flow forecasts and residual prediction intervals;
- model-aligned permutation-importance XAI;
- balance-, reserve-, capacity-, lead-time- and cost-aware replenishment;
- a strict compliance audit that fails when governed business inputs are absent.

## Required governed inputs

Copy `business_event_calendar_template.csv` to the configured confidential data
location and replace every example with dates approved by the bank. It must cover
the complete training and forecast horizon. Event dates are deliberately not
guessed by the code.

Copy `branch_operations_template.csv` to the configured confidential data
location and provide one current record per branch. Opening cash, reserve limits,
vault capacity, lead times and costs cannot be derived reliably from transaction
flows and are therefore never silently defaulted.

Both files are CSVs and are excluded by the repository's existing `*.csv` rule.

## Commands

Run Phase 1 through Phase 6 in order after placing both governed input files in
`Bank DataSet`. Phase 3 deliberately stops when the approved event calendar is
absent; Phase 6 deliberately stops before recommendations when operational cash
inputs are absent. Model-ready and output CSVs remain covered by the repository
CSV ignore rule.

## Interpretation boundary

`flow_deficit = max(withdrawals - deposits, 0)` is not called a physical cash
replenishment requirement. Replenishment is calculated separately from opening
cash, conservative prediction bounds, minimum reserve and vault capacity.

XAI feature importance describes statistical model reliance, not causality.
