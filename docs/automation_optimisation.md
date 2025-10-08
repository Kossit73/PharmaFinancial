# Automation and Optimisation Opportunities

This project already supports a wide range of interactive tooling. The recent
update introduces deterministic caching inside the Streamlit app so repeated
runs reuse parsed assumptions and model outputs instead of recalculating every
widget interaction. The following ideas can further automate workflows without
risking the integrity of the financial engine:

## 1. Background Model Rebuilds
* Move expensive scenario batches (e.g. Monte Carlo iterations) into
  `st.session_state` jobs executed asynchronously via `st.spinner` callbacks.
* Persist the last successful payload fingerprint and reuse the cached results
  while a background recalculation completes, ensuring analysts always see the
  most recent stable outputs.

## 2. Snapshot Management
* Provide a "Save Scenario" button that writes the current payload fingerprint
  and key metrics to disk. This enables quick switching between cases and
  creates an auditable trail of investor-ready assumptions.
* Introduce a comparison view that aligns two snapshots so deltas across
  revenue, margin, and cash flow statements are computed automatically.

## 3. Continuous Export Automation
* Schedule the CLI exporter (`python -m pharma_financial.cli --output <dir>`) via
  a CI pipeline or cron job to distribute refreshed PDF/Excel packs to
  stakeholders after each assumption change.
* Add Slack or Teams webhooks to notify investors when a new consolidated report
  is generated, linking directly to the exported artefacts.

## 4. Performance Monitoring
* Wrap the caching helpers with lightweight timing instrumentation to surface
  how long each rebuild takes and display the metrics in the Key Metrics
  dashboard. This helps identify assumption sets that produce heavy models.
* Extend the AI configuration summary with success/error telemetry so users know
  when forecasts were produced from cached responses or fresh provider calls.

These enhancements focus on automation and optimisation pathways while keeping
all core financial computations intact. They can be rolled out incrementally as
team priorities evolve.
