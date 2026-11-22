# Metric catalogue

An interactive catalogue of the framework's metrics: filter by dimension, category, agent scope and evidence, open a metric for its description and implementation details. The page design is by Chara (ARC).

Open one of the two files directly in a browser. Each is self-contained, so nothing else needs to sit next to it.

| File | What it is |
|---|---|
| `aixpert_catalogue.html` | the catalogue as designed |
| `aixpert_catalogue_picker.html` | the same page with metric picking |

## Picking metrics

In the picker version, use **Add to my selection** on a card (or in the metric detail window) to build the set of metrics a use case needs. The **My selection** section shows how the selection covers the ten dimensions, and a dimension with no registered metric says so. **Download selection (JSON)** writes `{"schema": "aixpert-metric-selection/1", "label": ..., "metrics": [ids]}`. **Load selection** reads it back (a JSON list of ids or a text file of ids also works).

The demo dashboards read that file: load it under **Metrics shown**, or open a dashboard with it already applied:

```bash
python -m harness.diagflow_demo.dashboard --metrics uc3_selection.json
python -m harness.maintie_demo.dashboard --metrics uc3_selection.json
```

Metrics in the selection that a run did not measure are listed under the dashboard table with what they would need, so the selection doubles as a coverage check.

The selection stays in the page (and in the browser's local storage when it allows it). Nothing is sent anywhere.

## Editing and rebuilding

`src/` holds the catalogue as designed (`index.html`, `styles.css`, `catalogue.js`, `catalogue-data.js`, `assets/logo.png`): visible text in `index.html`, colours and layout in `styles.css`, metrics and tooltips in `catalogue-data.js`, behaviour in `catalogue.js`. `picker/` holds the picking feature. After editing either, rebuild the two single-file pages:

```bash
python docs/catalogue/build_catalogue.py          # write the pages
python docs/catalogue/build_catalogue.py --check  # fail when they are out of date (a test runs this)
```

The picker is applied to `src/` with explicit patches; if an edit to `src/` moves one of them, the build fails and says which.

`src/catalogue-data.js` lists the metrics shown. It currently has 47, and the registry has more (the diagnostic-dialogue metrics of the DiagFlowBench demo are not in it yet), so those cannot be picked until they are added there.
