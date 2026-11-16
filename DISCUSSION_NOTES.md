# ANA675 Week 2 discussion notes

## Why our reconstruction has fewer samples

Verified for the **one-hour forecast** using the 47 published event windows, provisionally interpreted as UTC, with rounded durations:

- 9,495 candidate input/target windows before checking missing readings. The paper reports 9,484 usable samples; the 11-window difference at this stage may reflect its precise event endpoints versus the rounded published durations. This cause is not confirmed.
- 145 distinct required hourly observations are blank: 61 Casalmaggiore, 7 Colorno, 77 Ponte Verdi.
- One additional Ponte Verdi observation has conflicting source values (1.95 versus 2.03 meters) and remains unresolved.
- Those **146 distinct problematic observations invalidate 1,159 candidate windows**, because each observation is reused in multiple twelve-hour inputs. These counts are not 1,159 different missing raw readings.
- 8,336 windows remain before smoothing.
- Requiring complete native neighbors for the centered smoothing comparison excludes another **813 windows**, leaving **7,523** paired smoothed/unsmoothed one-hour examples. Both versions are evaluated on those same available examples.
- Raw ZIPs and CSVs are unchanged. We omit unusable training windows; we do not delete the original data or fill missing values.
- These are horizon-specific counts. Do not apply them to every forecast horizon. Details are in results/paper_protocol/coverage.csv and missing_window_audit.csv.

## What we can and cannot say about the authors

The paper describes smoothing ten-minute Ponte Verdi data before hourly sampling, but the reviewed methods do not specify how missing values were handled. We have not established whether the authors encountered the same missing records. They might have had a more complete historical export, interpolated missing values, used another unreported treatment, or defined exact windows differently. These are possibilities, not findings. Smoothing by itself does not establish that missing values were filled.

Suggested discussion wording:

> Our reconstruction excluded any input/target window requiring a missing or conflicting measurement. For the one-hour forecast, 145 blank hourly observations and one conflicting observation affected 1,159 windows, leaving 8,336 examples before smoothing. Requiring complete neighborhoods for our adapted smoothing procedure reduced the paired comparison to 7,523 examples. The paper reports 9,484 examples but does not specify its missing-value treatment. Differences in the available archive, precise event boundaries, or gap handling may explain part of this discrepancy; we cannot determine which from the reported methods.

## Current interpretation

This is a closest-available approximation with documented differences, not an exact replication. The paper's 20 event assignments are preserved for the new experiment. Native resolution, smoothing quadrature, event timezone, exact endpoints and physical normalization details remain unresolved. Centered smoothing uses later observations in archived records and is not a causal real-time forecast.
