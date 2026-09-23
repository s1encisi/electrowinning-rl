# Data and model contract

The repository contains no production measurements or pretrained plant models.
`ewrl generate-data --condition 1 --output runs/data` creates a synthetic example.
The generator is fully defined in `process.synthetic_response`; its random seed,
noise scales and feature domain are explicit.

## Local CSV input

`demo` and `train-surrogate` accept `--data path/to/data.csv`. Place private files
under `data/private/`, which is ignored by Git. They are read locally, with no
network upload. Derived artifacts under `runs/` are also ignored.

| Mode | Decision columns in order |
|---|---|
| 1 | `cu_in, temperature_a, current_a, flow_a, duration` |
| 2 | `cu_in, temperature_b, current_b, flow_b, duration` |
| 3 | `cu_in, temperature_a, current_a, flow_a, temperature_b, current_b, flow_b, duration` |

Every CSV also requires targets `cu_out, as_out, voltage`. For the default chronological
split, `sample_index` must be a finite unique numeric ordering of observation time.
The index is not a model input. Sort/encode actual timestamps before supplying it;
the program cannot infer observation time from arbitrary row IDs.

Units: copper/arsenic in g/L, voltage in V, temperature in °C, current in A,
flow in m³/h, duration in h. Values must be within the declared decision domain.
Labels must be finite and nonnegative; entirely missing features and infinite
inputs are rejected. Other missing feature values are median-imputed within the
training pipeline. At least 100 unique decision rows are required. Extra columns
are ignored and are never inferred as features.

The serial voltage label represents a common/average unit voltage. A different
voltage definition requires an explicit model and process-contract change.

The pipeline retains the first occurrence of repeated decision vectors. If repeated
measurements represent meaningful batch or temporal dynamics, adapt the data/split
contract rather than interpreting this demonstration's static surrogate as a
dynamic model.

## Generated artifacts

`surrogate.joblib` stores the model, mode, ordered features, source type, dataset
hash and training configuration. It is version-dependent and must be regenerated
after dependency changes. A user CSV is labeled `user-csv` in all reports; the
synthetic oracle comparison is then disabled. No industrial validation claim is
automatically attached to a user-supplied dataset.
