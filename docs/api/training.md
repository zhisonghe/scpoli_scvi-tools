# Training internals

These classes are used by `SCPOLI.train`; most users only set their options through `train`
(or `plan_kwargs` / `datasplitter_kwargs`).

::: scpolivi.train.SCPoliTrainingPlan
    options:
      members:
        - initialize_prototypes
        - train_latent

::: scpolivi.train.SCPoliDataSplitter

::: scpolivi.train.SCPoliEarlyStopping

::: scpolivi.train.scpoli_split
