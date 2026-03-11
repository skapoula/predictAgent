"""Domain exceptions for predictagent."""


class PredictAgentError(Exception):
    """Base class for all predictagent errors."""


class SchemaValidationError(PredictAgentError):
    """Input data is missing required columns or has invalid types."""


class IngestionError(PredictAgentError):
    """Failure during raw data loading or rollup."""


class FeatureEngineeringError(PredictAgentError):
    """Failure during feature computation."""


class SequencerError(PredictAgentError):
    """Failure during sequence building or splitting."""


class TrainingError(PredictAgentError):
    """Failure during model training."""


class RegistryError(PredictAgentError):
    """Failure reading from or writing to the model registry."""


class ModelNotFoundError(RegistryError):
    """No model found for the requested cell name and version."""


class InferenceError(PredictAgentError):
    """Failure during model inference."""


class InsufficientDataError(InferenceError):
    """Caller provided fewer rows than lookback_steps requires."""
