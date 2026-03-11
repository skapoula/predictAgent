import pytest
from predictagent.exceptions import (
    PredictAgentError,
    SchemaValidationError,
    IngestionError,
    FeatureEngineeringError,
    SequencerError,
    TrainingError,
    RegistryError,
    ModelNotFoundError,
    InferenceError,
    InsufficientDataError,
)


@pytest.mark.unit
def test_all_exceptions_are_subclasses_of_base():
    for exc_cls in [
        SchemaValidationError,
        IngestionError,
        FeatureEngineeringError,
        SequencerError,
        TrainingError,
        RegistryError,
        InferenceError,
    ]:
        assert issubclass(exc_cls, PredictAgentError)


@pytest.mark.unit
def test_model_not_found_is_registry_error():
    assert issubclass(ModelNotFoundError, RegistryError)


@pytest.mark.unit
def test_insufficient_data_is_inference_error():
    assert issubclass(InsufficientDataError, InferenceError)


@pytest.mark.unit
def test_raise_and_catch_as_base():
    with pytest.raises(PredictAgentError):
        raise IngestionError("bad data")
