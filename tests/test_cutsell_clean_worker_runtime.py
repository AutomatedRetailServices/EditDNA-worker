from cutsell_worker.config import load_runtime_config


def test_runtime_config_reuses_existing_infrastructure_variable_names_without_enabling_external_brain():
    config = load_runtime_config({
        "REDIS_URL": "redis://example",
        "OPENAI_API_KEY": "secret",
        "AWS_ACCESS_KEY_ID": "key",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "us-east-1",
        "S3_BUCKET": "bucket",
        "RUNPOD_API_KEY": "runpod",
        "RUNPOD_TEMPLATE_ID": "template",
    })
    assert config.queue_ready is True
    assert config.openai_api_key_present is True
    assert config.brain_backend == "runpod_local"
    assert config.semantic_ready is False
    assert config.visual_ready is False
    assert config.clean_cut_judge_ready is False
    assert config.storage_ready is True
    assert config.runpod_api_key_present is True
    assert config.runpod_template_id == "template"
    assert config.asr_model == "medium"


def test_runtime_config_reports_missing_secrets_without_exposing_values():
    config = load_runtime_config({})
    assert config.queue_ready is False
    assert config.semantic_ready is False
    assert config.visual_ready is False
    assert config.storage_ready is False
    assert config.runpod_api_key_present is False
    assert config.brain_backend == "runpod_local"
