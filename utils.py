import json, logging

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_llm_output(output):
    try:
        parsed = json.loads(output)
        assert "issues" in parsed
        assert "recommendations" in parsed
        return True, parsed
    except (json.JSONDecodeError, AssertionError):
        return False, None
