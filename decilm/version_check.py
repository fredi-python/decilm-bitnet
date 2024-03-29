import transformers
from packaging import version

MIN_VERSION = "4.35.2"


def check_transformers_version():
    if version.parse(transformers.__version__) < version.parse(MIN_VERSION):
        raise ImportError(
            f"You are using transformers=={transformers.__version__}, but transformers>={MIN_VERSION} is required to use DeciLM. Please upgrade transformers."
        )
