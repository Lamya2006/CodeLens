"""Seed AI-style baseline snippets into Pinecone."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.pinecone_tool import PineconeStore


AI_SNIPPETS = [
    """from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/users/<string:username>", methods=["GET"])
def get_user_profile(username: str):
    \"\"\"Return a single user profile with defensive error handling.\"\"\"
    try:
        if not username.strip():
            return jsonify({"error": "Username is required."}), 400
        profile = {"username": username, "active": True}
        return jsonify(profile), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
""",
    """def sort_numbers_ascending(values: list[int]) -> list[int]:
    \"\"\"Return a sorted copy of the provided numeric values.\"\"\"
    try:
        return sorted(values)
    except Exception as exc:
        raise ValueError(f"Unable to sort values: {exc}") from exc
""",
    """class CustomerRecord:
    \"\"\"Represent a customer record with readable string helpers.\"\"\"

    def __init__(self, customer_name: str, account_balance: float) -> None:
        \"\"\"Initialize the customer record.\"\"\"
        self.customer_name = customer_name
        self.account_balance = account_balance

    @property
    def is_in_good_standing(self) -> bool:
        \"\"\"Return whether the customer has a non-negative balance.\"\"\"
        return self.account_balance >= 0

    def __repr__(self) -> str:
        \"\"\"Return a detailed developer representation.\"\"\"
        return f"CustomerRecord(customer_name={self.customer_name!r}, account_balance={self.account_balance!r})"

    def __str__(self) -> str:
        \"\"\"Return a user-friendly summary string.\"\"\"
        return f"{self.customer_name} ({self.account_balance:.2f})"
""",
    """from typing import Iterator

class ManagedFile:
    \"\"\"Provide a context manager wrapper for file access.\"\"\"

    def __init__(self, file_path: str, mode: str) -> None:
        \"\"\"Store file configuration for later use.\"\"\"
        self.file_path = file_path
        self.mode = mode
        self.handle = None

    def __enter__(self):
        \"\"\"Open the file and return the handle.\"\"\"
        self.handle = open(self.file_path, self.mode, encoding="utf-8")
        return self.handle

    def __exit__(self, exc_type, exc, traceback) -> bool:
        \"\"\"Close the file handle and do not suppress exceptions.\"\"\"
        if self.handle is not None:
            self.handle.close()
        return False
""",
    """from collections.abc import Callable

def retry_decorator_factory(max_attempts: int) -> Callable:
    \"\"\"Create a decorator that retries a callable a fixed number of times.\"\"\"
    def decorator(function: Callable) -> Callable:
        \"\"\"Wrap the provided function with retry behavior.\"\"\"
        def wrapper(*args, **kwargs):
            \"\"\"Execute the wrapped function with retries.\"\"\"
            last_exception = None
            for _ in range(max_attempts):
                try:
                    return function(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc
            raise RuntimeError(f"Operation failed after retries: {last_exception}") from last_exception
        return wrapper
    return decorator
""",
    """def calculate_total_revenue(order_values: list[float]) -> float:
    \"\"\"Calculate the total revenue from a collection of order values.\"\"\"
    try:
        return sum(float(order_value) for order_value in order_values)
    except Exception as exc:
        raise ValueError(f"Failed to calculate total revenue: {exc}") from exc
""",
    """def validate_user_input(payload: dict[str, str]) -> bool:
    \"\"\"Validate that required user input fields are present and non-empty.\"\"\"
    try:
        required_fields = ["email", "password"]
        return all(str(payload.get(field, "")).strip() for field in required_fields)
    except Exception as exc:
        raise ValueError(f"Invalid payload supplied: {exc}") from exc
""",
    """def build_lookup_dictionary(records: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    \"\"\"Convert a sequence of records into a dictionary keyed by identifier.\"\"\"
    try:
        return {record["id"]: record for record in records if "id" in record}
    except Exception as exc:
        raise ValueError(f"Unable to build lookup dictionary: {exc}") from exc
""",
    """def normalize_text_lines(lines: list[str]) -> list[str]:
    \"\"\"Trim whitespace from lines and remove empty entries.\"\"\"
    try:
        return [line.strip() for line in lines if line.strip()]
    except Exception as exc:
        raise ValueError(f"Unable to normalize text lines: {exc}") from exc
""",
    """def generate_fibonacci_sequence(item_count: int) -> list[int]:
    \"\"\"Generate a Fibonacci sequence with the requested number of items.\"\"\"
    try:
        sequence: list[int] = []
        previous_value, current_value = 0, 1
        for _ in range(item_count):
            sequence.append(previous_value)
            previous_value, current_value = current_value, previous_value + current_value
        return sequence
    except Exception as exc:
        raise ValueError(f"Unable to generate Fibonacci sequence: {exc}") from exc
""",
    """def merge_configuration(primary_config: dict[str, str], secondary_config: dict[str, str]) -> dict[str, str]:
    \"\"\"Merge two configuration dictionaries into a new dictionary.\"\"\"
    try:
        merged_config = dict(primary_config)
        merged_config.update(secondary_config)
        return merged_config
    except Exception as exc:
        raise ValueError(f"Unable to merge configuration: {exc}") from exc
""",
    """def compute_average_score(scores: list[float]) -> float:
    \"\"\"Compute the arithmetic mean of the supplied numeric scores.\"\"\"
    try:
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
    except Exception as exc:
        raise ValueError(f"Unable to compute average score: {exc}") from exc
""",
    """def make_slug_from_title(title: str) -> str:
    \"\"\"Create a URL slug from the provided title.\"\"\"
    try:
        return title.strip().lower().replace(" ", "-")
    except Exception as exc:
        raise ValueError(f"Unable to create slug: {exc}") from exc
""",
    """def count_words_by_frequency(text: str) -> dict[str, int]:
    \"\"\"Count word frequency for a block of text.\"\"\"
    try:
        frequency_map: dict[str, int] = {}
        for token in text.split():
            frequency_map[token] = frequency_map.get(token, 0) + 1
        return frequency_map
    except Exception as exc:
        raise ValueError(f"Unable to count words: {exc}") from exc
""",
    """def remove_empty_values(values: list[str]) -> list[str]:
    \"\"\"Return only non-empty string values.\"\"\"
    try:
        return [value for value in values if value]
    except Exception as exc:
        raise ValueError(f"Unable to remove empty values: {exc}") from exc
""",
    """def filter_active_users(users: list[dict[str, object]]) -> list[dict[str, object]]:
    \"\"\"Return only users marked as active.\"\"\"
    try:
        return [user for user in users if bool(user.get("active"))]
    except Exception as exc:
        raise ValueError(f"Unable to filter active users: {exc}") from exc
""",
    """function calculateTotalRevenue(orderValues) {
  /** Calculate the total revenue from the supplied order values. */
  try {
    return orderValues.reduce((accumulator, currentValue) => accumulator + Number(currentValue), 0);
  } catch (error) {
    throw new Error(`Failed to calculate total revenue: ${error}`);
  }
}
""",
    """function validateUserInput(payload) {
  /** Validate the user input payload before processing it. */
  try {
    return Boolean(payload?.email && payload?.password);
  } catch (error) {
    throw new Error(`Failed to validate user input: ${error}`);
  }
}
""",
    """function createSortedList(values) {
  /** Return a sorted shallow copy of the provided list. */
  try {
    return [...values].sort((leftValue, rightValue) => leftValue - rightValue);
  } catch (error) {
    throw new Error(`Failed to create a sorted list: ${error}`);
  }
}
""",
    """class ApplicationConfig {
  /** Represent application configuration with formatted output helpers. */
  constructor(applicationName, applicationVersion) {
    /** Initialize the configuration instance. */
    this.applicationName = applicationName;
    this.applicationVersion = applicationVersion;
  }

  toString() {
    /** Return a user-facing string summary. */
    return `${this.applicationName} v${this.applicationVersion}`;
  }
}
""",
    """function createContextManager(resourceFactory) {
  /** Create a lightweight context-style wrapper around a resource factory. */
  return {
    async run(callback) {
      /** Acquire the resource, invoke the callback, and always release the resource. */
      const resource = await resourceFactory();
      try {
        return await callback(resource);
      } catch (error) {
        throw new Error(`Context manager execution failed: ${error}`);
      } finally {
        if (resource?.close) {
          await resource.close();
        }
      }
    }
  };
}
""",
    """function createRetryDecorator(maxAttempts) {
  /** Create a decorator factory that retries asynchronous operations. */
  return function retryDecorator(targetFunction) {
    /** Wrap the target function with retry logic. */
    return async function wrappedFunction(...args) {
      /** Execute the wrapped function and retry on failure. */
      let lastError = null;
      for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        try {
          return await targetFunction(...args);
        } catch (error) {
          lastError = error;
        }
      }
      throw new Error(`Operation failed after retries: ${lastError}`);
    };
  };
}
""",
    """function mergeConfiguration(primaryConfiguration, secondaryConfiguration) {
  /** Merge two configuration objects into a new object. */
  try {
    return { ...primaryConfiguration, ...secondaryConfiguration };
  } catch (error) {
    throw new Error(`Failed to merge configuration: ${error}`);
  }
}
""",
    """function normalizeTextLines(lines) {
  /** Trim whitespace from lines and remove empty values. */
  try {
    return lines.map((line) => line.trim()).filter((line) => line.length > 0);
  } catch (error) {
    throw new Error(`Failed to normalize text lines: ${error}`);
  }
}
""",
    """function findRecordByIdentifier(records, identifier) {
  /** Retrieve the first record whose identifier matches the provided value. */
  try {
    return records.find((record) => record?.id === identifier) ?? null;
  } catch (error) {
    throw new Error(`Failed to find record by identifier: ${error}`);
  }
}
""",
]


def seed_ai_baseline() -> None:
    store = PineconeStore()
    chunks = []
    for index, snippet in enumerate(AI_SNIPPETS, start=1):
        language = "python" if snippet.lstrip().startswith(("from ", "def ", "class ")) else "javascript"
        chunks.append(
            {
                "id": f"ai-{index}",
                "text": snippet,
                "metadata": {
                    "file_path": f"ai_snippet_{index}.txt",
                    "language": language,
                    "repo": "ai-baseline",
                    "chunk_type": "file",
                    "symbol_name": f"ai_snippet_{index}",
                    "type": "ai",
                },
            }
        )
    store.upsert_chunks(chunks, namespace="ai-baseline")


if __name__ == "__main__":
    seed_ai_baseline()
    print("Seeded ai-baseline namespace.")
