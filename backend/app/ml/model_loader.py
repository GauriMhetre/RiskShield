"""
Model loader for RiskShield fraud detection.

This module provides the ModelLoader class, which loads and manages the trained
XGBoost model artifact (model + metadata) for use in the FastAPI scoring endpoint.

WHY A CLASS RATHER THAN STANDALONE FUNCTIONS?
==============================================

ModelLoader is built as a class (instantiated once at app startup, reused across requests)
rather than as standalone functions (called fresh each time) for these reasons:

1. PERFORMANCE: Loading a model from disk (~700 KB pickle file) takes time and I/O.
   Creating a single ModelLoader instance at startup and reusing it across all
   requests avoids repeatedly reloading the model. With thousands of concurrent
   requests/second, this difference is critical for latency.

2. RESOURCE EFFICIENCY: A model object in memory is shared across all requests.
   Creating 1000 separate model instances (one per request) would consume 1000x
   the memory and CPU. The class pattern keeps resource usage constant regardless
   of request volume.

3. TESTABILITY: A class with an __init__ method makes dependency injection and
   mocking easier. Unit tests can instantiate a ModelLoader with a fake model
   or mock load_model_artifact() without side effects affecting other tests.

4. EXPLICITNESS: Calling loader.predict_proba() makes it clear that the model
   is already loaded and ready. Calling a function like predict_with_model(feature_dict)
   leaves it ambiguous whether the function loads the model each time.

Pattern in Phase 4's FastAPI main.py (future):
  from backend.app.ml.model_loader import ModelLoader

  # At app startup (runs once):
  loader = ModelLoader(model_version="model_v1", save_dir="ml/models")

  @app.post("/score")
  def score_transaction(request: ScoreRequest):
      features = compute_features(txn, profile)
      fraud_prob = loader.predict_proba(features)
      threshold = loader.get_threshold()
      flagged = fraud_prob >= threshold
      return ScoreResponse(...)

This ensures the model is loaded once, used many times, and never reloaded.
"""

import numpy as np
from ml.train import load_model_artifact


class ModelLoader:
    """
    Loads and manages a trained fraud detection model.

    Instantiate this class once at application startup (e.g., in FastAPI's lifespan
    event) to load the model and metadata from disk, then reuse across all requests.

    Example:
        loader = ModelLoader(model_version="model_v1", save_dir="ml/models")
        fraud_prob = loader.predict_proba(feature_dict)
        threshold = loader.get_threshold()
        flagged = fraud_prob >= threshold
    """

    def __init__(self, model_version: str = "model_v1", save_dir: str = "ml/models"):
        """
        Initialize the model loader by loading the model artifact from disk.

        Args:
            model_version: Version identifier for the model (default "model_v1").
                         This becomes {model_version}.pkl and {model_version}_metadata.json
            save_dir: Directory where model artifacts are saved (default "ml/models").
                     Typically contains model_v1.pkl, model_v1_metadata.json, etc.

        Raises:
            FileNotFoundError: If either the model or metadata file is missing.
                             Message tells the user to run ml/train.py to generate them.
        """
        try:
            self.model, self.metadata = load_model_artifact(
                model_version=model_version,
                save_dir=save_dir,
            )
        except FileNotFoundError as e:
            # Provide a helpful error message pointing to how to generate the model
            raise FileNotFoundError(
                f"Cannot load model artifact '{model_version}' from '{save_dir}':\n{str(e)}\n\n"
                f"Please generate the model first by running:\n  python ml/train.py\n\n"
                f"This will train the model and save it to {save_dir}/{model_version}.pkl "
                f"and {save_dir}/{model_version}_metadata.json"
            ) from e

    def predict_proba(self, feature_dict: dict) -> float:
        """
        Predict fraud probability for a single transaction given its features.

        Args:
            feature_dict: Dictionary with exactly 10 features from compute_features().
                         Keys must match self.metadata['feature_columns'].
                         Example:
                         {
                             'txn_count_1h': 0,
                             'txn_count_24h': 0,
                             'amount_zscore': 20.0,
                             'amount_ratio_to_avg': 5.0,
                             'device_mismatch': 1,
                             'country_mismatch': 1,
                             'geo_distance_km': 1155.3,
                             'amount': 500.0,
                             'hour_of_day': 14,
                             'day_of_week': 5
                         }

        Returns:
            Float in [0, 1] representing the model's predicted probability of fraud.

        Raises:
            ValueError: If feature_dict is missing any keys required by the model.
                       The error message lists all missing keys.

        WHY METADATA['FEATURE_COLUMNS']?
        ================================

        This method MUST use self.metadata['feature_columns'] to order the feature
        array before calling model.predict_proba(). Here's why NOT doing this is a
        critical bug:

        RISK: Trusting dict insertion order
        -----------------------------------
        While Python dicts preserve insertion order (3.7+), the model was trained on a
        specific column order FIXED AT TRAINING TIME. If the dict keys happen to be in
        a different order than the training columns (e.g., because an upstream function
        constructs the dict in a different order), the feature array will be misaligned.

        SILENT BUG: The model will still produce a prediction (no error), but it will
        be WRONG. Example:
          - Training: features were ordered [feat_a, feat_b, feat_c]
          - Model learned: "if feat_a > 5, high fraud probability"
          - At inference: dict is {'feat_c': 1, 'feat_b': 2, 'feat_a': 10}
          - If we trusted dict order: [1, 2, 10] → model sees [1, 2, 10]
          - Model checks "if feat_a > 5" but feat_a is now in position 2, so it sees 2 (wrong!)
          - Result: low fraud probability even though feat_a is 10 (the actual indicator)

        This is the WORST KIND OF BUG because:
          1. No error is raised (silent failure)
          2. Predictions are plausibly wrong (model runs, but output is garbage)
          3. Hard to debug (model code looks correct, but behavior is wrong)

        SOLUTION: Always construct the array using self.metadata['feature_columns']:
          feature_array = [feature_dict[col] for col in self.metadata['feature_columns']]
          
        This ensures the model ALWAYS sees features in the exact order it was trained on,
        regardless of dict construction order upstream.
        """
        # Get the expected feature column order from metadata (this order is fixed from training)
        expected_columns = self.metadata["feature_columns"]

        # Validate that all required features are present
        missing_keys = set(expected_columns) - set(feature_dict.keys())
        if missing_keys:
            raise ValueError(
                f"Missing required features: {sorted(missing_keys)}. "
                f"Expected keys: {sorted(expected_columns)}. "
                f"Provided keys: {sorted(feature_dict.keys())}"
            )

        # CRITICAL: Construct feature array using the training column order, not dict order
        feature_array = np.array([feature_dict[col] for col in expected_columns])

        # Reshape for sklearn (needs 2D array with shape (1, n_features) for a single sample)
        feature_array = feature_array.reshape(1, -1)

        # Get probability prediction for the fraud class (class 1)
        fraud_probability = self.model.predict_proba(feature_array)[0, 1]

        return float(fraud_probability)

    def get_threshold(self) -> float:
        """
        Get the decision threshold for classifying transactions as fraud or legitimate.

        Returns:
            Float in [0, 1] representing the threshold. Transactions with
            fraud_probability >= threshold are flagged as fraud.

        Note:
            This threshold is tuned based on business requirements (e.g., precision
            floor) and is fixed at model training time. It is NOT recomputed per-request.
        """
        return self.metadata["threshold"]

    def get_model_version(self) -> str:
        """
        Get the version identifier of the loaded model.

        Returns:
            String identifier (e.g., "model_v1") of the model currently loaded.

        Use case:
            Log or return the model version in API responses so callers know which
            model was used to score their transaction. This is useful for debugging
            and for tracking which model variant is deployed.
        """
        return self.metadata["model_version"]
