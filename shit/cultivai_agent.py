"""
CultivAI — Plant Disease Specialist Agent
==========================================
Wraps the PlantVillage ResNet18 inference service as a CultivAI-compatible
specialist agent. Produces a standard AgentOutput the Coordinator can consume
directly — no knowledge of PyTorch or image processing required.

The underlying FastAPI service (app/main.py) and the model (app/inference.py)
are completely unchanged. This file is a thin coordination layer on top.

Usage (standalone):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # major_project/

    from disease_detection.cultivai_agent import PlantDiseaseAgent

    agent = PlantDiseaseAgent()
    print(agent.health_check())           # True if model weights present

    with open("leaf.jpg", "rb") as f:
        result = agent.run(image_bytes=f.read(), crop_context="tomato")

    print(result.status)                  # AgentStatus.SUCCESS / INCONCLUSIVE …
    print(result.summary)                 # "Possible disease: Tomato Early Blight (87%)"
    import json
    print(json.dumps(result.to_dict(), indent=2))
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
# Resolve roots so this file works regardless of the caller's cwd.
_AGENT_DIR = Path(__file__).parent.resolve()          # disease_detection/
_PROJECT_ROOT = _AGENT_DIR.parent.resolve()           # major_project/

# Make cultivai/ importable
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cultivai.contracts import AgentOutput, AgentStatus
from cultivai.base_agent import BaseAgent

# ── Constants ─────────────────────────────────────────────────────────────────

# Top-1 softmax probability below which we consider the result inconclusive.
# PlantVillage-trained models tend to be overconfident; 0.40 is a conservative
# threshold that avoids presenting a low-quality guess as a diagnosis.
_CONFIDENCE_THRESHOLD = 0.40

_SOURCES = [
    "PlantVillage dataset — https://www.kaggle.com/datasets/emmarex/plantdisease",
    "ResNet18 trained from scratch — 94.56% test accuracy, Macro F1 94.6%",
    "Live inference API — https://plantvillage-resnet18.onrender.com",
]

_LIMITATIONS = [
    "Model trained on PlantVillage (controlled lab images). Performance on "
    "real field images may be lower.",
    "Covers 15 classes across Pepper, Potato, and Tomato only. Other crops "
    "or diseases are not supported.",
    "Confidence scores are softmax probabilities, not calibrated diagnostic "
    "certainties.",
    "Nutrient deficiencies, pest damage, abiotic stress, and multiple "
    "simultaneous problems can visually resemble disease symptoms.",
    "Image quality and lighting affect accuracy; blurry or low-contrast "
    "images may yield unreliable results.",
    "This output is a preliminary screening aid. Consult a qualified "
    "agronomist for serious or uncertain cases before applying treatment.",
]


class PlantDiseaseAgent(BaseAgent):
    """
    Specialist agent for plant disease identification from crop leaf images.

    Wraps the existing ResNet18 model (app/inference.py) with:
      - Lazy model loading (loaded once on first run() call).
      - Inconclusive detection when top-1 confidence < 40%.
      - Crop mismatch warning when the farmer's stated crop disagrees with
        the model's predicted crop genus.
      - Separate messaging path for healthy predictions.
      - Graceful error handling — never raises exceptions to the Coordinator.

    Input kwargs (for run()):
        image_bytes  (bytes)        Raw image bytes (JPEG, PNG, etc.)
                                    Takes priority over image_path.
        image_path   (str | Path)   Path to image file. Used when
                                    image_bytes is not provided.
        top_k        (int)          Number of top predictions to return.
                                    Default: 3.
        crop_context (str)          Farmer-stated crop type (e.g. "tomato").
                                    Used for cross-checking and warnings.

    Output (AgentOutput.data keys):
        predictions      list[dict]  Top-k results: [{"label": …, "prob": …}]
        top_label        str         Raw label of the best prediction.
        top_confidence   float       Softmax probability of the best prediction.
        crop_context     str | None  Echoed from input for traceability.
        classes_supported list[str]  All 15 classes the model knows.
    """

    def __init__(self, model_dir: Optional[str] = None) -> None:
        """
        Args:
            model_dir: Directory containing plantvillage.pt and labels.json.
                       Defaults to disease_detection/app/.
        """
        self._model_dir = Path(model_dir) if model_dir else _AGENT_DIR / "app"
        self._model = None
        self._class_names = None
        self._transform = None

    @property
    def name(self) -> str:
        return "plant_disease"

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True when model weights and class labels are on disk."""
        return (
            (self._model_dir / "plantvillage.pt").exists()
            and (self._model_dir / "labels.json").exists()
        )

    # ── Model loading ─────────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """
        Lazy-load the ResNet18 weights on the first inference call.

        Temporarily changes the working directory to disease_detection/ so
        that the relative paths inside app/inference.py resolve correctly,
        then restores the original cwd regardless of outcome.
        """
        if self._model is not None:
            return

        original_cwd = os.getcwd()
        try:
            os.chdir(_AGENT_DIR)
            # Add disease_detection/ to sys.path so `from app.inference import …` works
            if str(_AGENT_DIR) not in sys.path:
                sys.path.insert(0, str(_AGENT_DIR))
            from app.inference import load_model
            self._model, self._class_names, self._transform = load_model()
        finally:
            os.chdir(original_cwd)

    # ── Core run ──────────────────────────────────────────────────────────────

    def run(
        self,
        image_bytes: Optional[bytes] = None,
        image_path: Optional[str] = None,
        top_k: int = 3,
        crop_context: Optional[str] = None,
        **kwargs,
    ) -> AgentOutput:
        """
        Classify a plant leaf image and return a structured diagnosis.

        Steps:
            1. Validate that at least one image source was provided.
            2. Load image bytes from disk if image_path was given.
            3. Lazy-load the model weights (only on the very first call).
            4. Run inference via app/inference.py:predict_image().
            5. Interpret confidence and crop context → status + summary.

        Returns:
            AgentOutput with status SUCCESS, INCONCLUSIVE, MISSING_INPUT,
            or ERROR.
        """

        # ── 1. Input validation ───────────────────────────────────────────────
        if image_bytes is None and image_path is None:
            return self._missing_input_output(
                "No image provided. Please upload a crop leaf image for "
                "disease analysis."
            )

        # ── 2. Load from disk if needed ───────────────────────────────────────
        if image_bytes is None:
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
            except (FileNotFoundError, OSError) as exc:
                return self._error_output(
                    "Could not read the provided image file. "
                    "Please check the path and try again.",
                    exc,
                )

        # ── 3 & 4. Model inference ────────────────────────────────────────────
        try:
            self._ensure_model_loaded()

            original_cwd = os.getcwd()
            try:
                os.chdir(_AGENT_DIR)
                from app.inference import predict_image
                predictions = predict_image(
                    image_bytes,
                    self._model,
                    self._transform,
                    self._class_names,
                    k=top_k,
                )
            finally:
                os.chdir(original_cwd)

        except Exception as exc:
            return self._error_output(
                "Model inference failed. The image may be corrupted, empty, "
                "or in an unsupported format. Please try a different image.",
                exc,
            )

        # ── 5. Interpret results ──────────────────────────────────────────────
        top = predictions[0]
        top_label: str = top["label"]
        top_confidence: float = top["prob"]

        # Human-readable label: "Tomato_Early_blight" → "Tomato Early Blight"
        display_label = top_label.replace("__", " — ").replace("_", " ").title()

        # Optional crop cross-check
        limitations = list(_LIMITATIONS)
        if crop_context:
            predicted_genus = top_label.split("_")[0].lower()
            stated = crop_context.strip().lower()
            matched_crop = top_label.split("_")[0]
            if stated and predicted_genus and (
                stated not in predicted_genus and predicted_genus not in stated
            ):
                limitations.insert(
                    0,
                    f"Crop mismatch: you indicated '{crop_context}' but the "
                    f"model's best match is for '{matched_crop}'. "
                    "This may indicate an unsupported crop or a "
                    "misidentification — treat this result with extra caution.",
                )

        # Determine status and build summary
        is_healthy = "healthy" in top_label.lower()

        if top_confidence < _CONFIDENCE_THRESHOLD:
            status = AgentStatus.INCONCLUSIVE
            summary = (
                f"The image did not closely match any known disease pattern "
                f"(best match: '{display_label}' at "
                f"{top_confidence:.0%} confidence). "
                "Please try a clearer, well-lit, close-up image of the "
                "affected leaf area."
            )
        elif is_healthy:
            status = AgentStatus.SUCCESS
            summary = (
                f"The leaf appears healthy ({display_label}, "
                f"{top_confidence:.0%} confidence). "
                "No disease pattern was detected from this image."
            )
        else:
            status = AgentStatus.SUCCESS
            summary = (
                f"Possible disease detected: {display_label} "
                f"({top_confidence:.0%} confidence). "
                "Verify with a local agronomist before applying any treatment."
            )

        return AgentOutput(
            agent=self.name,
            status=status,
            summary=summary,
            data={
                "predictions": predictions,
                "top_label": top_label,
                "top_confidence": round(top_confidence, 4),
                "crop_context": crop_context,
                "classes_supported": self._class_names,
            },
            sources=_SOURCES,
            limitations=limitations,
        )
