"""
CultivAI — Plant Disease Specialist Agent (ResNet50 / 38 Classes)
==================================================================
Wraps the Plant Disease Detection ResNet50 model (plant_disease_detection)
as a CultivAI-compatible specialist agent. Produces a standard AgentOutput
the Coordinator / FastAPI server can consume directly.

Covers 38 classes across 14 crops:
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato,
Raspberry, Soybean, Squash, Strawberry, Tomato.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_AGENT_DIR = Path(__file__).parent.resolve()          # plant_disease_detection/
_PROJECT_ROOT = _AGENT_DIR.parent.resolve()           # major_project/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cultivai.contracts import AgentOutput, AgentStatus
from cultivai.base_agent import BaseAgent

# ── 38 Disease & Healthy Classes ──────────────────────────────────────────────
CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

CLASS_DESCRIPTIONS = {
    'Apple___Apple_scab': "Apple plant affected by Apple scab (fungal)",
    'Apple___Black_rot': "Apple plant affected by Black rot (fungal)",
    'Apple___Cedar_apple_rust': "Apple plant affected by Cedar apple rust (fungal)",
    'Apple___healthy': "Apple plant — healthy and disease-free",
    'Blueberry___healthy': "Blueberry plant — healthy and disease-free",
    'Cherry_(including_sour)___Powdery_mildew': "Cherry plant affected by Powdery mildew",
    'Cherry_(including_sour)___healthy': "Cherry plant — healthy and disease-free",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Corn (maize) affected by Gray leaf spot",
    'Corn_(maize)___Common_rust_': "Corn (maize) affected by Common rust",
    'Corn_(maize)___Northern_Leaf_Blight': "Corn (maize) affected by Northern Leaf Blight",
    'Corn_(maize)___healthy': "Corn (maize) — healthy and disease-free",
    'Grape___Black_rot': "Grape plant affected by Black rot",
    'Grape___Esca_(Black_Measles)': "Grape plant affected by Esca (Black Measles)",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Grape plant affected by Leaf blight (Isariopsis Leaf Spot)",
    'Grape___healthy': "Grape plant — healthy and disease-free",
    'Orange___Haunglongbing_(Citrus_greening)': "Orange plant affected by Huanglongbing (Citrus greening)",
    'Peach___Bacterial_spot': "Peach plant affected by Bacterial spot",
    'Peach___healthy': "Peach plant — healthy and disease-free",
    'Pepper,_bell___Bacterial_spot': "Bell pepper plant affected by Bacterial spot",
    'Pepper,_bell___healthy': "Bell pepper plant — healthy and disease-free",
    'Potato___Early_blight': "Potato plant affected by Early blight",
    'Potato___Late_blight': "Potato plant affected by Late blight",
    'Potato___healthy': "Potato plant — healthy and disease-free",
    'Raspberry___healthy': "Raspberry plant — healthy and disease-free",
    'Soybean___healthy': "Soybean plant — healthy and disease-free",
    'Squash___Powdery_mildew': "Squash plant affected by Powdery mildew",
    'Strawberry___Leaf_scorch': "Strawberry plant affected by Leaf scorch",
    'Strawberry___healthy': "Strawberry plant — healthy and disease-free",
    'Tomato___Bacterial_spot': "Tomato plant affected by Bacterial spot",
    'Tomato___Early_blight': "Tomato plant affected by Early blight",
    'Tomato___Late_blight': "Tomato plant affected by Late blight",
    'Tomato___Leaf_Mold': "Tomato plant affected by Leaf Mold",
    'Tomato___Septoria_leaf_spot': "Tomato plant affected by Septoria leaf spot",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Tomato plant affected by Two-spotted spider mite",
    'Tomato___Target_Spot': "Tomato plant affected by Target Spot",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Tomato plant affected by Tomato Yellow Leaf Curl Virus",
    'Tomato___Tomato_mosaic_virus': "Tomato plant affected by Tomato mosaic virus",
    'Tomato___healthy': "Tomato plant — healthy and disease-free",
}

_CONFIDENCE_THRESHOLD = 0.35

_SOURCES = [
    "PlantVillage dataset (38 classes, 14 crops)",
    "ResNet50 deep learning model with custom classification head (plant_disease_detection)",
]

_LIMITATIONS = [
    "Model trained on PlantVillage (controlled lab images). Performance on real field images may vary.",
    "Supports 38 classes across Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.",
    "Confidence scores represent softmax probabilities, not absolute diagnostic certainties.",
    "Nutrient deficiencies, pest damage, and environmental stress can visually mimic disease symptoms.",
    "Always consult a qualified local agricultural officer or agronomist before taking chemical control measures.",
]


class PlantDiseaseAgent(BaseAgent):
    """
    Specialist agent for plant disease identification using ResNet50 (38 classes).

    Features:
      - Lazy model loading (instantiated on first call).
      - 38 classes covering 14 crops.
      - Top-k predictions with confidence scores and readable descriptions.
      - Low confidence check (< 35%) triggers INCONCLUSIVE status.
      - Automatic crop mismatch detection when farmer specifies a crop.
      - Optional LLM-based actionable disease advice if HF token is configured.
    """

    def __init__(self, weights_path: Optional[str | Path] = None) -> None:
        if weights_path:
            self._weights_path = Path(weights_path)
        else:
            self._weights_path = _AGENT_DIR / "resnet_weights.npz"

        self._model = None
        self._classes = CLASSES

    @property
    def name(self) -> str:
        return "plant_disease"

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Returns True if the model weights file exists and TensorFlow can be loaded."""
        if not self._weights_path.exists():
            return False
        try:
            import tensorflow as tf  # noqa: F401
            import numpy as np       # noqa: F401
            from PIL import Image    # noqa: F401
            return True
        except ImportError:
            return False

    # ── Model loading ─────────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        """Lazy-loads the ResNet50 architecture and applies weights from resnet_weights.npz."""
        if self._model is not None:
            return

        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"ResNet weights not found at {self._weights_path}. "
                "Ensure resnet_weights.npz is present in plant_disease_detection/."
            )

        import numpy as np
        import tensorflow as tf
        from tensorflow.keras import layers, models, Input
        from tensorflow.keras.applications import ResNet50

        inputs = Input(shape=(224, 224, 3))
        # Build ResNet50 architecture without pre-downloading weights
        resnet_base = ResNet50(weights=None, include_top=False, input_tensor=inputs)
        resnet_base.trainable = False

        x = resnet_base.output
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dense(256, activation="relu", name="head_dense")(x)
        x = layers.Dropout(0.4, name="head_dropout")(x)
        outputs = layers.Dense(len(self._classes), activation="softmax", name="predictions")(x)

        model = models.Model(inputs, outputs)

        weights = np.load(str(self._weights_path), allow_pickle=True)
        model.set_weights([weights[key] for key in weights])
        self._model = model

    # ── Actionable advice helper ──────────────────────────────────────────────

    def _get_care_advice(self, readable_condition: str, is_healthy: bool) -> Optional[str]:
        """Optionally fetches advice from Hugging Face InferenceClient if token is set."""
        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            return None

        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(
                provider="featherless-ai",
                api_key=token,
            )
            condition_type = "healthy" if is_healthy else "diseased"
            prompt = (
                f"You are an expert plant pathologist. The model detected that the plant is "
                f"'{readable_condition}', which means it is {condition_type}. "
                f"Give 3-4 concise, bulleted points of actionable care or treatment advice. "
                f"Direct practical advice only."
            )
            completion = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise, professional plant health expert. Respond with direct practical care instructions only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.6,
            )
            return completion.choices[0].message["content"].strip()
        except Exception:
            return None

    # ── Core run ──────────────────────────────────────────────────────────────

    def run(
        self,
        image_bytes: Optional[bytes] = None,
        image_path: Optional[str | Path] = None,
        top_k: int = 3,
        crop_context: Optional[str] = None,
        **kwargs,
    ) -> AgentOutput:
        """
        Classifies an uploaded crop leaf image and returns a structured diagnosis.
        """
        # 1. Validate inputs
        if image_bytes is None and image_path is None:
            return self._missing_input_output(
                "No image provided. Please upload a crop leaf image for disease analysis."
            )

        # 2. Load bytes if image_path was given
        if image_bytes is None:
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
            except (FileNotFoundError, OSError) as exc:
                return self._error_output(
                    "Could not read the provided image file. Please check the path and try again.",
                    exc,
                )

        # 3. Process image and run inference
        try:
            from PIL import Image
            import numpy as np
            from tensorflow.keras.applications.resnet50 import preprocess_input

            self._ensure_model_loaded()

            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil_img = pil_img.resize((224, 224))
            img_arr = np.array(pil_img, dtype=np.float32)
            img_batch = np.expand_dims(img_arr, axis=0)
            img_batch = preprocess_input(img_batch)

            preds = self._model.predict(img_batch, verbose=0)[0]

        except Exception as exc:
            return self._error_output(
                "Model inference failed. The image may be corrupted, unreadable, "
                "or required libraries (tensorflow, pillow) are unavailable.",
                exc,
            )

        # 4. Extract top_k predictions
        top_indices = np.argsort(preds)[::-1][: max(1, top_k)]
        predictions = []
        for idx in top_indices:
            label = self._classes[idx]
            prob = float(preds[idx])
            desc = CLASS_DESCRIPTIONS.get(label, label)
            predictions.append({
                "label": label,
                "prob": round(prob, 4),
                "description": desc,
            })

        top = predictions[0]
        top_label = top["label"]
        top_confidence = top["prob"]
        top_description = top["description"]

        # Parse crop genus from label e.g. "Corn_(maize)" -> "Corn"
        crop_genus = top_label.split("___")[0].split("_(")[0].replace("_", " ").strip()
        disease_name = top_label.split("___")[-1].replace("_", " ").strip()

        # 5. Crop mismatch detection
        limitations = list(_LIMITATIONS)
        if crop_context:
            stated = crop_context.strip().lower()
            detected_crop = crop_genus.lower()
            if stated and (stated not in detected_crop and detected_crop not in stated):
                limitations.insert(
                    0,
                    f"Crop mismatch warning: You indicated crop '{crop_context}', but the model's "
                    f"closest visual match is for '{crop_genus}' ({disease_name}). "
                    f"Please verify the leaf belongs to {crop_context} before applying treatments.",
                )

        # 6. Status determination & summary construction
        is_healthy = "healthy" in top_label.lower()

        if top_confidence < _CONFIDENCE_THRESHOLD:
            status = AgentStatus.INCONCLUSIVE
            summary = (
                f"The image did not closely match any known disease pattern "
                f"(best match: {top_description} at {top_confidence:.1%} confidence). "
                f"Please upload a clearer, well-lit, close-up photograph of the leaf."
            )
        elif is_healthy:
            status = AgentStatus.SUCCESS
            summary = (
                f"The leaf appears healthy: {crop_genus} (Healthy, {top_confidence:.1%} confidence). "
                f"No significant disease symptoms were identified."
            )
        else:
            status = AgentStatus.SUCCESS
            summary = (
                f"Possible disease detected: {top_description} "
                f"({top_confidence:.1%} confidence). "
                f"Recommended next step: inspect adjacent leaves and verify before treatment."
            )

        # 7. Actionable advice if available
        advice = self._get_care_advice(top_description, is_healthy)
        if advice:
            summary += f"\n\nPathologist Advice:\n{advice}"

        return AgentOutput(
            agent=self.name,
            status=status,
            summary=summary,
            data={
                "predictions": predictions,
                "top_label": top_label,
                "top_confidence": top_confidence,
                "top_description": top_description,
                "crop_genus": crop_genus,
                "crop_context": crop_context,
                "advice": advice,
                "classes_supported": self._classes,
            },
            sources=_SOURCES,
            limitations=limitations,
        )

