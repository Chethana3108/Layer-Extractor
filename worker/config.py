import os
from dataclasses import dataclass, field


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var {name}")
    return val or ""


@dataclass
class Settings:
    # --- Content Hub ---
    ch_base_url: str = field(default_factory=lambda: _env("CH_BASE_URL", required=True).rstrip("/"))
    ch_client_id: str = field(default_factory=lambda: _env("CH_CLIENT_ID", required=True))
    ch_client_secret: str = field(default_factory=lambda: _env("CH_CLIENT_SECRET", required=True))
    ch_username: str = field(default_factory=lambda: _env("CH_USERNAME"))
    ch_password: str = field(default_factory=lambda: _env("CH_PASSWORD"))
    ch_grant_type: str = field(default_factory=lambda: _env("CH_GRANT_TYPE", "password"))  # password | client_credentials

    # Schema names (must match what you create in Schema)
    prop_is_derived: str = field(default_factory=lambda: _env("PROP_IS_DERIVED", "IsDerivedCutout"))
    prop_label: str = field(default_factory=lambda: _env("PROP_LABEL", "CutoutLabel"))
    prop_confidence: str = field(default_factory=lambda: _env("PROP_CONFIDENCE", "CutoutConfidence"))
    prop_bbox: str = field(default_factory=lambda: _env("PROP_BBOX", "CutoutBoundingBox"))
    prop_status: str = field(default_factory=lambda: _env("PROP_STATUS", "ObjectExtractionStatus"))
    prop_count: str = field(default_factory=lambda: _env("PROP_COUNT", "ObjectExtractionCount"))
    prop_message: str = field(default_factory=lambda: _env("PROP_MESSAGE", "ObjectExtractionMessage"))
    prop_source_hash: str = field(default_factory=lambda: _env("PROP_SOURCE_HASH", "ObjectExtractionSourceHash"))
    rel_parent: str = field(default_factory=lambda: _env("REL_PARENT", "ParentAssetToCutoutAsset"))

    cutout_marker: str = "__cutout_"  # loop guard: filenames containing this are never processed

    # --- Service Bus ---
    sb_connection: str = field(default_factory=lambda: _env("SB_CONNECTION", required=True))
    sb_queue: str = field(default_factory=lambda: _env("SB_QUEUE", "ch-object-extraction"))

    # --- Models / extraction ---
    detector_model: str = field(default_factory=lambda: _env("DETECTOR_MODEL", "IDEA-Research/grounding-dino-base"))
    sam_model: str = field(default_factory=lambda: _env("SAM_MODEL", "facebook/sam-vit-large"))
    # Empty = built-in broad vocabulary (boxes.VOCABULARY, ~70 everyday object types).
    # Set e.g. "strip,carton,bottle" to restrict detection to your own list.
    labels: list[str] = field(default_factory=lambda: [
        s.strip() for s in _env("DETECT_LABELS", "").split(",") if s.strip()
    ])
    include_people: bool = field(default_factory=lambda: _env("INCLUDE_PEOPLE", "false").lower() == "true")
    generic_pass: bool = field(default_factory=lambda: _env("GENERIC_PASS", "true").lower() == "true")
    box_threshold: float = field(default_factory=lambda: float(_env("BOX_THRESHOLD", "0.35")))
    relaxed_threshold: float = field(default_factory=lambda: float(_env("RELAXED_THRESHOLD", "0.25")))
    generic_threshold: float = field(default_factory=lambda: float(_env("GENERIC_THRESHOLD", "0.40")))
    text_threshold: float = field(default_factory=lambda: float(_env("TEXT_THRESHOLD", "0.25")))
    nms_iou: float = field(default_factory=lambda: float(_env("NMS_IOU", "0.5")))
    min_side_ratio: float = field(default_factory=lambda: float(_env("MIN_SIDE_RATIO", "0.03")))  # vs image short side
    min_area_ratio: float = field(default_factory=lambda: float(_env("MIN_AREA_RATIO", "0.005")))  # drop specks
    max_area_ratio: float = field(default_factory=lambda: float(_env("MAX_AREA_RATIO", "0.90")))   # drop "whole image"
    max_objects: int = field(default_factory=lambda: int(_env("MAX_OBJECTS", "10")))
    min_objects_to_split: int = field(default_factory=lambda: int(_env("MIN_OBJECTS_TO_SPLIT", "2")))
    padding_ratio: float = field(default_factory=lambda: float(_env("PADDING_RATIO", "0.04")))
    detect_max_side: int = field(default_factory=lambda: int(_env("DETECT_MAX_SIDE", "2048")))
    context_ratio: float = field(default_factory=lambda: float(_env("CONTEXT_RATIO", "0.15")))   # crop margin fed to SAM
    min_output_side: int = field(default_factory=lambda: int(_env("MIN_OUTPUT_SIDE", "48")))     # absolute floor in px
    mask_min_span: float = field(default_factory=lambda: float(_env("MASK_MIN_SPAN", "0.8")))   # mask must span 80% of box
    refiner: str = field(default_factory=lambda: _env("REFINER", "birefnet"))  # birefnet | none
    refiner_model: str = field(default_factory=lambda: _env("REFINER_MODEL", "ZhengPeng7/BiRefNet"))
    output_mode: str = field(default_factory=lambda: _env("OUTPUT_MODE", "cutout"))  # cutout | crop
    # transparent = RGBA PNG (for Gen-AI) | white = JPEG on white (clean DAM previews) | both
    output_background: str = field(default_factory=lambda: _env("OUTPUT_BACKGROUND", "transparent"))

    rendition_wait_seconds: int = field(default_factory=lambda: int(_env("RENDITION_WAIT_SECONDS", "240")))
