"""
HTTP API for VLA plan generation (OpenAPI 3.x + Swagger UI via FastAPI).

Execution of plans remains in the MuJoCo CLI (`vla_controller.main`).
"""

import os
import tempfile
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from franka_emika_panda.vla_controller import get_vla_plan, sanitize_plan_targets

_DESCRIPTION = """\
Returns a JSON array of action steps for the Franka Panda workspace (see `vla_controller`).

Manipulation actions include `move_to_object`, `hover_over_object`, `close_gripper`, `open_gripper`, `move_home` (with `target_name` one of `red_box`, `blue_box`, `green_box` where applicable).

Emote actions include `emote_nod`, `emote_shake_no`, `emote_wave`, `emote_dance`, `emote_clap`, `emote_yes`, `emote_no`, `emote_rotate_wrist`, `emote_bow`, `emote_celebrate` (no `target_name`).

Requires `GROQ_API_KEY` (and optional `GROQ_BASE_URL`) in the environment, same as the CLI controller.
"""

app = FastAPI(
    title="VLA Planning API",
    description=_DESCRIPTION,
    version="1.0.0",
    openapi_tags=[
        {
            "name": "planning",
            "description": "Vision-language action plan generation using Groq (Llama vision).",
        },
        {
            "name": "health",
            "description": "Liveness and readiness checks.",
        },
    ],
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send browsers to Swagger UI; the app root has no other page."""
    return RedirectResponse(url="/docs")


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class PlanStep(BaseModel):
    """One step in the execution plan returned by the VLA model."""

    model_config = ConfigDict(extra="allow")

    action: str = Field(
        ...,
        description=(
            "Primitive action name: manipulation (e.g. move_to_object, hover_over_object, "
            "close_gripper, open_gripper, move_home) or emote_* fun actions."
        ),
    )
    target_name: str | None = Field(
        None,
        description="Object id for manipulation steps: red_box, blue_box, or green_box.",
    )


def _suffix_for_upload(content_type: str | None) -> str:
    if not content_type:
        return ".jpg"
    ct = content_type.lower()
    if "png" in ct:
        return ".png"
    if "webp" in ct:
        return ".webp"
    return ".jpg"


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Return service liveness."""
    return HealthResponse(status="ok")


@app.post(
    "/api/v1/plan",
    response_model=list[PlanStep],
    tags=["planning"],
    summary="Generate a VLA plan from an image and command",
)
async def create_plan(
    image: Annotated[
        UploadFile,
        File(
            description="Workspace image (JPEG, PNG, or WebP) matching the MuJoCo camera view.",
        ),
    ],
    command: Annotated[
        str,
        Form(
            description='Natural language command, e.g. "move to the red box" or "do a small wave".',
        ),
    ],
) -> list[PlanStep]:
    """
    Upload a rendered scene image plus a user command; receive a JSON array of plan steps.

    Steps are normalized with `sanitize_plan_targets` (target names from command when missing).
    Full execution-time sanitization and MuJoCo execution are not performed over HTTP.
    """
    text = (command or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="command must be non-empty")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=422, detail="image file is empty")

    suffix = _suffix_for_upload(image.content_type)
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            os.write(fd, raw)
        finally:
            os.close(fd)

        plan: Any = get_vla_plan(tmp_path, text)
        if not isinstance(plan, list):
            raise HTTPException(
                status_code=422,
                detail="VLA model returned JSON that is not an array",
            )
        sanitized = sanitize_plan_targets(plan, text)
        try:
            return [PlanStep.model_validate(step) for step in sanitized]
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors()) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
