"""FastAPI server exposing the Pharmaceuticals financial model."""
from __future__ import annotations

import logging
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Mapping, MutableMapping, Sequence

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.openapi.utils import get_openapi
from jose import JWTError, jwt

from ..model_registry import MODEL_REGISTRY, ModelSpec
from ..core.report import REPORT_FORMATS, generate_report
from ..services.paystack import PaystackClient, PaystackError, SubscriptionStatus
from ..services.subscription_store import StoredSubscriptionRecord, get_subscription_store
from ..services.user_store import UserRecord, get_user_store
from .schemas import (
    AuthLoginRequest,
    AuthUpdateRequest,
    SubscriptionCheckRequest,
    SubscriptionCheckResponse,
    SubscriptionCheckoutRequest,
    SubscriptionCheckoutResponse,
    SubscriptionStatusRecord,
    SubscriptionStatusUpsert,
    SubscriptionVerifyRequest,
    SubscriptionVerifyResponse,
    ValidationResponse,
)


def _input_payload(payload: Any) -> Dict[str, Any]:
    """Convert request payloads (Pydantic models or dicts) into plain dicts using aliases."""

    if hasattr(payload, "model_dump"):
        return payload.model_dump(by_alias=True)  # type: ignore[call-arg, attr-defined]
    return dict(payload or {})


def _resolve_inputs(payload: Mapping[str, Any] | None, spec: ModelSpec) -> Any:
    if payload:
        return spec.parse_inputs(payload)
    return spec.load_inputs(None)


def _resolve_subscription_email(request_email: str | None, context: AuthContext | None) -> str | None:
    """Determine the subscription email tied to the request.

    - When auth is disabled: fall back to the request email.
    - When using API token: trust the provided email (for service-to-service).
    - Otherwise: require the caller's email to match the provided email (when given), and
      default to the caller's email when absent.
    """

    if context is None:
        return request_email
    if context.method == "api_token":
        return request_email
    caller_email = (context.email or "").strip().lower()
    if not caller_email:
        raise HTTPException(status_code=400, detail="Authenticated user email missing.")
    if request_email and request_email.strip().lower() != caller_email:
        raise HTTPException(status_code=403, detail="Cannot manage subscriptions for another user.")
    return caller_email


def _ensure_user_exists(email: str | None) -> None:
    if not email:
        return
    store = get_user_store()
    if store.get_user(email) is None:
        raise HTTPException(status_code=404, detail="User not found.")


def _extract_email_from_transaction(payload: Mapping[str, Any] | None) -> str:
    """Return the best-effort email value from a Paystack transaction payload."""

    if not isinstance(payload, Mapping):
        return ""
    email = payload.get("email") or ""
    customer = payload.get("customer")
    if not email and isinstance(customer, Mapping):
        email = customer.get("email") or ""
    metadata = payload.get("metadata")
    if not email and isinstance(metadata, Mapping):
        email = metadata.get("email") or metadata.get("user_email") or ""
    return str(email or "").strip().lower()


def _run_model(inputs: Any, spec: ModelSpec):
    outputs = spec.run_core(inputs)
    return spec.build_response(outputs)


def _record_payload(record: StoredSubscriptionRecord) -> SubscriptionStatusRecord:
    return SubscriptionStatusRecord(
        email=record.email,
        is_active=record.is_active,
        status_message=record.status_message,
        updated_at=record.updated_at,
        source=record.source,
        expires_at=record.expires_at,
        payload=record.payload,
    )


def _status_from_record(record: StoredSubscriptionRecord) -> SubscriptionStatus:
    return SubscriptionStatus(
        email=record.email,
        is_active=record.is_active,
        message=record.status_message,
        payload=record.payload,
    )


def _cached_subscription_status(email: str | None) -> tuple[SubscriptionStatus, float] | None:
    if not email:
        return None
    store = get_subscription_store()
    if store is None:
        return None
    record = store.get_status(email)
    if record is None:
        return None
    if record.is_expired():
        store.remove_status(email)
        return None
    return _status_from_record(record), record.updated_at


def _persist_subscription_status(status: SubscriptionStatus, *, source: str) -> None:
    if not status.email:
        return
    store = get_subscription_store()
    if store is None:
        return
    ttl_seconds = SUBSCRIPTION_ACTIVE_TTL_SECONDS if status.is_active else SUBSCRIPTION_INACTIVE_TTL_SECONDS
    try:
        store.write_status(status, source=source, ttl_seconds=ttl_seconds)
    except Exception:  # pragma: no cover - defensive persistence
        LOGGER.debug("Unable to persist subscription status for %s", status.email)


LOGGER = logging.getLogger(__name__)

API_TOKEN_ENV = "FINANCIAL_MODELS_API_TOKEN"
API_TOKEN_HEADER = "X-API-Key"
GOOGLE_AUDIENCE_ENV = "FINANCIAL_MODELS_GOOGLE_AUDIENCE"
GOOGLE_VALID_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}
JWT_SECRET_ENV = "FINANCIAL_MODELS_AUTH_SECRET"
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 3600
CORS_ALLOWED_ORIGINS_ENV = "FINANCIAL_MODELS_CORS_ORIGINS"
SUBSCRIPTION_ACTIVE_TTL_SECONDS = float(os.getenv("SUBSCRIPTION_ACTIVE_TTL_SECONDS", 10 * 60))
SUBSCRIPTION_INACTIVE_TTL_SECONDS = float(os.getenv("SUBSCRIPTION_INACTIVE_TTL_SECONDS", 2 * 60))

try:  # pragma: no cover - optional dependency
    from google.oauth2 import id_token as google_id_token  # type: ignore
    from google.auth.transport import requests as google_requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    google_id_token = None  # type: ignore
    google_requests = None  # type: ignore


@dataclass
class AuthContext:
    """Represents the authenticated caller."""

    method: str
    subject: str | None = None
    email: str | None = None
    claims: Mapping[str, Any] | None = None
    user: UserRecord | None = None


def _expected_api_token() -> str | None:
    token = os.getenv(API_TOKEN_ENV, "").strip()
    return token or None


def _google_audiences() -> list[str]:
    value = os.getenv(GOOGLE_AUDIENCE_ENV, "")
    audiences = [item.strip() for item in value.split(",") if item.strip()]
    return audiences


def _jwt_secret() -> str | None:
    secret = os.getenv(JWT_SECRET_ENV, "").strip()
    return secret or None


def _allowed_origins() -> list[str]:
    raw = os.getenv(CORS_ALLOWED_ORIGINS_ENV, "")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if not origins:
        return ["http://localhost:4200"]
    if "*" in origins:
        return ["*"]
    return origins


def _issue_jwt(user: UserRecord) -> str:
    secret = _jwt_secret()
    if secret is None:
        raise RuntimeError("JWT secret is not configured.")
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "provider": user.provider,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=JWT_EXPIRY_SECONDS),
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def _verify_jwt(token: str) -> Mapping[str, Any]:
    secret = _jwt_secret()
    if secret is None:
        raise RuntimeError("JWT secret is not configured.")
    return jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])


def _verify_google_token(token: str, audiences: Sequence[str]) -> Mapping[str, Any]:
    """Validate a Google ID token for one of the configured audiences."""

    if google_id_token is None or google_requests is None:  # pragma: no cover - configuration guard
        raise RuntimeError("Google authentication requires the 'google-auth' package.")

    request = google_requests.Request()
    errors: MutableMapping[str, Exception] = {}
    for audience in audiences:
        try:
            payload = google_id_token.verify_oauth2_token(token, request, audience)
        except Exception as exc:  # pragma: no cover - upstream validation details unsuitable for unit tests
            errors[audience] = exc
            continue
        issuer = str(payload.get("iss") or "")
        if issuer not in GOOGLE_VALID_ISSUERS:
            errors[audience] = ValueError("Unexpected token issuer.")
            continue
        return payload
    if errors:
        LOGGER.debug("Google token verification failed: %s", errors)
    raise ValueError("Unable to verify Google ID token for the configured audience(s).")


def require_authorization(
    x_api_key: str | None = Header(default=None, alias=API_TOKEN_HEADER),
    authorization: str | None = Header(default=None, convert_underscores=False),
) -> AuthContext | None:
    """Enforce API token or Google social auth when configured."""

    expected = _expected_api_token()
    google_audiences = _google_audiences()
    secret = _jwt_secret()

    if expected is None and not google_audiences and not secret:
        return None  # auth disabled

    if expected is not None and x_api_key:
        if secrets.compare_digest(x_api_key.strip(), expected):
            return AuthContext(method="api_token")

    if authorization:
        scheme, _, token = authorization.strip().partition(" ")
        token = token.strip()
        if scheme.lower() == "bearer" and token:
            if secret:
                try:
                    payload = _verify_jwt(token)
                    email = payload.get("email")
                    user = None
                    if email:
                        user_store = get_user_store()
                        user = user_store.get_user(email)
                    return AuthContext(
                        method="jwt",
                        subject=str(payload.get("sub") or ""),
                        email=email,
                        claims=payload,
                        user=user,
                    )
                except JWTError:
                    pass
                except RuntimeError as exc:  # pragma: no cover - missing secret
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
            if google_audiences:
                try:
                    payload = _verify_google_token(token, google_audiences)
                except RuntimeError as exc:  # pragma: no cover - dependency guard
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
                except ValueError:
                    pass
                else:
                    user = None
                    try:
                        user_store = get_user_store()
                        user = user_store.ensure_user(
                            email=payload.get("email") or "",
                            name=payload.get("name") or payload.get("email"),
                            provider="google",
                        )
                    except Exception:
                        LOGGER.debug("Unable to upsert Google user", exc_info=True)
                    return AuthContext(
                        method="google",
                        subject=str(payload.get("sub") or ""),
                        email=payload.get("email"),
                        claims=payload,
                        user=user,
                    )

    detail = "Unauthorized request."
    if expected is not None and not x_api_key:
        detail = "Missing API token."
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


@lru_cache(maxsize=1)
def get_paystack_client() -> PaystackClient:
    """Create (and cache) a Paystack client based on env vars."""

    secret_key = os.getenv("PAYSTACK_SECRET_KEY")
    plan_code = os.getenv("PAYSTACK_PLAN_CODE")
    default_amount = os.getenv("PAYSTACK_PLAN_AMOUNT_KOBO")
    amount = int(default_amount) if default_amount and default_amount.isdigit() else None
    callback_url = os.getenv("PAYSTACK_CALLBACK_URL")
    cancel_url = os.getenv("PAYSTACK_CANCEL_ACTION_URL")
    return PaystackClient(
        secret_key=secret_key,
        plan_code=plan_code,
        default_amount_kobo=amount,
        callback_url=callback_url,
        cancel_action_url=cancel_url,
        fetch_plan_amount=True,
    )


def create_app() -> FastAPI:
    """Instantiate the FastAPI application."""

    app = FastAPI(
        title="Pharmaceuticals Financial Model API",
        version="1.0.0",
        description="HTTP interface for running the Pharmaceuticals financial engine.",
    )

    origins = _allowed_origins()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    @app.get("/health")
    def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/auth/register")
    def register_user(email: str, password: str, name: str | None = None) -> Dict[str, Any]:
        secret = _jwt_secret()
        if secret is None:
            raise HTTPException(status_code=500, detail=f"{JWT_SECRET_ENV} is not configured.")
        store = get_user_store()
        try:
            user = store.create_user(email=email, password=password, name=name, provider="local")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        token = _issue_jwt(user)
        return {"access_token": token, "token_type": "bearer", "user": {"email": user.email, "name": user.name}}

    @app.post("/auth/login")
    async def login_user(
        request: Request, payload: AuthLoginRequest | None = Body(default=None)
    ) -> Dict[str, Any]:
        secret = _jwt_secret()
        if secret is None:
            raise HTTPException(status_code=500, detail=f"{JWT_SECRET_ENV} is not configured.")
        store = get_user_store()
        username: str | None = None
        password: str | None = None
        if payload is not None:
            username = payload.email
            password = payload.password
        else:
            content_type = request.headers.get("content-type", "")
            if "application/x-www-form-urlencoded" in content_type:
                form = await request.form()
                username = form.get("username")
                password = form.get("password")
        if not username or not password:
            raise HTTPException(status_code=400, detail="Email and password are required.")
        user = store.verify_user(email=username, password=password)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        token = _issue_jwt(user)
        return {"access_token": token, "token_type": "bearer", "user": {"email": user.email, "name": user.name}}

    @app.get("/auth/me")
    def current_user(context: AuthContext | None = Depends(require_authorization)) -> Dict[str, Any]:
        if context is None or not context.email:
            raise HTTPException(status_code=401, detail="Unauthorized.")
        return {"email": context.email, "method": context.method, "claims": context.claims}

    @app.patch("/auth/me")
    def update_current_user(
        update: AuthUpdateRequest,
        context: AuthContext | None = Depends(require_authorization),
    ) -> Dict[str, Any]:
        if context is None or not context.email:
            raise HTTPException(status_code=401, detail="Unauthorized.")
        store = get_user_store()
        if update.password and context.method not in {"jwt", "api_token"}:
            raise HTTPException(status_code=400, detail="Password changes require local account authentication.")
        try:
            user = store.update_user(context.email, name=update.name, password=update.password)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"email": user.email, "name": user.name}

    @app.delete("/auth/users/{email}", status_code=204)
    def delete_user(
        email: str,
        context: AuthContext | None = Depends(require_authorization),
    ) -> None:
        normalized = (email or "").strip().lower()
        if context is None or not context.email or context.email.lower() != normalized:
            raise HTTPException(status_code=403, detail="Forbidden.")
        store = get_user_store()
        store.delete_user(normalized)
        return None

    @app.get("/auth/users")
    def list_users(context: AuthContext | None = Depends(require_authorization)) -> Dict[str, Any]:
        if context is None:
            raise HTTPException(status_code=401, detail="Unauthorized.")
        store = get_user_store()
        users = store.list_users()
        return {
            "users": [
                {
                    "email": user.email,
                    "name": user.name,
                    "provider": user.provider,
                    "created_at": user.created_at,
                }
                for user in users
            ]
        }

    def _register_model_routes(model_type: str, spec: ModelSpec) -> None:
        RunRequestModel = spec.run_request_model
        ValidateRequestModel = spec.validate_request_model
        # Ensure Pydantic models are fully built for OpenAPI generation
        if hasattr(RunRequestModel, "model_rebuild"):
            RunRequestModel.model_rebuild()  # type: ignore[attr-defined]
        if hasattr(ValidateRequestModel, "model_rebuild"):
            ValidateRequestModel.model_rebuild()  # type: ignore[attr-defined]

        def run_model_versioned(
            request: RunRequestModel, _: AuthContext | None = Depends(require_authorization)
        ):
            try:
                payload = _input_payload(request.inputs) if request.inputs is not None else None
                inputs = _resolve_inputs(payload, spec)
            except Exception as exc:  # pragma: no cover - validation handled explicitly in /inputs/{model_type}/validate
                raise HTTPException(status_code=400, detail=f"Invalid inputs: {exc}") from exc
            return _run_model(inputs, spec)

        def validate_inputs_versioned(
            request: ValidateRequestModel,
            _: AuthContext | None = Depends(require_authorization),
        ) -> ValidationResponse:
            try:
                spec.parse_inputs(_input_payload(request.inputs))
            except Exception as exc:
                return ValidationResponse(valid=False, message=str(exc))
            return ValidationResponse(valid=True, message="Inputs parsed successfully.")

        run_model_versioned.__annotations__["request"] = RunRequestModel
        validate_inputs_versioned.__annotations__["request"] = ValidateRequestModel

        app.add_api_route(
            f"/model/{model_type}/run",
            run_model_versioned,
            methods=["POST"],
            response_model=spec.response_model,
        )
        app.add_api_route(
            f"/inputs/{model_type}/validate",
            validate_inputs_versioned,
            methods=["POST"],
            response_model=ValidationResponse,
        )

        if spec.build_report_sections:

            class ReportRequest(RunRequestModel):  # type: ignore[misc,valid-type]
                format: str  # type: ignore[assignment]

            def generate_report_versioned(
                request: ReportRequest, _: AuthContext | None = Depends(require_authorization)
            ):
                fmt = (request.format or "").strip().upper()
                if fmt not in REPORT_FORMATS:
                    raise HTTPException(status_code=400, detail=f"Unsupported report format '{request.format}'.")
                try:
                    payload = _input_payload(request.inputs) if getattr(request, "inputs", None) is not None else None
                    inputs = _resolve_inputs(payload, spec)
                    result = spec.run_core(inputs)
                    sections = spec.build_report_sections(inputs, result)
                    data, mime, filename = generate_report(
                        sections,
                        fmt,
                        report_name=spec.report_name or f"{model_type}_financial_report",
                        report_title=spec.report_title or spec.name,
                    )
                except Exception as exc:  # pragma: no cover - report generation edge cases
                    raise HTTPException(status_code=400, detail=f"Unable to generate report: {exc}") from exc
                headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
                return Response(content=data, media_type=mime, headers=headers)

            generate_report_versioned.__annotations__["request"] = ReportRequest  # type: ignore[attr-defined]

            app.add_api_route(
                f"/report/{model_type}/generate",
                generate_report_versioned,
                methods=["POST"],
                response_class=Response,
            )

    for model_type, spec in MODEL_REGISTRY.items():
        _register_model_routes(model_type, spec)

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {}).update(
            {
                "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
                "apiKeyAuth": {"type": "apiKey", "name": API_TOKEN_HEADER, "in": "header"},
            }
        )
        openapi_schema["security"] = [{"bearerAuth": []}, {"apiKeyAuth": []}]
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[assignment]

    @app.post("/subscriptions/checkout", response_model=SubscriptionCheckoutResponse)
    def create_subscription_checkout(
        request: SubscriptionCheckoutRequest,
        client: PaystackClient = Depends(get_paystack_client),
        _: AuthContext | None = Depends(require_authorization),
    ) -> SubscriptionCheckoutResponse:
        email = _resolve_subscription_email(request.email, _)
        _ensure_user_exists(email)
        metadata: Dict[str, Any] = {"source": "financial_models_api"}
        if request.metadata:
            metadata.update(request.metadata)
        if request.scenario:
            metadata.setdefault("scenario", request.scenario)
        metadata.setdefault("email", email or request.email)
        try:
            checkout_url = client.create_subscription_checkout(
                email or request.email,
                metadata=metadata,
            )
        except PaystackError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if not checkout_url:
            raise HTTPException(status_code=500, detail="Unable to generate Paystack checkout URL.")
        return SubscriptionCheckoutResponse(email=email or request.email, checkout_url=checkout_url)

    @app.post("/subscriptions/verify", response_model=SubscriptionVerifyResponse)
    def verify_subscription(
        request: SubscriptionVerifyRequest,
        client: PaystackClient = Depends(get_paystack_client),
        _: AuthContext | None = Depends(require_authorization),
    ) -> SubscriptionVerifyResponse:
        reference = (request.reference or "").strip()
        if not reference:
            raise HTTPException(status_code=400, detail="Reference is required.")
        try:
            tx = client.verify_transaction(reference)
        except PaystackError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        if not isinstance(tx, Mapping):
            raise HTTPException(status_code=502, detail="Unexpected Paystack verification response.")

        status_value = str(tx.get("status") or "").strip().lower()
        if status_value != "success":
            return SubscriptionVerifyResponse(
                email=None,
                is_active=False,
                message="Transaction not successful.",
                payload=tx,
            )

        email = _extract_email_from_transaction(tx)
        if not email:
            return SubscriptionVerifyResponse(
                email=None,
                is_active=False,
                message="Transaction verified but no email found in Paystack payload.",
                payload=tx,
            )

        try:
            sub_status = client.has_active_subscription(email)
        except PaystackError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        _persist_subscription_status(sub_status, source="api:verify")

        return SubscriptionVerifyResponse(
            email=email,
            is_active=sub_status.is_active,
            message=sub_status.message,
            payload=sub_status.payload or tx,
        )

    @app.post("/subscriptions/check", response_model=SubscriptionCheckResponse)
    def check_subscription(
        request: SubscriptionCheckRequest,
        client: PaystackClient = Depends(get_paystack_client),
        _: AuthContext | None = Depends(require_authorization),
    ) -> SubscriptionCheckResponse:
        email = _resolve_subscription_email(request.email, _)
        _ensure_user_exists(email)
        cached = _cached_subscription_status(email or request.email)
        if cached:
            status, updated_at = cached
            return SubscriptionCheckResponse(
                email=status.email,
                is_active=status.is_active,
                message=status.message,
                payload=status.payload,
                cached=True,
                cached_at=updated_at,
            )
        try:
            status = client.has_active_subscription(email or request.email)
        except PaystackError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if not isinstance(status, SubscriptionStatus):
            raise HTTPException(status_code=500, detail="Unexpected Paystack response.")
        _persist_subscription_status(status, source="api:check")
        return SubscriptionCheckResponse(
            email=email or status.email,
            is_active=status.is_active,
            message=status.message,
            payload=status.payload,
            cached=False,
            cached_at=time.time(),
        )

    @app.get("/subscriptions/status", response_model=SubscriptionStatusRecord)
    def get_subscription_status(
        email: str,
        _: AuthContext | None = Depends(require_authorization),
    ) -> SubscriptionStatusRecord:
        resolved_email = _resolve_subscription_email(email, _)
        store = get_subscription_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Subscription store unavailable.")
        record = store.get_status(resolved_email or email)
        if record is None:
            raise HTTPException(status_code=404, detail="Subscription not found.")
        if record.is_expired():
            store.remove_status(resolved_email or email)
            raise HTTPException(status_code=404, detail="Subscription not found.")
        return _record_payload(record)

    @app.post("/subscriptions/status", response_model=SubscriptionStatusRecord)
    def upsert_subscription_status(
        request: SubscriptionStatusUpsert,
        _: AuthContext | None = Depends(require_authorization),
    ) -> SubscriptionStatusRecord:
        resolved_email = _resolve_subscription_email(request.email, _)
        _ensure_user_exists(resolved_email)
        store = get_subscription_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Subscription store unavailable.")
        status = SubscriptionStatus(
            email=resolved_email or request.email,
            is_active=request.is_active,
            message=request.status_message,
            payload=request.payload,
        )
        store.write_status(status, source=request.source or "api", ttl_seconds=request.ttl_seconds)
        record = store.get_status(resolved_email or request.email)
        if record is None:
            raise HTTPException(status_code=500, detail="Unable to persist subscription.")
        return _record_payload(record)

    @app.delete("/subscriptions/status", status_code=204)
    def delete_subscription_status(
        email: str,
        _: AuthContext | None = Depends(require_authorization),
    ) -> None:
        resolved_email = _resolve_subscription_email(email, _)
        store = get_subscription_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Subscription store unavailable.")
        store.remove_status(resolved_email or email)
        return None

    return app


app = create_app()

__all__ = ["app", "create_app", "get_paystack_client"]
