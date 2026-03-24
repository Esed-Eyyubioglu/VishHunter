from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .database import Base, SessionLocal, engine, get_db
from .ml import pipeline
from .models import Case, User
from .security import create_access_token, decode_access_token, hash_password, verify_password
from .services import (
    any_users_exist,
    audit_rows,
    case_rows,
    create_case_from_upload,
    create_initial_admin,
    current_system_settings,
    dashboard_metrics,
    decrypted_transcript,
    ensure_directories,
    get_case_detail,
    log_event,
    monthly_case_volume,
    reprocess_existing_cases,
    update_system_settings,
    user_rows,
    weekly_case_trend,
)


ensure_directories()
app = FastAPI(title=settings.app_name)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")
templates = Jinja2Templates(directory=str(settings.template_dir))


def role_label(role: str) -> str:
    return {
        "administrator": "Administrator",
        "technician": "Cybersecurity Technician",
        "analyst": "Junior Security Analyst",
    }.get(role, role.title())


templates.env.filters["role_label"] = role_label


@app.on_event("startup")
def startup() -> None:
    ensure_directories()
    Base.metadata.create_all(bind=engine)


def current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    token = request.cookies.get("vishhunter_token")
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    return db.scalar(select(User).where(User.email == payload.get("sub")))


def require_user(user: User | None = Depends(current_user)) -> User:
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive")
    return user


def require_role(*allowed: str):
    def dependency(user: User = Depends(require_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
        return user

    return dependency


def common_context(request: Request, user: User | None = None, **extra: object) -> dict:
    context = {"request": request, "user": user, "active_path": request.url.path}
    context.update(extra)
    return context


@app.get("/")
def root(user: User | None = Depends(current_user), db: Session = Depends(get_db)) -> RedirectResponse:
    if not any_users_exist(db):
        return RedirectResponse("/setup", status_code=status.HTTP_302_FOUND)
    return RedirectResponse("/dashboard" if user else "/login", status_code=status.HTTP_302_FOUND)


@app.get("/setup")
def setup_page(request: Request, db: Session = Depends(get_db), user: User | None = Depends(current_user)) -> Response:
    if any_users_exist(db):
        return RedirectResponse("/dashboard" if user else "/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("setup.html", common_context(request, error=None))


@app.post("/setup")
def setup_submit(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
) -> Response:
    if any_users_exist(db):
        return RedirectResponse("/login", status_code=status.HTTP_302_FOUND)
    admin = create_initial_admin(db, full_name=full_name, email=email, password=password)
    db.commit()
    token = create_access_token(admin.email, admin.role)
    response = RedirectResponse("/settings", status_code=status.HTTP_302_FOUND)
    response.set_cookie("vishhunter_token", token, httponly=True, samesite="lax")
    return response


@app.get("/login")
def login_page(request: Request, db: Session = Depends(get_db), user: User | None = Depends(current_user)) -> Response:
    if not any_users_exist(db):
        return RedirectResponse("/setup", status_code=status.HTTP_302_FOUND)
    if user:
        return RedirectResponse("/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("login.html", common_context(request, error=None))


@app.post("/login")
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
) -> Response:
    if not any_users_exist(db):
        return RedirectResponse("/setup", status_code=status.HTTP_302_FOUND)
    user = db.scalar(select(User).where(User.email == email))
    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "login.html",
            common_context(request, error="Invalid credentials."),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    if not user.is_active:
        return templates.TemplateResponse(
            "login.html",
            common_context(request, error="This account is inactive."),
            status_code=status.HTTP_403_FORBIDDEN,
        )

    user.last_login_at = datetime.utcnow()
    log_event(db, user, "auth.login", "Successful user login")
    db.commit()

    token = create_access_token(user.email, user.role)
    response = RedirectResponse("/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie("vishhunter_token", token, httponly=True, samesite="lax")
    return response


@app.get("/logout")
def logout() -> RedirectResponse:
    response = RedirectResponse("/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("vishhunter_token")
    return response


@app.get("/dashboard")
def dashboard(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)) -> Response:
    metrics = dashboard_metrics(db)
    cases = case_rows(db)[:5]
    trend = weekly_case_trend(db)
    return templates.TemplateResponse(
        "dashboard.html",
        common_context(
            request,
            user=user,
            metrics=metrics,
            recent_cases=cases,
            trend_labels=trend["labels"],
            trend_values=trend["values"],
        ),
    )


@app.get("/upload")
def upload_page(request: Request, user: User = Depends(require_role("technician", "administrator"))) -> Response:
    return templates.TemplateResponse(
        "upload.html",
        common_context(request, user=user, success=None, error=None, model_status=pipeline.model_status()),
    )


@app.post("/upload")
def upload_submit(
    request: Request,
    audio_file: UploadFile = File(...),
    user: User = Depends(require_role("technician", "administrator")),
    db: Session = Depends(get_db),
) -> Response:
    if not audio_file.filename:
        return templates.TemplateResponse(
            "upload.html",
            common_context(request, user=user, error="Please choose an audio file.", success=None, model_status=pipeline.model_status()),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    try:
        case = create_case_from_upload(db, file=audio_file, uploaded_by=user)
    except Exception as exc:
        db.rollback()
        return templates.TemplateResponse(
            "upload.html",
            common_context(
                request,
                user=user,
                error=f"Upload failed: {exc}",
                success=None,
                model_status=pipeline.model_status(),
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    db.commit()
    return RedirectResponse(f"/cases/{case.id}", status_code=status.HTTP_302_FOUND)


@app.get("/cases")
def cases_page(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)) -> Response:
    return templates.TemplateResponse("cases.html", common_context(request, user=user, cases=case_rows(db)))


@app.get("/cases/{case_id}")
def case_detail_page(request: Request, case_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)) -> Response:
    case = get_case_detail(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    transcript = decrypted_transcript(case)
    audio_url = f"/uploads/{Path(case.audio_record.storage_path).name}" if case.audio_record and case.audio_record.storage_path else None
    return templates.TemplateResponse(
        "case_detail.html",
        common_context(request, user=user, case=case, transcript=transcript, audio_url=audio_url),
    )


@app.post("/cases/{case_id}/review")
def review_case(
    case_id: str,
    review_status: str = Form(...),
    review_notes: str = Form(""),
    user: User = Depends(require_role("analyst", "administrator")),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    case = db.scalar(select(Case).where(Case.id == case_id))
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    case.review_status = review_status
    case.review_notes = review_notes
    case.validated_at = datetime.utcnow()
    log_event(db, user, "case.review", f"{case.case_number} marked as {review_status}")
    db.commit()
    return RedirectResponse(f"/cases/{case_id}", status_code=status.HTTP_302_FOUND)


@app.get("/reports")
def reports_page(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)) -> Response:
    metrics = dashboard_metrics(db)
    monthly = monthly_case_volume(db)
    return templates.TemplateResponse(
        "reports.html",
        common_context(
            request,
            user=user,
            metrics=metrics,
            risk_counts=metrics["distribution"],
            monthly_labels=monthly["labels"],
            monthly_totals=monthly["values"],
        ),
    )


@app.get("/validation")
def validation_page(
    request: Request,
    user: User = Depends(require_role("analyst", "administrator")),
    db: Session = Depends(get_db),
) -> Response:
    pending_cases = [case for case in case_rows(db) if case.review_status == "pending"]
    return templates.TemplateResponse("validation.html", common_context(request, user=user, cases=pending_cases))


@app.get("/users")
def users_page(request: Request, user: User = Depends(require_role("administrator")), db: Session = Depends(get_db)) -> Response:
    return templates.TemplateResponse("users.html", common_context(request, user=user, users=user_rows(db), error=None))


@app.post("/users")
def create_user(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    role: str = Form(...),
    password: str = Form(...),
    user: User = Depends(require_role("administrator")),
    db: Session = Depends(get_db),
) -> Response:
    if db.scalar(select(User).where(User.email == email)):
        return templates.TemplateResponse(
            "users.html",
            common_context(request, user=user, users=user_rows(db), error="User email already exists."),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    created_user = User(full_name=full_name, email=email, role=role, password_hash=hash_password(password))
    db.add(created_user)
    db.flush()
    log_event(db, user, "user.create", f"Created user {created_user.email} with role {created_user.role}")
    db.commit()
    return RedirectResponse("/users", status_code=status.HTTP_302_FOUND)


@app.post("/users/{user_id}/toggle")
def toggle_user(user_id: str, user: User = Depends(require_role("administrator")), db: Session = Depends(get_db)) -> RedirectResponse:
    target = db.scalar(select(User).where(User.id == user_id))
    if target and target.id != user.id:
        target.is_active = not target.is_active
        log_event(db, user, "user.toggle", f"{target.email} set active={target.is_active}")
        db.commit()
    return RedirectResponse("/users", status_code=status.HTTP_302_FOUND)


@app.post("/users/{user_id}/update")
def update_user(
    user_id: str,
    full_name: str = Form(...),
    role: str = Form(...),
    is_active: str | None = Form(default=None),
    user: User = Depends(require_role("administrator")),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    target = db.scalar(select(User).where(User.id == user_id))
    if target:
        target.full_name = full_name
        target.role = role
        target.is_active = is_active == "on"
        log_event(db, user, "user.update", f"Updated user {target.email}")
        db.commit()
    return RedirectResponse("/users", status_code=status.HTTP_302_FOUND)


@app.post("/users/{user_id}/delete")
def delete_user(user_id: str, user: User = Depends(require_role("administrator")), db: Session = Depends(get_db)) -> RedirectResponse:
    target = db.scalar(select(User).where(User.id == user_id))
    if not target or target.id == user.id:
        return RedirectResponse("/users", status_code=status.HTTP_302_FOUND)
    if target.uploaded_cases or target.assigned_cases:
        raise HTTPException(status_code=400, detail="User has related cases. Reassign or deactivate the account instead.")
    log_event(db, user, "user.delete", f"Deleted user {target.email}")
    db.delete(target)
    db.commit()
    return RedirectResponse("/users", status_code=status.HTTP_302_FOUND)


@app.get("/audit")
def audit_page(request: Request, user: User = Depends(require_role("administrator")), db: Session = Depends(get_db)) -> Response:
    return templates.TemplateResponse("audit.html", common_context(request, user=user, logs=audit_rows(db)))


@app.get("/settings")
def settings_page(request: Request, user: User = Depends(require_role("administrator"))) -> Response:
    return templates.TemplateResponse(
        "settings.html",
        common_context(request, user=user, app_settings=current_system_settings(), model_status=pipeline.model_status(), message=None),
    )


@app.post("/settings")
def update_settings(
    request: Request,
    high_threshold: float = Form(...),
    medium_threshold: float = Form(...),
    max_training_samples: int = Form(...),
    positive_dataset_path: str = Form(...),
    negative_dataset_path: str = Form(...),
    user: User = Depends(require_role("administrator")),
    db: Session = Depends(get_db),
) -> Response:
    if not 0 < medium_threshold < high_threshold < 1:
        return templates.TemplateResponse(
            "settings.html",
            common_context(
                request,
                user=user,
                app_settings=current_system_settings(),
                model_status=pipeline.model_status(),
                message="Thresholds are invalid. Use 0 < medium < high < 1.",
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    if not Path(positive_dataset_path).exists() or not Path(negative_dataset_path).exists():
        return templates.TemplateResponse(
            "settings.html",
            common_context(
                request,
                user=user,
                app_settings=current_system_settings(),
                model_status=pipeline.model_status(),
                message="One or more dataset paths do not exist.",
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    updated = update_system_settings(
        {
            "high_threshold": high_threshold,
            "medium_threshold": medium_threshold,
            "max_training_samples": max_training_samples,
            "positive_dataset_path": positive_dataset_path,
            "negative_dataset_path": negative_dataset_path,
        }
    )
    log_event(db, user, "settings.update", "Updated model thresholds and dataset paths")
    db.commit()
    return templates.TemplateResponse(
        "settings.html",
        common_context(request, user=user, app_settings=updated, model_status=pipeline.model_status(), message="Settings saved."),
    )


@app.post("/settings/retrain")
def retrain_models(request: Request, user: User = Depends(require_role("administrator")), db: Session = Depends(get_db)) -> Response:
    metadata = pipeline.train_from_public_datasets(force_retrain=True)
    log_event(db, user, "model.retrain", "Retrained hybrid pipeline from public datasets")
    db.commit()
    return templates.TemplateResponse(
        "settings.html",
        common_context(
            request,
            user=user,
            app_settings=current_system_settings(),
            model_status=pipeline.model_status(),
            message=f"Retraining complete. {metadata.get('sample_count', 0)} samples used.",
        ),
    )


@app.post("/settings/reprocess")
def reprocess_cases(request: Request, user: User = Depends(require_role("administrator")), db: Session = Depends(get_db)) -> Response:
    updated = reprocess_existing_cases(db)
    log_event(db, user, "model.reprocess", f"Reprocessed {updated} stored cases with current model artifacts")
    db.commit()
    return templates.TemplateResponse(
        "settings.html",
        common_context(
            request,
            user=user,
            app_settings=current_system_settings(),
            model_status=pipeline.model_status(),
            message=f"Reprocessing complete. {updated} stored cases updated.",
        ),
    )


@app.get("/cases/{case_id}/report.pdf")
def export_case_report(case_id: str, user: User = Depends(require_user), db: Session = Depends(get_db)) -> StreamingResponse:
    case = get_case_detail(db, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    transcript = decrypted_transcript(case)
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title=f"{case.case_number} Report")
    styles = getSampleStyleSheet()

    rows = [
        ["Case Number", case.case_number],
        ["Title", case.title],
        ["Risk Level", case.risk_level.title()],
        ["Audio Score", f"{case.audio_score:.0%}"],
        ["Text Score", f"{case.text_score:.0%}"],
        ["Confidence", f"{case.confidence_score:.0%}"],
        ["Model Verdict", case.model_verdict.title()],
        ["Review Status", case.review_status.replace('_', ' ').title()],
        ["Assigned Analyst", case.assigned_to_user.full_name if case.assigned_to_user else "Unassigned"],
    ]
    table = Table(rows, colWidths=[130, 360])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8"))]))

    story = [
        Paragraph("<b>VishHunter Case Report</b>", styles["Title"]),
        Spacer(1, 12),
        table,
        Spacer(1, 16),
        Paragraph(f"<b>Hybrid Summary</b>: {case.summary}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(f"<b>Indicators</b>: {', '.join(case.indicators)}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(f"<b>Transcript</b>: {transcript}", styles["BodyText"]),
    ]
    doc.build(story)
    buffer.seek(0)
    filename = f"{case.case_number.lower()}-report.pdf"
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/dashboard-metrics")
def dashboard_api(user: User = Depends(require_user), db: Session = Depends(get_db)) -> dict:
    return dashboard_metrics(db)
