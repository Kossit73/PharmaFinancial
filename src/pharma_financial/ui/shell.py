"""Shared presentation helpers for the pharma workspace."""

from __future__ import annotations


def inject_app_theme() -> None:
    from .. import app as legacy

    legacy.st.markdown(
        """
        <style>
        :root {
            --pharma-ink: #0f172a;
            --pharma-muted: #475569;
            --pharma-brand: #1d4ed8;
            --pharma-shell: #e2e8f0;
            --pharma-soft: rgba(29, 78, 216, 0.06);
        }
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }
        .designer-hero {
            margin-bottom: 1.2rem;
            padding: 1.8rem 1.9rem;
            border-radius: 28px;
            border: 1px solid rgba(29, 78, 216, 0.12);
            background:
                linear-gradient(135deg, rgba(233, 240, 255, 0.96), rgba(255, 255, 255, 0.94)),
                linear-gradient(135deg, rgba(29, 78, 216, 0.05), rgba(56, 189, 248, 0.06));
            box-shadow: 0 24px 48px rgba(15, 23, 42, 0.08);
        }
        .designer-kicker {
            margin: 0 0 0.45rem 0;
            font-size: 0.78rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--pharma-brand);
            font-weight: 700;
        }
        .designer-title {
            margin: 0;
            font-size: clamp(2rem, 2.8vw, 3.15rem);
            line-height: 1.02;
            color: var(--pharma-ink);
            font-weight: 800;
        }
        .designer-copy {
            max-width: 55rem;
            margin: 0.7rem 0 0 0;
            color: var(--pharma-muted);
            font-size: 1rem;
            line-height: 1.6;
        }
        .designer-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }
        .designer-badge {
            padding: 0.42rem 0.78rem;
            border-radius: 999px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: rgba(255, 255, 255, 0.92);
            color: var(--pharma-brand);
            font-size: 0.82rem;
            font-weight: 700;
        }
        .pharma-section-card {
            padding: 1rem 1.1rem;
            margin: 0.35rem 0 1rem 0;
            border-radius: 18px;
            border: 1px solid var(--pharma-shell);
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
        }
        .pharma-section-card h3, .pharma-section-card h4 {
            margin-bottom: 0.35rem;
        }
        .pharma-section-caption {
            color: var(--pharma-muted);
            margin: 0;
        }
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            background: linear-gradient(135deg, #1d4ed8, #0891b2);
            color: white;
            border-color: transparent;
            box-shadow: 0 12px 24px rgba(8, 145, 178, 0.16);
        }
        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"] {
            border-radius: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_model_hero() -> None:
    from .. import app as legacy

    badges = "".join(
        f'<span class="designer-badge">{label}</span>'
        for label in (
            "Scenario dashboards",
            "Financial statements",
            "Monte Carlo insights",
            "Executive layout",
        )
    )
    legacy.st.markdown(
        f"""
        <section class="designer-hero">
            <p class="designer-kicker">Pharma planning suite</p>
            <h1 class="designer-title">Longevity Pharmaceuticals Financial Model</h1>
            <p class="designer-copy">
                Review revenue, operations, financing, and risk assumptions in a cleaner executive shell
                built for management review and investor presentation.
            </p>
            <div class="designer-badges">{badges}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_shell() -> None:
    inject_app_theme()
    render_model_hero()


def render_section_header(title: str, caption: str | None = None) -> None:
    from .. import app as legacy

    body = f"<h3>{title}</h3>"
    if caption:
        body += f'<p class="pharma-section-caption">{caption}</p>'
    legacy.st.markdown(
        f'<section class="pharma-section-card">{body}</section>',
        unsafe_allow_html=True,
    )

