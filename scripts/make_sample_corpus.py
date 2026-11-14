"""Generate a small, realistic enterprise PDF corpus.

Four documents, 13 pages total, full of specific dates, numbers and policy
details. That specificity is the point: it makes citations checkable by hand and
gives the eval harness objective answers to grade against.

    python scripts/make_sample_corpus.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "data" / "docs"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, spaceAfter=14, leading=24)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=8)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=9, textColor="#666666")


DOCUMENTS: dict[str, dict] = {
    "Northwind Employee Handbook": {
        "subtitle": "People Operations · Version 7.2 · Effective 1 March 2026",
        "pages": [
            [
                ("h2", "1. About Northwind"),
                (
                    "p",
                    "Northwind Analytics is a data infrastructure company headquartered in Berlin with "
                    "regional offices in Lisbon and Austin. At the time of writing the company employs "
                    "412 people across 14 countries. This handbook describes the policies that apply to "
                    "all permanent employees unless a local addendum states otherwise.",
                ),
                ("h2", "2. Employment and probation"),
                (
                    "p",
                    "All new employees serve a probation period of 90 calendar days. During probation "
                    "either party may terminate the contract with two weeks of written notice. After "
                    "probation the statutory notice period applies, which is three months to the end of "
                    "a calendar month for employees with more than two years of service.",
                ),
                (
                    "p",
                    "Performance reviews take place twice a year, in April and October. Compensation is "
                    "reviewed annually in January, and the review considers both the April and October "
                    "ratings.",
                ),
            ],
            [
                ("h2", "3. Paid leave"),
                (
                    "p",
                    "Full-time employees receive 24 days of paid annual leave per calendar year, in "
                    "addition to public holidays. Leave accrues monthly at a rate of two days per month. "
                    "Employees may carry over a maximum of five unused days into the following year, and "
                    "carried-over days must be used before 31 March or they are forfeited.",
                ),
                ("h2", "4. Sick leave and family leave"),
                (
                    "p",
                    "Employees receive 10 paid sick days per year. Absences longer than three consecutive "
                    "days require a medical certificate submitted to People Operations within five working "
                    "days. Sick days do not carry over.",
                ),
                (
                    "p",
                    "Parental leave is 16 weeks fully paid for the primary caregiver and 6 weeks fully "
                    "paid for the secondary caregiver. Parental leave may be split into a maximum of three "
                    "blocks and must be taken within 12 months of the child's birth or adoption.",
                ),
                ("h2", "5. Public holidays"),
                (
                    "p",
                    "Northwind observes the public holidays of the employee's country of employment. In "
                    "Germany this currently means nine national holidays plus the state holidays of the "
                    "employee's registered workplace.",
                ),
            ],
            [
                ("h2", "6. Hybrid working"),
                (
                    "p",
                    "Northwind operates a hybrid model. Employees are expected to be in the office a "
                    "minimum of three days per week, Tuesday through Thursday. The remaining days may be "
                    "worked remotely, including from another country for up to 20 working days per year, "
                    "subject to tax clearance.",
                ),
                (
                    "p",
                    "Core collaboration hours are 10:00 to 16:00 Central European Time. Meetings should be "
                    "scheduled inside this window. Employees in the Austin office may offset core hours by "
                    "one hour with manager approval.",
                ),
                ("h2", "7. Equipment"),
                (
                    "p",
                    "Every employee receives a laptop budget of EUR 2,500 refreshed every three years, plus "
                    "an annual home-office stipend of EUR 500. Equipment purchased with company funds "
                    "remains company property and must be returned within 14 days of the last working day.",
                ),
            ],
            [
                ("h2", "8. Business expenses"),
                (
                    "p",
                    "Expense claims must be submitted within 30 days of the expense being incurred. Claims "
                    "submitted after 90 days will not be reimbursed. Itemised receipts are mandatory for "
                    "any single expense above EUR 25.",
                ),
                (
                    "p",
                    "Travel class depends on flight duration: economy for flights under six hours, premium "
                    "economy for flights between six and nine hours, and business class for flights longer "
                    "than nine hours. Hotel spending is capped at EUR 220 per night in tier-one cities and "
                    "EUR 150 per night elsewhere.",
                ),
                ("h2", "9. Approval matrix"),
                (
                    "p",
                    "Expenses up to EUR 500 require line-manager approval. Expenses between EUR 500 and EUR "
                    "5,000 require director approval. Anything above EUR 5,000 requires CFO approval and a "
                    "written business case.",
                ),
            ],
        ],
    },
    "Northwind Security Policy": {
        "subtitle": "Information Security · Version 4.0 · Classification: Internal",
        "pages": [
            [
                ("h2", "1. Scope"),
                (
                    "p",
                    "This policy applies to all employees, contractors and third parties who access "
                    "Northwind systems or data. It is owned by the Chief Information Security Officer and "
                    "reviewed every 12 months.",
                ),
                ("h2", "2. Data classification"),
                (
                    "p",
                    "Northwind recognises four classification levels. Public data may be shared freely. "
                    "Internal data is limited to employees. Confidential data is limited to named teams on "
                    "a need-to-know basis. Restricted data covers personal data, credentials and financial "
                    "records, and requires explicit approval from the Data Protection Officer before any "
                    "transfer outside the EU.",
                ),
                (
                    "p",
                    "Confidential and Restricted data must never be stored on personal devices, in personal "
                    "cloud accounts, or pasted into third-party AI tools that are not on the approved vendor "
                    "list.",
                ),
            ],
            [
                ("h2", "3. Authentication"),
                (
                    "p",
                    "Multi-factor authentication is mandatory for every system that stores Internal data or "
                    "above. Passwords must be at least 14 characters and are checked against a breach "
                    "corpus on every change. Passwords must be rotated every 180 days, and reuse of the "
                    "previous five passwords is blocked.",
                ),
                (
                    "p",
                    "Shared accounts are prohibited. Service accounts must be owned by a named individual, "
                    "documented in the service catalogue, and rotated every 90 days.",
                ),
                ("h2", "4. Access reviews"),
                (
                    "p",
                    "Access rights are reviewed quarterly by system owners. Accounts that have not "
                    "authenticated for 60 days are automatically disabled. Elevated production access is "
                    "granted for a maximum of eight hours and expires automatically.",
                ),
            ],
            [
                ("h2", "5. Incident response"),
                (
                    "p",
                    "Any suspected security incident must be reported to security@northwind.example within "
                    "one hour of discovery. Employees must not attempt to investigate an incident "
                    "themselves; the on-call security engineer coordinates the response.",
                ),
                (
                    "p",
                    "Incidents are classified using a four-level severity scale. Severity 1 covers active "
                    "data breach or full service compromise and requires a response within 15 minutes. "
                    "Severity 4 covers policy violations with no data impact and is handled within five "
                    "business days.",
                ),
                ("h2", "6. Breach notification"),
                (
                    "p",
                    "Where a personal data breach is confirmed, Northwind notifies the supervisory authority "
                    "within 72 hours of becoming aware of it, in line with the GDPR. Affected data subjects "
                    "are notified without undue delay where the breach is likely to result in a high risk "
                    "to their rights and freedoms.",
                ),
                (
                    "p",
                    "A post-incident review is mandatory for every Severity 1 and Severity 2 incident and "
                    "must be completed within 10 business days.",
                ),
            ],
        ],
    },
    "Atlas Platform API Guide": {
        "subtitle": "Developer Documentation · API v2 · Last updated 9 February 2026",
        "pages": [
            [
                ("h2", "1. Overview"),
                (
                    "p",
                    "The Atlas Platform API is a REST interface for ingesting documents, running extraction "
                    "jobs and retrieving structured results. All requests use HTTPS and JSON. The base URL "
                    "is https://api.atlas.example/v2.",
                ),
                ("h2", "2. Authentication"),
                (
                    "p",
                    "The API uses the OAuth 2.0 client-credentials flow. Exchange your client id and secret "
                    "at the /oauth/token endpoint to receive a bearer token. Access tokens expire after "
                    "3,600 seconds and refresh tokens are not issued; simply request a new token.",
                ),
                (
                    "p",
                    "Send the token in the Authorization header as a Bearer value. Requests with an expired "
                    "token return HTTP 401 with the error code token_expired.",
                ),
            ],
            [
                ("h2", "3. Rate limits"),
                (
                    "p",
                    "Standard plans allow 1,000 requests per minute per API key. Enterprise plans allow "
                    "5,000 requests per minute. Bursts of up to twice the per-minute limit are tolerated "
                    "for 10 seconds.",
                ),
                (
                    "p",
                    "When a limit is exceeded the API returns HTTP 429 with a Retry-After header indicating "
                    "the number of seconds to wait. Clients must implement exponential backoff with jitter "
                    "and must not retry more than five times.",
                ),
                ("h2", "4. Pagination"),
                (
                    "p",
                    "List endpoints use cursor pagination. Pass the cursor parameter returned in the "
                    "previous response. Page sizes default to 50 and may be set up to a maximum of 200 "
                    "records per page.",
                ),
            ],
            [
                ("h2", "5. Core endpoints"),
                (
                    "p",
                    "POST /documents uploads a document and returns a job identifier. Supported formats are "
                    "PDF, DOCX, PNG and JPEG, with a maximum file size of 100 MB. GET /documents/{id}/status "
                    "returns one of queued, processing, completed or failed.",
                ),
                (
                    "p",
                    "GET /documents/{id}/extractions returns the structured extraction output. Results are "
                    "retained for 90 days on standard plans and 365 days on enterprise plans, after which "
                    "they are permanently deleted.",
                ),
                ("h2", "6. Idempotency"),
                (
                    "p",
                    "All POST requests accept an Idempotency-Key header. Keys are retained for 24 hours. "
                    "Replaying a request with the same key within that window returns the original response "
                    "instead of creating a duplicate job.",
                ),
            ],
            [
                ("h2", "7. Error codes"),
                (
                    "p",
                    "The API returns standard HTTP status codes with a machine-readable error code. "
                    "invalid_request (400) indicates a malformed payload. unauthorized (401) indicates a "
                    "missing or invalid token. forbidden (403) indicates insufficient scopes. "
                    "not_found (404) indicates an unknown resource. payload_too_large (413) indicates a file "
                    "over 100 MB. rate_limited (429) indicates throttling.",
                ),
                (
                    "p",
                    "Server errors use internal_error (500) and service_unavailable (503). Retries are safe "
                    "for GET requests and for POST requests that include an Idempotency-Key.",
                ),
                ("h2", "8. Webhooks"),
                (
                    "p",
                    "Webhook deliveries are signed with HMAC-SHA256 using the secret issued in the console. "
                    "Failed deliveries are retried for 24 hours with exponential backoff, starting 30 "
                    "seconds after the first failure. Deliveries are considered failed on any non-2xx "
                    "response or a timeout of 10 seconds.",
                ),
            ],
        ],
    },
    "Atlas Product Brief": {
        "subtitle": "Product Marketing · Q1 2026 · Internal",
        "pages": [
            [
                ("h2", "Positioning"),
                (
                    "p",
                    "Atlas is a document intelligence platform for regulated industries. It extracts "
                    "structured data from contracts, invoices and identity documents, and produces an audit "
                    "trail for every field it populates. The primary buyers are operations leaders in "
                    "insurance, banking and logistics.",
                ),
                ("h2", "Pricing"),
                (
                    "p",
                    "There are three plans. Starter costs USD 49 per user per month and includes 2,000 "
                    "document pages per month. Growth costs USD 199 per user per month and includes 25,000 "
                    "pages plus priority support. Enterprise pricing is custom and includes SSO, private "
                    "deployment and a dedicated solutions engineer.",
                ),
            ],
            [
                ("h2", "Service level agreement"),
                (
                    "p",
                    "Enterprise customers are covered by a 99.95 percent monthly uptime commitment. If "
                    "uptime falls below that threshold, customers receive service credits starting at 10 "
                    "percent of the monthly fee and scaling to 50 percent for uptime below 99.0 percent.",
                ),
                ("h2", "Support"),
                (
                    "p",
                    "Starter customers receive email support with a target first response of two business "
                    "days. Growth customers receive a four-hour response during business hours. Enterprise "
                    "customers receive a one-hour response, 24 hours a day, seven days a week, plus a named "
                    "technical account manager.",
                ),
                ("h2", "Roadmap highlights"),
                (
                    "p",
                    "Q2 2026 delivers multi-language extraction for Spanish and Portuguese. Q3 introduces a "
                    "human-in-the-loop review queue with field-level confidence thresholds. Q4 targets "
                    "on-premise deployment for two anchor customers in the banking segment.",
                ),
            ],
        ],
    },
    "Northwind Data Retention Policy": {
        "subtitle": "Legal & Compliance · Version 2.1 · Classification: Internal",
        "pages": [
            [
                ("h2", "1. Purpose"),
                (
                    "p",
                    "This policy defines how long Northwind retains business records and how they are "
                    "destroyed. It applies to all systems of record, including the CRM, the support desk, "
                    "the data warehouse and internal collaboration tools.",
                ),
                ("h2", "2. Principles"),
                (
                    "p",
                    "Records are retained only as long as there is a documented business or legal reason. "
                    "Retention is applied automatically by system configuration rather than by manual "
                    "review. Legal hold overrides every retention rule and must be approved by the General "
                    "Counsel.",
                ),
            ],
            [
                ("h2", "3. Retention schedule"),
                (
                    "p",
                    "Support tickets and their attachments are retained for 24 months. Invoices, tax records "
                    "and contracts are retained for 10 years. Job applicant CVs are deleted after 6 months "
                    "unless the candidate opts in to the talent pool, in which case they are kept for 24 "
                    "months. Access logs are retained for 12 months. Internal chat messages are retained "
                    "for 90 days. Meeting recordings are retained for 30 days.",
                ),
                (
                    "p",
                    "Production database backups are encrypted at rest and rotated on a 35-day cycle. "
                    "Backups are stored in two regions within the EU and are never restored outside a "
                    "controlled incident process.",
                ),
            ],
            [
                ("h2", "4. Deletion requests"),
                (
                    "p",
                    "Where a data subject exercises the right to erasure, the Data Protection Officer logs "
                    "the request and the responsible system owner completes deletion within 30 days. "
                    "Deletion is confirmed in writing, and the confirmation is itself retained for three "
                    "years as evidence of compliance.",
                ),
                ("h2", "5. Vendor records"),
                (
                    "p",
                    "Records held by processors must be deleted within 60 days of contract termination. "
                    "Exit certificates are required from every vendor that handles Confidential or "
                    "Restricted data.",
                ),
            ],
        ],
    },
    "Northwind Sales Playbook": {
        "subtitle": "Revenue Operations · FY26 · Internal",
        "pages": [
            [
                ("h2", "1. Ideal customer profile"),
                (
                    "p",
                    "Northwind sells to mid-market and enterprise analytics teams with more than 50 data "
                    "practitioners. The economic buyer is usually a VP of Data or a Chief Data Officer. "
                    "Deals under USD 25,000 in annual contract value are handled by the inside sales team.",
                ),
                ("h2", "2. Qualification"),
                (
                    "p",
                    "Opportunities are qualified with MEDDIC. A deal may not enter the proposal stage "
                    "without an identified economic buyer, a documented decision criteria and a target "
                    "close date within the current or next quarter.",
                ),
            ],
            [
                ("h2", "3. Discounting authority"),
                (
                    "p",
                    "Account executives may approve discounts up to 15 percent. The regional VP of Sales "
                    "may approve up to 25 percent. Anything beyond 25 percent requires the Chief Revenue "
                    "Officer. Multi-year commitments of three years or more unlock an additional 5 percent "
                    "discount without further approval.",
                ),
                (
                    "p",
                    "Discounts against list price are recorded in the CRM at the line-item level. Deals "
                    "without a recorded discount justification are rejected by deal desk.",
                ),
            ],
            [
                ("h2", "4. Deal stages"),
                (
                    "p",
                    "The pipeline has six stages: discovery, validation, business case, proposal, "
                    "negotiation and closed. A deal may remain in any single stage for a maximum of 45 "
                    "days before it is flagged for review. Stalled deals are reviewed in the weekly "
                    "forecast call.",
                ),
                ("h2", "5. Proof of value"),
                (
                    "p",
                    "Standard proof-of-value engagements run for 30 days with a defined success metric "
                    "agreed in writing before the environment is provisioned. Extensions beyond 30 days "
                    "require director approval.",
                ),
            ],
        ],
    },
    "Atlas Customer Support Handbook": {
        "subtitle": "Customer Experience · Version 3.4 · Internal",
        "pages": [
            [
                ("h2", "1. Support channels"),
                (
                    "p",
                    "Customers reach Atlas through in-app chat, email and, for enterprise accounts, a "
                    "dedicated Slack Connect channel. Phone support is available to enterprise customers "
                    "only. Chat is staffed 06:00 to 22:00 Central European Time on weekdays.",
                ),
                ("h2", "2. Response targets"),
                (
                    "p",
                    "Targets are measured from the moment a ticket is created, not from first reply. "
                    "Starter accounts have a two business day target. Growth accounts have a four hour "
                    "target during business hours. Enterprise accounts have a one hour target around the "
                    "clock.",
                ),
            ],
            [
                ("h2", "3. Severity matrix"),
                (
                    "p",
                    "Severity 1 means the customer cannot process documents at all; the target first "
                    "response is 30 minutes and the ticket is escalated to the on-call engineer "
                    "immediately. Severity 2 means a major feature is degraded; first response within 4 "
                    "hours. Severity 3 means a minor feature issue; response within one business day. "
                    "Severity 4 covers questions and cosmetic issues; response within three business days.",
                ),
                ("h2", "4. Escalation"),
                (
                    "p",
                    "Any Severity 1 ticket must be escalated to the on-call engineer within 10 minutes of "
                    "creation and to the engineering manager within 30 minutes. Customers are updated "
                    "every 60 minutes until the ticket is resolved.",
                ),
            ],
            [
                ("h2", "5. Credits and refunds"),
                (
                    "p",
                    "Service credits are issued automatically when monthly uptime falls below the "
                    "contractual threshold. Credits are applied to the next invoice and never paid out in "
                    "cash. Refunds for annual plans are available within 30 days of purchase and are "
                    "prorated for unused whole months.",
                ),
                ("h2", "6. Account health"),
                (
                    "p",
                    "Accounts with fewer than five active users for 21 consecutive days are flagged as "
                    "at-risk and routed to the customer success team for an outreach sequence.",
                ),
            ],
        ],
    },
    "Northwind Vendor Onboarding Guide": {
        "subtitle": "Procurement & Security · Version 1.8 · Internal",
        "pages": [
            [
                ("h2", "1. Overview"),
                (
                    "p",
                    "Every third party that processes Northwind data must complete onboarding before a "
                    "purchase order is issued. Vendors are grouped into three tiers based on the "
                    "classification of data they touch and their level of system access.",
                ),
                ("h2", "2. Tier definitions"),
                (
                    "p",
                    "Tier 3 vendors handle Public data only and require a signed order form. Tier 2 "
                    "vendors handle Internal data and require a security questionnaire plus a signed data "
                    "processing agreement. Tier 1 vendors handle Confidential or Restricted data and "
                    "require a full assessment, including evidence of SOC 2 Type II and an annual "
                    "penetration test.",
                ),
            ],
            [
                ("h2", "3. Security assessment"),
                (
                    "p",
                    "The security questionnaire contains 140 controls across access management, "
                    "encryption, logging, business continuity and sub-processor management. Assessments "
                    "are valid for 12 months, after which a refresh is required. A failed assessment may "
                    "be remediated within 60 days at the vendor's cost.",
                ),
                (
                    "p",
                    "Vendors must notify Northwind of any security incident affecting Northwind data "
                    "within 24 hours of discovery, which is stricter than the 72 hour regulatory window.",
                ),
            ],
            [
                ("h2", "4. Contract terms"),
                (
                    "p",
                    "Standard contracts include a 30 day termination for convenience clause, a 90 day "
                    "notice period for price increases, and a cap on liability of 12 months of fees. "
                    "Vendors must maintain cyber liability insurance of at least EUR 5 million.",
                ),
                ("h2", "5. Renewal"),
                (
                    "p",
                    "Contracts are reviewed 90 days before expiry. Auto-renewal is disabled by default and "
                    "must be explicitly approved by the budget owner.",
                ),
            ],
        ],
    },
}


def build(name: str, spec: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name.replace(' ', '_')}.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        title=name,
        author="Northwind Analytics",
    )

    story = [Paragraph(name, H1), Paragraph(spec["subtitle"], SMALL), Spacer(1, 8)]
    for page_index, blocks in enumerate(spec["pages"], start=1):
        for kind, text in blocks:
            story.append(Paragraph(text, H2 if kind == "h2" else BODY))
        if page_index < len(spec["pages"]):
            story.append(PageBreak())
    doc.build(story)
    return path


def main() -> None:
    created = [build(name, spec) for name, spec in DOCUMENTS.items()]
    for path in created:
        print(f"  created {path.relative_to(OUT.parent.parent)}")
    print(f"\n{len(created)} documents written to {OUT}")


if __name__ == "__main__":
    main()
