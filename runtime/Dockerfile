FROM python@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26
LABEL qfbench2.interface_version="2.0" \
      org.opencontainers.image.title="Evidence Inference Final Native Contract Candidate" \
      org.opencontainers.image.description="Explicit same-response inferential links; unseen-family performance unmeasured" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/evidence EVIDENCE_PROTOCOL=single \
    EVIDENCE_REQUIRE_VERIFIED_HOUSE=0 EVIDENCE_SINGLE_REQUIRE_MODEL_DELIVERY=0 EVIDENCE_ACCOUNTING_CONTRACT=0 EVIDENCE_REASONING_MODE=direct \
    EVIDENCE_PRACTICE_HEADS=1 \
    XDG_CACHE_HOME=/tmp/cache MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba \
    EVIDENCE_FINAL_PRACTICE_SCOPE=official-practice-final-6029140058
WORKDIR /opt/evidence
COPY evidence_inference /opt/evidence/evidence_inference
COPY analyze /usr/local/bin/analyze
COPY ARTIFACT_PROVENANCE.md LICENSE /opt/evidence/
USER 65534:65534
CMD ["analyze", "--help"]
