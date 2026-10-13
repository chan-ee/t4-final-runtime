"""Narrow organizer authorization for practice-derived constants in every Final task.

This never grants later runtime evidence, other future training data or another
neural checkpoint. Only the reviewed public-practice artifact gets this scope.
"""
import hashlib
import os

POLICY='official-practice-final-6029140058'
URL='https://github.com/Agenthon-2026/track4-analysis-public/issues/24#issuecomment-6029140058'
APPROVED_SHA='0e58319cb96d0e88271aca3e9083a6499cfd4f89747a7ebd31f97fe3ea4a4a57'


def practice_final_allowed(raw):
    return (os.environ.get('EVIDENCE_FINAL_PRACTICE_SCOPE')==POLICY
            and hashlib.sha256(raw).hexdigest()==APPROVED_SHA)


def decision_exception(decision):
    return (decision.get('kind')=='statistical_practice_anchor'
            and decision.get('final_practice_scope')==POLICY
            and decision.get('model_sha256')==APPROVED_SHA
            and os.environ.get('EVIDENCE_FINAL_PRACTICE_SCOPE')==POLICY)
