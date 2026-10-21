
import json
from datetime import date
from uuid import uuid4

import streamlit as st
from langgraph.types import Command

from claim_core import create_claim_workflow


st.set_page_config(page_title="Insurance Claim Review")

st.title("Insurance Claim Review")
st.write("Test fictional claims using sample policy rules.")

api_key = st.sidebar.text_input(
    "OpenAI API key",
    type="password"
)

st.sidebar.write(
    "Sample coverage limit: ₹5,00,000. "
    "Claims above ₹1,00,000 need human review."
)

st.caption(
    "Use fictional details only. Claim details are sent to OpenAI "
    "to prepare the summary. Document names are checked, "
    "but document contents and authenticity are not verified."
)

with st.form("claim_form"):
    customer_name = st.text_input(
        "Sample customer name",
        value="Sample Customer"
    )

    claim_amount = st.number_input(
        "Claim amount (₹)",
        min_value=1.0,
        value=45000.0,
        step=1000.0
    )

    bill_amount = st.number_input(
        "Hospital bill amount (₹)",
        min_value=1.0,
        value=45000.0,
        step=1000.0
    )

    policy_start = st.date_input(
        "Policy start date",
        value=date(2026, 1, 1)
    )

    policy_end = st.date_input(
        "Policy end date",
        value=date(2026, 12, 31)
    )

    treatment_date = st.date_input(
        "Treatment date",
        value=date(2026, 6, 15)
    )

    documents = st.multiselect(
        "Documents listed with the claim",
        ["ID proof", "Claim form", "Hospital bill"],
        default=["ID proof", "Claim form", "Hospital bill"]
    )

    consent = st.checkbox(
        "I am using fictional details and agree to send "
        "the claim details to OpenAI."
    )

    submitted = st.form_submit_button("Check claim")


if submitted:
    if not api_key.strip():
        st.warning("Please enter your OpenAI API key.")

    elif not consent:
        st.warning("Please confirm the checkbox.")

    elif not customer_name.strip():
        st.warning("Please enter a sample customer name.")

    else:
        claim = {
            "claim_id": str(uuid4()),
            "customer_name": customer_name.strip(),
            "claim_amount": claim_amount,
            "bill_amount": bill_amount,
            "treatment_date": treatment_date.isoformat(),
            "policy_start": policy_start.isoformat(),
            "policy_end": policy_end.isoformat(),
            "documents": documents
        }

        st.session_state.pop("claim_result", None)
        st.session_state.pop("claim_workflow", None)
        st.session_state.pop("claim_config", None)

        try:
            workflow = create_claim_workflow(api_key.strip())
            config = {
                "configurable": {"thread_id": str(uuid4())}
            }

            with st.spinner("Checking the claim..."):
                result = workflow.invoke(claim, config=config)

            st.session_state["claim_workflow"] = workflow
            st.session_state["claim_config"] = config
            st.session_state["claim_result"] = result

        except Exception:
            st.error(
                "The claim could not be processed. Check your API key, "
                "API balance, and internet connection, then try again."
            )


result = st.session_state.get("claim_result")

if result:
    st.subheader("Claim results")

    st.write("**Claim ID:**", result["claim_id"])
    st.write("**Claim amount:**", f"₹{result['claim_amount']:,.2f}")
    st.write("**Hospital bill:**", f"₹{result['bill_amount']:,.2f}")

    st.write(
        "**All required documents listed:**",
        "Yes" if result["documents_valid"] else "No"
    )

    if result["missing_documents"]:
        st.write(
            "**Missing documents:**",
            ", ".join(result["missing_documents"])
        )

    st.write("**Eligibility check:**", result["eligibility_reason"])

    for warning in result["warning_signs"]:
        st.warning(warning)

    st.write("**Summary:**", result["claim_summary"])

    if result.get("__interrupt__"):
        st.info("Paused for human review.")

        with st.form("review_form"):
            st.write("Act as the reviewer for this fictional test.")

            reviewer_decision = st.selectbox(
                "Reviewer decision",
                ["approve", "reject"]
            )

            reviewer_note = st.text_area("Reason for the decision")
            review_submitted = st.form_submit_button("Submit review")

        if review_submitted:
            if not reviewer_note.strip():
                st.warning("Please provide a reason.")

            else:
                try:
                    with st.spinner("Saving the review..."):
                        reviewed_result = st.session_state[
                            "claim_workflow"
                        ].invoke(
                            Command(resume={
                                "decision": reviewer_decision,
                                "note": reviewer_note.strip()
                            }),
                            config=st.session_state["claim_config"]
                        )

                    st.session_state["claim_result"] = reviewed_result
                    st.rerun()

                except Exception:
                    st.error(
                        "The review could not be saved. Please try again."
                    )

    else:
        st.write("**Decision:**", result["decision"])
        st.write("**Reason:**", result["decision_reason"])

    download_result = {
        key: value
        for key, value in result.items()
        if key != "__interrupt__"
    }

    download_result["status"] = (
        "Paused for human review"
        if result.get("__interrupt__")
        else result["decision"]
    )

    st.download_button(
        "Download claim result",
        data=json.dumps(
            download_result,
            indent=2,
            ensure_ascii=False
        ),
        file_name="claim_result.json",
        mime="application/json"
    )

st.caption(
    "This is a student demonstration using fictional rules. "
    "Warning signs do not prove fraud. Pending reviews are kept "
    "only while the app session and server remain active."
)
