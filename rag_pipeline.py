import ollama

from retriever import retrieve_evidence
from external_evidence import retrieve_external_evidence
from claim_verifier import verify_answer_claims


# ============================================================
# TRUTHGUARD END-TO-END PIPELINE
# ============================================================


# ============================================================
# 1. GENERATE ANSWER
# ============================================================

def generate_answer(question, evidence):

    evidence_text = "\n\n".join(
        item["text"]
        for item in evidence
    )

    prompt = f"""
You are TruthGuard AI.

Answer EVERY part of the user's question.

Use ONLY the evidence provided below.

Do not use outside knowledge.
Do not invent information.
Do not skip any part of the question.

If the evidence does not answer a part of the question,
explicitly say:

"The provided evidence is insufficient to answer this part."

IMPORTANT:
If the question contains multiple parts, answer each part separately.

EVIDENCE:
{evidence_text}

QUESTION:
{question}

Give a clear answer covering every part of the question.
"""

    response = ollama.chat(
        model="llama3:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response[
        "message"
    ][
        "content"
    ].strip()


# ============================================================
# 2. PRINT EVIDENCE
# ============================================================

def print_evidence(evidence):

    for i, item in enumerate(
        evidence,
        start=1
    ):

        print(
            f"\n## Evidence {i}"
        )

        print(
            "\nText       : "
            + item["text"]
        )

        print(
            "Source     : "
            + item.get(
                "source",
                "Unknown"
            )
        )

        print(
            "Document   : "
            + item.get(
                "document",
                "Unknown"
            )
        )

        if "chunk_id" in item:

            print(
                "Chunk ID    : "
                + str(
                    item["chunk_id"]
                )
            )

        if "relevance" in item:

            print(
                "Relevance  : "
                f"{item['relevance'] * 100:.2f}%"
            )

        if item.get("url"):

            print(
                "URL        : "
                + item["url"]
            )


# ============================================================
# 3. TRUTHGUARD QUERY
# ============================================================

def truthguard_query(question):

    print(
        "\nRetrieving local evidence..."
    )

    local_evidence = retrieve_evidence(
        question,
        top_k=3
    )


    # ========================================================
    # LOCAL EVIDENCE
    # ========================================================

    if local_evidence:

        evidence = local_evidence

        evidence_type = "LOCAL"


    # ========================================================
    # EXTERNAL EVIDENCE
    # ========================================================

    else:

        print(
            "No sufficiently relevant local evidence found."
        )

        print(
            "Searching external evidence..."
        )

        external_evidence = (
            retrieve_external_evidence(
                question,
                limit=3
            )
        )


        if not external_evidence:

            return {

                "question":
                    question,

                "answer":
                    (
                        "The available evidence is "
                        "insufficient to answer the question."
                    ),

                "evidence":
                    [],

                "evidence_type":
                    "NONE",

                "verification":
                    {
                        "claims": [],

                        "overall":
                            {
                                "overall_relation":
                                    "INSUFFICIENT",

                                "supported":
                                    0,

                                "contradicted":
                                    0,

                                "insufficient":
                                    1,

                                "support_percentage":
                                    0.0
                            }
                    }
            }


        evidence = external_evidence

        evidence_type = "EXTERNAL"


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    print(
        "Generating answer..."
    )

    answer = generate_answer(
        question,
        evidence
    )


    # ========================================================
    # CLAIM-LEVEL VERIFICATION
    # ========================================================

    print(
        "Running claim-level TruthGuard verification..."
    )

    verification = verify_answer_claims(
        answer,
        evidence
    )


    return {

        "question":
            question,

        "answer":
            answer,

        "evidence":
            evidence,

        "evidence_type":
            evidence_type,

        "verification":
            verification
    }


# ============================================================
# 4. TERMINAL APPLICATION
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=========================================="
    )

    print(
        "       TRUTHGUARD END-TO-END ENGINE"
    )

    print(
        "=========================================="
    )


    question = input(
        "\nEnter your question: "
    )


    result = truthguard_query(
        question
    )


    # ========================================================
    # QUESTION
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "QUESTION"
    )

    print(
        "=========================================="
    )

    print(
        result["question"]
    )


    # ========================================================
    # ANSWER
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "TRUTHGUARD ANSWER"
    )

    print(
        "=========================================="
    )

    print(
        result["answer"]
    )


    # ========================================================
    # EVIDENCE
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "EVIDENCE"
    )

    print(
        "=========================================="
    )


    if not result["evidence"]:

        print(
            "\nNo evidence found."
        )

    else:

        print(
            "\nEvidence Type: "
            + result["evidence_type"]
        )

        print_evidence(
            result["evidence"]
        )


    # ========================================================
    # CLAIM VERIFICATION
    # ========================================================

    verification = result[
        "verification"
    ]


    print(
        "\n=========================================="
    )

    print(
        "TRUTHGUARD CLAIM VERIFICATION"
    )

    print(
        "=========================================="
    )


    for claim in verification[
        "claims"
    ]:

        print()

        print(
            f"Claim {claim['claim_id']}:"
        )

        print(
            "Text       : "
            + claim["claim"]
        )

        print(
            "Subject    : "
            + str(
                claim.get(
                    "subject"
                )
            )
        )

        print(
            "Relation   : "
            + claim["relation"]
        )

        print(
            "Similarity : "
            f"{claim['semantic_similarity'] * 100:.2f}%"
        )

        print(
            "Confidence : "
            f"{claim['confidence']:.2f}%"
        )


        if claim["evidence"]:

            print(
                "Evidence   : "
                + claim[
                    "evidence"
                ]["text"]
            )

            print(
                "Source     : "
                + claim[
                    "evidence"
                ].get(
                    "source",
                    "Unknown"
                )
            )


        print(
            "Explanation: "
            + claim[
                "explanation"
            ]
        )

        print(
            "---"
        )


    # ========================================================
    # OVERALL RESULT
    # ========================================================

    overall = verification[
        "overall"
    ]


    print(
        "\n=========================================="
    )

    print(
        "OVERALL CLAIM VERIFICATION"
    )

    print(
        "=========================================="
    )


    print(
        "\nOverall Relation : "
        + overall[
            "overall_relation"
        ]
    )

    print(
        "Supported       : "
        + str(
            overall[
                "supported"
            ]
        )
    )

    print(
        "Contradicted    : "
        + str(
            overall[
                "contradicted"
            ]
        )
    )

    print(
        "Insufficient    : "
        + str(
            overall[
                "insufficient"
            ]
        )
    )

    print(
        "Support         : "
        f"{overall['support_percentage']:.2f}%"
    )


    print(
        "\n=========================================="
    )